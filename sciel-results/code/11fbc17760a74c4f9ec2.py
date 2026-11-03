import itertools
import time
import random
import heapq
import numpy as np
from collections import defaultdict
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute block_tsubs and tsub_blocks
    block_tsubs = []
    tsub_blocks = [[] for _ in range(n_t)]
    for b in all_blocks:
        tsubs_in_b = [t_sub_to_idx[tsub] for tsub in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        for tsub_idx in tsubs_in_b:
            tsub_blocks[tsub_idx].append(len(block_tsubs) - 1)
    
    def time_ok():
        return time.time() < deadline
    
    def get_lp_duals():
        """Solve LP relaxation to get dual weights for t-subsets."""
        try:
            # Build incidence matrix M (n_t x n_blocks)
            rows, cols, data = [], [], []
            for j in range(n_blocks):
                for i in block_tsubs[j]:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)
            
            M = csr_matrix((data, (rows, cols)), shape=(n_t, n_blocks))
            
            c = np.ones(n_blocks)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_blocks
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-8})
            
            if res.success and res.ineqlin.marginals is not None:
                duals = -res.ineqlin.marginals
                duals = np.maximum(duals, 0)
                # Normalize
                if np.sum(duals) > 0:
                    duals = duals / np.sum(duals) * n_t
                else:
                    duals = np.ones(n_t) / n_t
            else:
                duals = np.ones(n_t) / n_t
        except Exception:
            duals = np.ones(n_t) / n_t
        
        return duals
    
    def greedy_cover(duals, seed=0):
        """LP-guided greedy with random tie-breaking."""
        rng = random.Random(seed)
        
        # Initialize block weights from duals
        block_weights = np.zeros(n_blocks)
        for j in range(n_blocks):
            if block_tsubs[j]:
                block_weights[j] = np.sum(duals[block_tsubs[j]])
        
        # Heap of (-weight, block_id)
        heap = [(-block_weights[j], j) for j in range(n_blocks)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_blocks, dtype=bool)
        blocks = []
        uncovered_count = n_t
        
        while heap and uncovered_count > 0:
            if not time_ok():
                break
            
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                continue
            
            # Collect all blocks with same weight for random tie-breaking
            candidates = [j]
            while heap:
                neg_w2, j2 = heapq.heappop(heap)
                w2 = -neg_w2
                if picked[j2] or w2 > block_weights[j2] + 1e-9 or w2 <= 1e-9:
                    continue
                if abs(w2 - w) < 1e-9:
                    candidates.append(j2)
                else:
                    heapq.heappush(heap, (neg_w2, j2))
                    break
            
            # Pick random candidate
            j = candidates[rng.randint(0, len(candidates) - 1)]
            picked[j] = True
            blocks.append(j)
            
            # Update coverage and weights
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    uncovered_count -= 1
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in tsub_blocks[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))
        
        return blocks
    
    def local_search(sol_blocks):
        """Local search: remove redundant blocks, try replacements."""
        if not sol_blocks:
            return sol_blocks
        
        sol_blocks = list(sol_blocks)
        used_set = set(sol_blocks)
        
        # Build cover_map: tsub_idx -> count of blocks in solution covering it
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_id in sol_blocks:
            for tsub_idx in block_tsubs[b_id]:
                cover_count[tsub_idx] += 1
        
        def is_redundant(b_id):
            for tsub_idx in block_tsubs[b_id]:
                if cover_count[tsub_idx] == 1:
                    return False
            return True
        
        def remove_block(b_id):
            sol_blocks.remove(b_id)
            used_set.discard(b_id)
            for tsub_idx in block_tsubs[b_id]:
                cover_count[tsub_idx] -= 1
        
        def add_block(b_id):
            sol_blocks.append(b_id)
            used_set.add(b_id)
            for tsub_idx in block_tsubs[b_id]:
                cover_count[tsub_idx] += 1
        
        # Pass 1: Remove all redundant blocks
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(sol_blocks) - 1, -1, -1):
                if is_redundant(sol_blocks[i]):
                    remove_block(sol_blocks[i])
                    changed = True
        
        # Pass 2: Try to replace blocks to enable further reduction
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(sol_blocks)):
                if not time_ok():
                    break
                b_id = sol_blocks[i]
                
                # Find unique t-subsets for b_id
                unique_tsubs = []
                for tsub_idx in block_tsubs[b_id]:
                    if cover_count[tsub_idx] == 1:
                        unique_tsubs.append(tsub_idx)
                
                if not unique_tsubs:
                    remove_block(b_id)
                    improved = True
                    break
                
                # Try to find a replacement block
                best_new_block = -1
                best_overlap = -1
                
                # For efficiency, only check blocks that cover the first unique tsub
                candidate_blocks = tsub_blocks[unique_tsubs[0]]
                
                for new_b_id in candidate_blocks:
                    if new_b_id == b_id or new_b_id in used_set:
                        continue
                    
                    # Check if new_b_id covers all unique_tsubs
                    valid = True
                    overlap = 0
                    for tsub_idx in unique_tsubs:
                        if tsub_idx not in block_tsubs[new_b_id]:
                            valid = False
                            break
                        # Count how many t-subsets in new_b_id are currently covered only by b_id
                        if cover_count[tsub_idx] == 1:
                            overlap += 1
                    if not valid:
                        continue
                    
                    # Also count how many t-subsets in new_b_id are covered by other blocks
                    # (potential to make other blocks redundant)
                    for tsub_idx in block_tsubs[new_b_id]:
                        if tsub_idx not in set(unique_tsubs) and cover_count[tsub_idx] >= 1:
                            overlap += 0.5  # Small bonus for overlapping with covered tsubs
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_new_block = new_b_id
                    
                    if best_overlap > 5:  # Good enough, break early
                        break
                
                if best_new_block != -1:
                    remove_block(b_id)
                    add_block(best_new_block)
                    improved = True
                    break
        
        # Final validation
        final_cover = np.zeros(n_t, dtype=np.int32)
        for b_id in sol_blocks:
            for tsub_idx in block_tsubs[b_id]:
                final_cover[tsub_idx] += 1
        
        if np.any(final_cover == 0):
            # Fallback: return empty to signal failure
            return []
        
        return sol_blocks
    
    # Main: Try multiple seeds with LP-guided greedy, then local search
    duals = get_lp_duals()
    
    best_solution = None
    best_len = float('inf')
    
    # Try several seeds for diversity
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng = random.Random(42)
    rng.shuffle(seeds)
    
    for seed in seeds:
        if not time_ok():
            break
        
        greedy_blocks = greedy_cover(duals, seed)
        
        if not greedy_blocks:
            continue
        
        # Check if complete
        covered = np.zeros(n_t, dtype=bool)
        for b_id in greedy_blocks:
            for tsub_idx in block_tsubs[b_id]:
                covered[tsub_idx] = True
        
        if not np.all(covered):
            # Fallback: add blocks for remaining uncovered
            remaining = np.where(~covered)[0]
            for r_idx in remaining:
                for b_id in tsub_blocks[r_idx]:
                    if b_id not in greedy_blocks:
                        greedy_blocks.append(b_id)
                        for tsub_idx in block_tsubs[b_id]:
                            covered[tsub_idx] = True
                        break
        
        # Local search
        ls_blocks = local_search(greedy_blocks)
        
        if ls_blocks and len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_solution = ls_blocks
        
        # Early stop if we reach a good solution
        if best_len <= 155:
            break
    
    if best_solution is None or len(best_solution) == 0:
        # Emergency fallback: simple greedy
        covered = np.zeros(n_t, dtype=bool)
        blocks = []
        uncovered_count = n_t
        while uncovered_count > 0 and time_ok():
            best_b = -1
            best_gain = -1
            for b_id in range(n_blocks):
                if b_id in set(blocks):
                    continue
                gain = sum(1 for ts in block_tsubs[b_id] if not covered[ts])
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_id
            if best_b == -1 or best_gain == 0:
                break
            blocks.append(best_b)
            for ts in block_tsubs[best_b]:
                if not covered[ts]:
                    covered[ts] = True
                    uncovered_count -= 1
        best_solution = blocks
    
    return [list(all_blocks[i]) for i in best_solution]
