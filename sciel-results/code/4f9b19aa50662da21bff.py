import numpy as np
import itertools
import time
import random
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # 1. Precomputation
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each block
    block_t_indices = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])
        
    # Precompute inverse: triple -> list of blocks
    triple_to_blocks = [[] for _ in range(n_ts)]
    for bi, indices in enumerate(block_t_indices):
        for ti in indices:
            triple_to_blocks[ti].append(bi)

    # 2. Greedy Construction
    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        gain = np.array([len(indices) for indices in block_t_indices], dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() - start_time > time_limit - 10:
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(best_idx)
            
            subs = block_t_indices[best_idx]
            for ti in subs:
                if not covered[ti]:
                    covered[ti] = True
                    for other_bi in triple_to_blocks[ti]:
                        gain[other_bi] -= 1
            
            if np.all(covered):
                break
        
        return blocks

    # 3. Local Search
    def local_search_remove(blocks_list):
        current = list(blocks_list)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    def local_search_2for1(blocks_list, time_budget):
        """
        Try to replace 2 blocks with 1 block.
        """
        current = list(blocks_list)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            if time.time() - start_time > time_limit - 5:
                break
                
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            n_blocks = len(current)
            found = False
            
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            max_pairs_checked = 1000
            pairs_checked = 0
            
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    if pairs_checked >= max_pairs_checked:
                        break
                    if time.time() > deadline:
                        break
                    
                    a = current[indices[i_idx]]
                    b = current[indices[j_idx]]
                    
                    subs_a = block_t_indices[a]
                    subs_b = block_t_indices[b]
                    
                    unique_a = [s for s in subs_a if coverage_count[s] == 1]
                    unique_b = [s for s in subs_b if coverage_count[s] == 1]
                    
                    if not unique_a and not unique_b:
                        continue
                    
                    req_set = set(unique_a) | set(unique_b)
                    
                    if not req_set:
                        continue
                    
                    first_tri = next(iter(req_set))
                    candidates = triple_to_blocks[first_tri]
                    
                    best_c = -1
                    for c in candidates:
                        if c == a or c == b:
                            continue
                        valid = True
                        req_list = list(req_set)
                        c_subs = block_t_indices[c]
                        for tri in req_list:
                            if tri not in c_subs:
                                valid = False
                                break
                        if valid:
                            best_c = c
                            break
                    
                    if best_c != -1:
                        new_current = []
                        for idx in current:
                            if idx == a or idx == b:
                                continue
                            new_current.append(idx)
                        new_current.append(best_c)
                        
                        current = new_current
                        found = True
                        break
                        
                if found:
                    break
                    
            if not found:
                break
                
        return current

    # 4. ILP-based exact polish: find minimum subset of candidate blocks that covers all triples
    def ilp_min_cover(candidate_block_indices, time_budget=30.0):
        """
        Given a list of candidate block indices, find the minimum subset
        that covers all t-subsets using ILP.
        Returns the optimal subset if found within time, else None.
        """
        if time.time() - start_time > time_limit - 5:
            return None
        
        deadline = time.time() + time_budget
        n_candidates = len(candidate_block_indices)
        
        if n_candidates == 0:
            return []
        
        # Build constraint matrix: A[i,j] = 1 if block j covers triple i
        # A has shape (n_ts, n_candidates)
        # For each candidate block j, get its triple indices
        A_rows = []
        A_cols = []
        A_vals = []
        
        for j, b_idx in enumerate(candidate_block_indices):
            for ti in block_t_indices[b_idx]:
                A_rows.append(ti)
                A_cols.append(j)
                A_vals.append(1)
        
        if not A_rows:
            return []
        
        # Use scipy's milp with sparse matrix
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_ts, n_candidates))
        
        # Objective: minimize sum of x_j
        c = np.ones(n_candidates)
        
        # Constraints: A @ x >= 1 (each triple covered at least once)
        # scipy milp uses: lb <= A @ x <= ub
        # We need A @ x >= 1, so: A @ x >= 1
        # This is: -A @ x <= -1, or in scipy form: A_ub @ x <= b_ub
        # Actually scipy milp uses LinearConstraint(lb, ub, A)
        # So we can do: LinearConstraint(1, np.inf, A_sparse)
        
        try:
            constraint = LinearConstraint(A_sparse, lb=1, ub=np.inf)
            bounds = Bounds(lb=0, ub=1)
            
            res = milp(
                c=c,
                constraints=constraint,
                integrality=np.ones(n_candidates),
                bounds=bounds,
                options={'time_limit': time_budget, 'mip_rel_gap': 0.0}
            )
            
            if time.time() > deadline:
                return None
            
            if res.status == 0:  # Optimal
                x = res.x
                selected = [candidate_block_indices[i] for i in range(n_candidates) if x[i] > 0.5]
                return selected
            elif res.status == 1:  # Feasible
                x = res.x
                selected = [candidate_block_indices[i] for i in range(n_candidates) if x[i] > 0.5]
                # Verify it's actually a valid cover
                cover_check = np.zeros(n_ts, dtype=bool)
                for b_idx in selected:
                    for ti in block_t_indices[b_idx]:
                        cover_check[ti] = True
                if np.all(cover_check):
                    return selected
                return None
            else:
                return None
        except Exception:
            return None

    def ilp_polish(blocks_list, time_budget=20.0):
        """
        Use ILP to find the minimum cover from the current blocks plus
        some additional candidate blocks that might improve the solution.
        """
        if time.time() - start_time > time_limit - 5:
            return list(blocks_list)
        
        current = list(blocks_list)
        n_current = len(current)
        
        # Start with just the current blocks
        result = ilp_min_cover(current, time_budget=min(10.0, time_budget))
        if result is not None:
            return result
        
        # Try adding some candidate blocks that might help
        # Find blocks that cover many uncovered triples if we remove some current blocks
        # For now, just try the current set with ILP
        return current

    # 5. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy + local search to get good initial solutions
    greedy_results = []
    for seed in range(15):
        if time.time() - start_time > time_limit - 30:
            break
            
        greedy_blocks = greedy(seed)
        refined = local_search_remove(greedy_blocks)
        
        if time.time() - start_time < time_limit - 25:
            remaining = time_limit - (time.time() - start_time) - 25
            if remaining > 5.0:
                refined = local_search_2for1(refined, min(3.0, remaining * 0.3))
            refined = local_search_remove(refined)
        
        count = len(refined)
        greedy_results.append((count, refined))
        
        if count < best_count:
            best_count = count
            best_blocks = refined
    
    # Phase 2: ILP polish on the best solutions
    # Sort by count, try ILP on the top few
    greedy_results.sort(key=lambda x: x[0])
    
    for count, blocks in greedy_results[:5]:
        if time.time() - start_time > time_limit - 10:
            break
        
        if count >= best_count and best_blocks is not None:
            # Only try if it might improve
            if count > best_count:
                continue
        
        ilp_result = ilp_polish(blocks, time_budget=min(15.0, time_limit - (time.time() - start_time) - 5))
        
        if ilp_result is not None:
            # Verify validity
            cover_check = np.zeros(n_ts, dtype=bool)
            for b_idx in ilp_result:
                for ti in block_t_indices[b_idx]:
                    cover_check[ti] = True
            
            if np.all(cover_check):
                new_count = len(ilp_result)
                if new_count < best_count:
                    best_count = new_count
                    best_blocks = ilp_result
    
    # Phase 3: Final local search on the best ILP result
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        final = local_search_remove(best_blocks)
        if len(final) < best_count:
            best_count = len(final)
            best_blocks = final
    
    if best_blocks is None:
        return []

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
