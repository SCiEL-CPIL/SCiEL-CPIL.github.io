import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence data
    # block_tsubs: list of t-subset indices for each block
    # tsub_blocks: list of block indices for each t-subset
    block_tsubs = []
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            tsubs.append(idx)
            tsub_blocks[idx].append(bi)
        block_tsubs.append(tsubs)

    best_blocks = None
    best_len = float('inf')

    def time_left():
        return deadline - time.time()

    def remove_redundant(blocks):
        """Remove blocks that do not cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        return [blocks[i] for i in range(len(blocks)) if kept[i]]

    def local_search_1for1(blocks):
        """Try to replace a block with another that covers its unique subsets."""
        if not blocks:
            return blocks
        
        blocks = remove_redundant(blocks)
        
        # Precompute t-subsets for each block in the current solution
        sol_tsubs = []
        for b in blocks:
            tsubs = set()
            for ts in itertools.combinations(b, t):
                tsubs.add(tsub_index[ts])
            sol_tsubs.append(tsubs)
        
        # Count coverage
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for tsubs in sol_tsubs:
            for idx in tsubs:
                cover_count[idx] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove redundant first
            new_blocks = remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                improved = True
                # Recalculate sol_tsubs and cover_count if needed, but loop restarts
                continue
            
            # Try 1-for-1 swap
            # For each block i, find its unique t-subsets
            # If we can find a block j not in solution that covers all unique t-subsets of i,
            # and j covers at least as many t-subsets as i (or just to change structure),
            # we might swap. 
            # A better heuristic: If block i has unique subsets U_i, and there exists block j 
            # such that U_i subset of T_j, and T_j subset of T_i is false (j covers more or different),
            # swapping might not reduce count immediately but could help later.
            # However, a direct count reduction is:
            # If we swap i with j, and j covers all unique subsets of i, the count stays same.
            # But if j also covers subsets that were previously covered only by other blocks,
            # it might allow removing those other blocks.
            
            # Let's try a simpler aggressive move:
            # If we can find a block j that covers ALL t-subsets covered by block i (T_i subset of T_j),
            # then we can remove i. But T_i subset of T_j implies i is redundant if j is in solution.
            # This is handled by remove_redundant.
            
            # Let's try: Can we replace a set of 2 blocks with 1 block?
            # This is expensive.
            
            # Let's stick to: Try to swap block i with block j if j covers the unique subsets of i,
            # AND j is "better" in some sense (e.g., covers more total subsets, or has higher LP weight).
            # For now, let's just do random restarts of the construction with perturbed weights.
            pass
        
        return blocks

    def lp_guided_greedy(duals, seed=0):
        rng = random.Random(seed)
        covered = np.zeros(n_tsubsets, dtype=bool)
        picked = np.zeros(n_blocks, dtype=bool)
        blocks = []
        
        # Initialize weights based on duals
        block_weights = np.zeros(n_blocks)
        for j in range(n_blocks):
            if block_tsubs[j]:
                block_weights[j] = np.sum(duals[block_tsubs[j]])
                
        # Heap of (-weight, index)
        heap = [(-block_weights[j], j) for j in range(n_blocks) if block_weights[j] > 1e-9]
        heapq.heapify(heap)
        
        while heap and time_left() > 5:
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            # Check if weight is stale (should have decreased)
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                continue
            
            # Check if it covers any uncovered subset
            covers_uncovered = False
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covers_uncovered = True
                    break
            if not covers_uncovered:
                continue
                
            picked[j] = True
            blocks.append(list(all_blocks[j]))
            
            # Update weights
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in tsub_blocks[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-9:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
            
            if not np.any(~covered):
                break
        
        # Fallback for any remaining uncovered
        if not np.all(covered):
            # Simple greedy for remaining
            rem_uncovered = np.where(~covered)[0]
            # Convert to set of indices for fast lookup? 
            # We can just run a standard greedy on the remaining uncovered subsets
            # But we need to be careful not to pick blocks already picked? 
            # Actually, we can pick any block.
            
            # Re-calculate uncovered status
            uncovered_set = set(rem_uncovered)
            
            while uncovered_set and time_left() > 5:
                best_block = None
                best_count = -1
                # Sample a subset of blocks to speed up? Or scan all?
                # Scanning 21k blocks * 35 subsets is fast.
                for j in range(n_blocks):
                    count = 0
                    for s_idx in block_tsubs[j]:
                        if s_idx in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_block = j
                    # Early break if we found a block covering all? Unlikely.
                
                if best_block is None or best_count == 0:
                    break
                    
                blocks.append(list(all_blocks[best_block]))
                for s_idx in block_tsubs[best_block]:
                    if s_idx in uncovered_set:
                        uncovered_set.remove(s_idx)
                        
        return blocks

    # Solve LP to get duals
    duals = None
    try:
        # Build sparse matrix A for LP: M.T @ x >= 1 => -M.T @ x <= -1
        rows = []
        cols = []
        data = []
        for j in range(n_blocks):
            for s_idx in block_tsubs[j]:
                rows.append(s_idx)
                cols.append(j)
                data.append(1.0)
        
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
        
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # res.ineqlin.marginals are for A_ub @ x <= b_ub
            # The constraint is -M.T @ x <= -1
            # Dual variable y >= 0 for M @ y >= 1 (primal min, dual max)
            # In linprog, marginals are d(Obj)/d(b). 
            # Obj = c^T x. Constraint -M^T x <= -1.
            # If we relax -1 to -1+delta, obj changes by marginal * delta.
            # Standard duality: c^T x = y^T 1.
            # The marginal for constraint i is the shadow price.
            # For min problem with <= constraints, marginals are <= 0.
            # y_i = -marginal_i.
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
            # Normalize?
            # duals = duals / (np.sum(duals) + 1e-9)
            
    except Exception as e:
        # Fallback to uniform weights if LP fails
        duals = np.ones(n_tsubsets)

    # Run multiple trials with different seeds and perturbations
    num_trials = 5
    for i in range(num_trials):
        if time_left() < 10:
            break
            
        # Perturb duals slightly to break symmetry
        if i > 0:
            perturbed_duals = duals * (1 + 0.1 * np.random.random(n_tsubsets))
        else:
            perturbed_duals = duals
            
        blocks = lp_guided_greedy(perturbed_duals, seed=i)
        
        if blocks:
            blocks = remove_redundant(blocks)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
        # If we have a very good solution, stop early
        if best_len <= 140:
            break

    if best_blocks is None:
        # Final fallback: standard greedy
        duals = np.ones(n_tsubsets)
        best_blocks = lp_guided_greedy(duals, seed=0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
