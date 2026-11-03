import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, csc_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    # Also build sparse matrix for LP
    block_tsubs = [None] * n_blocks
    rows = []
    cols = []
    data = []
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        for ts_idx in tsubs:
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    # Sparse matrix M where M[ts_idx, bi] = 1 if block bi covers ts_idx
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # Precompute t-subset sets for all blocks (for use in local search)
    block_tsub_sets = [set(block_tsubs[i]) for i in range(n_blocks)]
    
    # 2. LP Relaxation to get dual weights
    # Minimize sum(x_j)
    # Subject to M.T @ x >= 1
    # Which is -M.T @ x <= -1
    c = np.ones(n_blocks)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_tsubsets)
    bounds = [(0, 1)] * n_blocks
    
    dual_weights = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Dual variables correspond to the inequality constraints
            # res.ineqlin.marginals are the duals for A_ub @ x <= b_ub
            # The original constraint was M.T @ x >= 1
            # Dual y_i >= 0. The marginal is typically -y_i for minimization with <= constraints.
            # Let's verify signs. If we minimize c^T x s.t. Ax >= b, dual is max b^T y s.t. A^T y <= c, y >= 0.
            # Here we converted to -M.T x <= -1.
            # Dual of min c^T x s.t. Ax <= b is max b^T y s.t. A^T y >= -c, y <= 0? No.
            # Standard form: min c^T x s.t. Ax = b, x >= 0.
            # HiGHS marginals for inequality constraints in `linprog` (method='highs'):
            # For A_ub @ x <= b_ub, the marginal is the change in objective per unit increase in b_ub.
            # If we relax b_ub (make it less negative, i.e., larger), the constraint becomes looser.
            # Actually, let's just use the absolute value or check if they are positive.
            # In many solvers, the dual for >= constraint is positive.
            # Let's assume the marginals returned are the duals y_i.
            # If they are negative, we take abs. If positive, we use them.
            dual_weights = np.abs(res.ineqlin.marginals)
            # Ensure they are non-negative
            dual_weights[dual_weights < 0] = 0
            # If all are zero, fallback to uniform weights
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
    except Exception:
        dual_weights = np.ones(n_tsubsets)
        
    # 3. LP-Guided Greedy Initialization (Heap-based lazy update)
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        # Current weight of each block: sum of dual_weights of uncovered t-subsets in it
        block_weights = np.array([np.sum(dual_weights[block_tsubs[i]]) for i in range(n_blocks)], dtype=np.float64)
        
        # Build max-heap: (-weight, block_index)
        heap = [(-block_weights[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        selected_blocks = []
        selected_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Pop from heap until we find a valid (unselected, up-to-date) block
            valid_block = None
            while heap:
                neg_w, bi = heapq.heappop(heap)
                if bi in selected_set:
                    continue
                # Check if weight is up-to-date (recompute if needed)
                current_w = np.sum(dual_weights[block_tsubs[bi][uncovered[block_tsubs[bi]]]])
                if abs(-neg_w - current_w) < 1e-6:
                    valid_block = (bi, current_w)
                    break
                else:
                    # Weight changed, push updated value
                    heapq.heappush(heap, (-current_w, bi))
            
            if valid_block is None:
                break
            
            bi, w = valid_block
            if w <= 1e-9:
                break
            
            selected_blocks.append(bi)
            selected_set.add(bi)
            
            # Update uncovered and weights
            ts_list = block_tsubs[bi]
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                # Update weights for blocks covering these t-subsets
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    block_indices = M[ts_idx, :].indices
                    for b_idx in block_indices:
                        if b_idx not in selected_set:
                            block_weights[b_idx] -= w_sub
                            block_weights[b_idx] = max(block_weights[b_idx], 0)
                            heapq.heappush(heap, (-block_weights[b_idx], b_idx))
                    
        return selected_blocks

    def walksat_refine(block_indices, seed=0):
        """WalkSAT-style stochastic local search to minimize block count."""
        rng = random.Random(seed)
        
        # Current state: set of selected block indices
        selected = set(block_indices)
        
        # Compute coverage: for each t-subset, count how many selected blocks cover it
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            # Vectorized increment
            cover_count[block_tsubs[bi]] += 1
        
        # Track best solution found
        best_selected = set(selected)
        best_count = len(selected)
        
        # Parameters
        p_remove = 0.5  # Probability of trying to remove a block
        max_steps = 50000  # Increased steps
        no_improve_limit = 5000  # Increased patience
        
        step = 0
        no_improve = 0
        
        while step < max_steps and time.time() < deadline - 2:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > no_improve_limit:
                    break
            
            if not selected:
                break
            
            # Convert to list for random access
            sel_list = list(selected)
            
            if rng.random() < p_remove:
                # Try to remove a random block
                bi = sel_list[rng.randint(0, len(sel_list)-1)]
                
                # Check if removing it leaves any t-subset uncovered
                ts_indices = block_tsubs[bi]
                
                # Faster check:
                min_count = np.min(cover_count[ts_indices])
                if min_count > 1:
                    # Safe to remove
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                else:
                    # WalkSAT move: try to swap or force remove?
                    # For now, skip.
                    pass
            else:
                # Try to swap: remove one block, add another
                # Pick a block that has unique coverage (hard to remove)
                bi_to_remove = sel_list[rng.randint(0, len(sel_list)-1)]
                
                ts_indices = block_tsubs[bi_to_remove]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    # Block is redundant, remove it
                    selected.discard(bi_to_remove)
                    cover_count[ts_indices] -= 1
                    continue
                
                # Try to find a replacement block that covers all unique t-subsets
                # Use precomputed sets for faster subset checking
                n_samples = 50  # Increased samples
                found_swap = False
                
                for _ in range(n_samples):
                    ci = rng.randint(0, n_blocks - 1)
                    if ci in selected:
                        continue
                    
                    # Check if ci covers all unique t-subsets using precomputed set
                    cand_set = block_tsub_sets[ci]
                    # Check if unique_tsubs is subset of cand_set
                    if all(ts in cand_set for ts in unique_tsubs):
                        # Perform swap
                        selected.discard(bi_to_remove)
                        cover_count[ts_indices] -= 1
                        
                        ci_ts_indices = block_tsubs[ci]
                        selected.add(ci)
                        cover_count[ci_ts_indices] += 1
                        
                        found_swap = True
                        break
                
                if not found_swap:
                    # WalkSAT: with some probability, make a random move
                    # that may temporarily worsen the solution
                    if rng.random() < 0.10:  # Increased escape probability
                        # Randomly add a block (to escape local minimum)
                        ci = rng.randint(0, n_blocks - 1)
                        if ci not in selected:
                            selected.add(ci)
                            cover_count[block_tsubs[ci]] += 1
        
        # Final polish: try 1-for-1 and 2-for-1 swaps
        for _ in range(5):  # Increased polish iterations
            if time.time() > deadline - 2:
                break
            improved = False
            for bi in list(selected):
                ts_indices = block_tsubs[bi]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    improved = True
                    continue
                
                # Try to find a replacement
                for ci in range(n_blocks):
                    if ci in selected:
                        continue
                    cand_set = block_tsub_sets[ci]
                    if all(ts in cand_set for ts in unique_tsubs):
                        selected.discard(bi)
                        cover_count[ts_indices] -= 1
                        ci_ts_indices = block_tsubs[ci]
                        selected.add(ci)
                        cover_count[ci_ts_indices] += 1
                        improved = True
                        break
                
                if len(selected) < best_count:
                    best_selected = set(selected)
                    best_count = len(selected)
            
            if not improved:
                break
        
        # Convert back to block tuples
        return [all_blocks[bi] for bi in best_selected]

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    num_restarts = 15  # Increased restarts
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Get initial solution from LP-Greedy
        init_blocks = lp_greedy(seed)
        if not init_blocks:
            continue
        
        # Refine with WalkSAT
        refined = walksat_refine(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
            
        # Early termination if we hit a good score
        if best_len <= 145:
             # Try to squeeze more
             pass
             
    if best_blocks is None:
        # Fallback: simple greedy with heap
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        gain = np.array([len(block_tsubs[i]) for i in range(n_blocks)], dtype=np.int32)
        heap = [(-gain[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        selected_set = set()
        
        while np.any(uncovered):
            valid_block = None
            while heap:
                neg_g, bi = heapq.heappop(heap)
                if bi in selected_set:
                    continue
                # Recompute gain
                ts_list = block_tsubs[bi]
                active = ts_list[uncovered[ts_list]]
                current_g = len(active)
                if abs(-neg_g - current_g) < 1:
                    valid_block = (bi, current_g)
                    break
                else:
                    heapq.heappush(heap, (-current_g, bi))
            
            if valid_block is None or valid_block[1] == 0:
                break
            
            bi, g = valid_block
            blocks_used.append(bi)
            selected_set.add(bi)
            
            ts_list = block_tsubs[bi]
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts = ts_list[active_mask]
                uncovered[active_ts] = False
                
                for ts_idx in active_ts:
                    block_indices = M[ts_idx, :].indices
                    for b_idx in block_indices:
                        if b_idx not in selected_set:
                            gain[b_idx] -= 1
                            gain[b_idx] = max(gain[b_idx], 0)
                            heapq.heappush(heap, (-gain[b_idx], b_idx))
        
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
