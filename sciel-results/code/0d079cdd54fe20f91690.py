import itertools
import time
import random
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
        
    # 3. LP-Guided Greedy Initialization
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        # Current weight of each block: sum of dual_weights of uncovered t-subsets in it
        # We can update this dynamically.
        # Initial weight for block j is sum(dual_weights[block_tsubs[j]])
        block_weights = np.array([np.sum(dual_weights[block_tsubs[i]]) for i in range(n_blocks)], dtype=np.float64)
        
        selected_blocks = []
        
        # To speed up finding max weight, we can just scan. n_blocks is ~50k.
        # Scanning 50k floats is fast in NumPy.
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find block with max weight among those not selected (implicitly handled by weight=0 if fully covered?)
            # Actually, if a block only covers already covered t-subsets, its weight should be 0.
            # We update weights by subtracting dual_weights of covered t-subsets.
            
            max_idx = np.argmax(block_weights)
            max_w = block_weights[max_idx]
            
            if max_w <= 1e-9:
                # No block covers any uncovered t-subset? Should not happen if valid.
                # Fallback: pick any block with max raw coverage
                break
            
            # Tie-breaking: if multiple blocks have same max weight, pick randomly
            # For speed, we might just pick the argmax. Randomness comes from seed in later steps or slight noise.
            # To add randomness: find all indices with weight close to max
            # This is expensive. Let's just add small random noise to weights initially?
            # Or just pick argmax. The LP weights are distinct enough usually.
            # Let's try to pick randomly among top 5% if we want diversity, but for now, greedy is fine.
            
            selected_blocks.append(max_idx)
            
            # Update uncovered and weights
            # For each t-subset in block max_idx:
            #   if it was uncovered:
            #     mark covered
            #     subtract dual_weights[ts] from all blocks containing ts
            #     This is expensive if done naively.
            
            # Optimization:
            # We have M (sparse).
            # We want to subtract dual_weights[uncovered_ts_in_block] from blocks that contain them.
            # Let ts_list = block_tsubs[max_idx]
            # active_ts = ts_list[uncovered[ts_list]]
            # For each ts in active_ts:
            #   w_to_sub = dual_weights[ts]
            #   blocks_containing_ts = M[ts, :].indices (from sparse matrix)
            #   block_weights[blocks_containing_ts] -= w_to_sub
            
            ts_list = block_tsubs[max_idx]
            # Get indices of t-subsets in this block that are currently uncovered
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                
                # Mark them as covered
                uncovered[active_ts_indices] = False
                
                # Update weights
                # We need to iterate over these t-subsets and update block weights
                # This is the bottleneck.
                # Alternative: Just recalculate weights for candidate blocks?
                # Or use the sparse matrix structure.
                
                # Let's use the precomputed list of blocks for each t-subset?
                # We didn't precompute `tsub_blocks` as lists of indices, but we have M.
                # M is CSR. M[ts, :] gives row ts.
                # M.tocsr()
                # Let's convert M to CSR once outside if not already.
                # It is CSR.
                
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    # Get blocks covering this t-subset
                    # M[ts_idx, :] is a sparse row.
                    # .indices gives column indices (block indices)
                    block_indices = M[ts_idx, :].indices
                    # Subtract w_sub from their weights
                    block_weights[block_indices] -= w_sub
                    
                    # Ensure non-negative (due to floating point errors)
                    # block_weights[block_indices] = np.maximum(block_weights[block_indices], 0)
                    # Doing max on array slice is fast.
                    block_weights[block_indices] = np.maximum(block_weights[block_indices], 0)
                    
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
        max_steps = 30000
        no_improve_limit = 3000
        
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
                n_samples = 30
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
                    if rng.random() < 0.05:
                        # Randomly add a block (to escape local minimum)
                        ci = rng.randint(0, n_blocks - 1)
                        if ci not in selected:
                            selected.add(ci)
                            cover_count[block_tsubs[ci]] += 1
        
        # Final polish: try 1-for-1 and 2-for-1 swaps
        for _ in range(3):
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
    num_restarts = 10
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
        # Fallback: simple greedy
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
             max_gain = max(gain)
             if max_gain == 0: break
             candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
             chosen = candidates[0]
             blocks_used.append(chosen)
             for ts_idx in block_tsubs[chosen]:
                 if ts_idx in uncovered:
                     uncovered.discard(ts_idx)
                     for bi in M[ts_idx, :].indices:
                         if gain[bi] > 0:
                             gain[bi] -= 1
             gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
