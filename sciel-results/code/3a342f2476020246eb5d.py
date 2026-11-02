"""Improved Greedy-Then-Exact Hybrid for covering design C(24,7,3) with Randomized Restarts."""
import itertools
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Construct a covering design using Randomized Greedy with Restarts, 
    Fast Redundancy Removal, and ILP Completion for residuals.
    """
    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-blocks and the indices of t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the indices of the t-subsets it covers
    # Shape: (n_blocks, C(k,t))
    c_k_t = len(list(itertools.combinations(range(k), t)))
    block_triple_indices = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = tsub_to_idx[ts]
            
    def greedy_pass(rng):
        """Run a single pass of randomized greedy set cover."""
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected_indices = []
        
        for _ in range(n_blocks):
            if not uncovered.any():
                break
            
            gains = uncovered[block_triple_indices].sum(axis=1)
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            pick_idx = candidates[rng.integers(len(candidates))]
            
            selected_indices.append(pick_idx)
            uncovered[block_triple_indices[pick_idx]] = False
            
        return selected_indices, uncovered

    def remove_redundant(block_indices):
        """Remove redundant blocks using cover counts."""
        if not block_indices:
            return []
            
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for idx in block_indices:
            cover_count[block_triple_indices[idx]] += 1
            
        final_indices = []
        changed = True
        current_indices = list(block_indices)
        
        while changed:
            changed = False
            new_indices = []
            # Rebuild cover_count for the current set
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for idx in current_indices:
                cover_count[block_triple_indices[idx]] += 1
                
            for idx in current_indices:
                if np.min(cover_count[block_triple_indices[idx]]) > 1:
                    changed = True
                else:
                    new_indices.append(idx)
                    
            current_indices = new_indices
            
        return current_indices

    def complete_with_ilp(block_indices, uncovered):
        """Complete the cover using ILP if there are uncovered triples."""
        if not uncovered.any():
            return block_indices
            
        uncovered_indices = np.where(~uncovered)[0]
        if len(uncovered_indices) == 0:
            return block_indices
            
        is_unc = np.zeros(n_tsubs, dtype=bool)
        is_unc[uncovered_indices] = True
        useful_mask = is_unc[block_triple_indices].any(axis=1)
        candidate_indices = np.where(useful_mask)[0]
        
        if len(candidate_indices) == 0:
            return block_indices
            
        n_candidates = len(candidate_indices)
        A = np.zeros((len(uncovered_indices), n_candidates), dtype=np.int8)
        unc_idx_to_row = {unc_index: i for i, unc_index in enumerate(uncovered_indices)}
        
        for col, block_idx in enumerate(candidate_indices):
            ts_indices = block_triple_indices[block_idx]
            for ts_idx in ts_indices:
                if ts_idx in unc_idx_to_row:
                    A[unc_idx_to_row[ts_idx], col] = 1
                    
        c = np.ones(n_candidates, dtype=float)
        constraint = LinearConstraint(A, lb=np.ones(len(uncovered_indices)), ub=np.inf)
        bounds = Bounds(lb=np.zeros(n_candidates), ub=np.ones(n_candidates))
        integrality = np.ones(n_candidates, dtype=np.int8)
        
        try:
            result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if result.status == 0 and result.x is not None:
                selected_candidates = np.where(result.x > 0.5)[0]
                for idx in selected_candidates:
                    block_indices.append(candidate_indices[idx])
        except Exception:
            pass
            
        return block_indices

    # Main Loop: Randomized Restarts
    best_blocks = None
    best_size = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Run 1 iteration to measure
    t0 = time.time()
    sel, unc = greedy_pass(rng)
    sel = remove_redundant(sel)
    t1 = time.time()
    iter_time = t1 - t0
    
    num_iters = int(110 / iter_time)
    num_iters = max(10, min(num_iters, 5000))
    
    for i in range(num_iters):
        sel, unc = greedy_pass(rng)
        # If greedy didn't cover everything (unlikely), use ILP
        if unc.any():
             sel = complete_with_ilp(sel, unc)
        sel = remove_redundant(sel)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_blocks = sel
            
        if time.time() - t0 > 110:
            break
            
    if best_blocks is None:
        sel, _ = greedy_pass(rng)
        best_blocks = remove_redundant(sel)
        
    return [list(all_blocks[i]) for i in best_blocks]
