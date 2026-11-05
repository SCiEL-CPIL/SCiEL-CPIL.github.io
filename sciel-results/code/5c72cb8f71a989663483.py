import itertools
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # Precompute
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Build sparse incidence matrix
    rows = []
    cols = []
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            rows.append(i)
            cols.append(j)
    
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_tsubs, num_kblocks))
    A_csc = A.tocsc()
    
    block_tsub_lists = [None] * num_kblocks
    for j in range(num_kblocks):
        start = A_csc.indptr[j]
        end = A_csc.indptr[j + 1]
        block_tsub_lists[j] = A_csc.indices[start:end].tolist()
    
    # Convert A to dense for fast vectorized operations (num_tsubs * num_kblocks)
    # For C(19,7,4): 3876 * 50388 ≈ 195M entries, too large for dense
    # Use sparse matrix-vector product instead
    
    # Weighted greedy with NumPy vectorization
    # Score = gain / (1 + overlap_with_selected) to encourage diversity
    selected = []
    uncovered = np.ones(num_tsubs, dtype=np.int8)
    
    # Precompute block gains (number of t-subsets each block covers)
    block_gains = np.array([len(block_tsub_lists[j]) for j in range(num_kblocks)], dtype=np.int32)
    
    greedy_deadline = deadline - 55.0
    
    # First few iterations: pure greedy (maximize gain)
    # Later: weighted greedy to break ties and improve quality
    max_greedy_iters = 200
    iter_count = 0
    
    while np.any(uncovered) and iter_count < max_greedy_iters:
        if time.monotonic() > greedy_deadline:
            break
        
        # Compute gains for all blocks against uncovered
        # gains[j] = sum of A[:,j] * uncovered
        gains = A.T.dot(uncovered).astype(np.int32)
        
        if iter_count < 10:
            # Pure greedy for first few steps
            best_block = int(np.argmax(gains))
        else:
            # Weighted: score = gain / (1 + 0.1 * selected_overlap)
            # Compute overlap with already selected blocks
            if selected:
                sel_mask = np.zeros(num_kblocks, dtype=np.int32)
                for sj in selected:
                    sel_mask[sj] = 1
                # Approximate overlap: how many t-subs in block j are also in selected blocks
                # This is expensive, so use a simpler heuristic:
                # score = gain * (1 + 0.05 * block_gains[j] / max_gain)
                max_gain = gains.max() if gains.max() > 0 else 1
                scores = gains.astype(np.float64) * (1.0 + 0.05 * block_gains.astype(np.float64) / max_gain)
            else:
                scores = gains.astype(np.float64)
            
            best_block = int(np.argmax(scores))
        
        if gains[best_block] <= 0:
            break
        
        selected.append(best_block)
        uncovered -= A[:, best_block].toarray().ravel().astype(np.int8)
        uncovered = np.maximum(uncovered, 0)
        iter_count += 1
    
    # If greedy didn't cover everything, fall back to bitmask-based greedy for remainder
    if np.any(uncovered):
        # Build bitmask for remaining
        remaining_tsubs = np.where(uncovered > 0)[0]
        if len(remaining_tsubs) > 0:
            # Map remaining tsubs to compact indices
            rem_to_compact = {ts: i for i, ts in enumerate(remaining_tsubs)}
            num_rem = len(remaining_tsubs)
            
            rem_masks = [0] * num_kblocks
            for j in range(num_kblocks):
                mask = 0
                for ti in block_tsub_lists[j]:
                    if ti in rem_to_compact:
                        mask |= (1 << rem_to_compact[ti])
                rem_masks[j] = mask
            
            full_rem_mask = (1 << num_rem) - 1
            unc_rem = full_rem_mask
            while unc_rem:
                if time.monotonic() > deadline - 50.0:
                    break
                best_gain = 0
                best_block = -1
                for j in range(num_kblocks):
                    if j in selected:
                        continue
                    common = rem_masks[j] & unc_rem
                    if common:
                        gain = common.bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_block = j
                if best_block == -1:
                    break
                selected.append(best_block)
                unc_rem &= ~rem_masks[best_block]
    
    # Local search: try to replace pairs of blocks with one better block
    local_deadline = deadline - 45.0
    if selected and time.monotonic() < local_deadline:
        # Compute coverage counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for j in selected:
            for ti in block_tsub_lists[j]:
                counts[ti] += 1
        
        # Try removing blocks that are redundant
        changed = True
        while changed and time.monotonic() < local_deadline:
            changed = False
            new_sel = []
            for j in selected:
                red = True
                for ti in block_tsub_lists[j]:
                    if counts[ti] == 1:
                        red = False
                        break
                if red:
                    for ti in block_tsub_lists[j]:
                        counts[ti] -= 1
                    changed = True
                else:
                    new_sel.append(j)
            selected = new_sel
        
        # Try to swap blocks: remove one, add a better one
        swap_deadline = deadline - 40.0
        if time.monotonic() < swap_deadline:
            # Recompute counts after pruning
            counts = np.zeros(num_tsubs, dtype=np.int32)
            for j in selected:
                for ti in block_tsub_lists[j]:
                    counts[ti] += 1
            
            # For each selected block, see if removing it and adding a different block helps
            # We want to minimize total blocks, so try: remove block j, find if remaining is still covered
            # If not covered, the uncovered tsubs must be covered by some other block
            # Try swapping: for each pair (j1, j2) in selected, can we replace with one block?
            # This is O(n^2) which is too slow. Instead, try single removals.
            
            # Try adding a block that covers many currently-singly-covered tsubs
            # and then removing a block that becomes redundant
            for attempt in range(3):
                if time.monotonic() > swap_deadline:
                    break
                
                # Find tsubs covered exactly once
                singly_covered = np.where(counts == 1)[0]
                if len(singly_covered) == 0:
                    break
                
                # For each non-selected block, compute how many singly-covered tsubs it covers
                gains_single = np.zeros(num_kblocks, dtype=np.int32)
                for j in range(num_kblocks):
                    if j in selected:
                        continue
                    for ti in block_tsub_lists[j]:
                        if counts[ti] == 1:
                            gains_single[j] += 1
                
                best_add = int(np.argmax(gains_single))
                if gains_single[best_add] > 0:
                    # Add this block
                    new_counts = counts.copy()
                    for ti in block_tsub_lists[best_add]:
                        new_counts[ti] += 1
                    
                    # Now try to remove a block that becomes redundant
                    removed = False
                    for j in list(selected):
                        red = True
                        for ti in block_tsub_lists[j]:
                            if new_counts[ti] <= 1:
                                red = False
                                break
                        if red:
                            for ti in block_tsub_lists[j]:
                                new_counts[ti] -= 1
                            selected.remove(j)
                            selected.append(best_add)
                            counts = new_counts
                            removed = True
                            break
                    if not removed:
                        # Revert
                        pass
    
    # ILP refinement
    remaining = deadline - time.monotonic() - 5.0
    if remaining > 10.0 and selected:
        c = np.ones(num_kblocks)
        constraint = LinearConstraint(A, lb=1, ub=np.inf)
        bound = Bounds(lb=0, ub=1)
        x0 = np.zeros(num_kblocks)
        for j in selected:
            x0[j] = 1.0
        try:
            result = milp(c=c, constraints=constraint, integrality=np.ones(num_kblocks),
                          bounds=bound, x0=x0,
                          options={"time_limit": remaining, "mip_rel_gap": 0.0})
            if result.success and result.x is not None:
                ilp_sel = [j for j in range(num_kblocks) if result.x[j] > 0.5]
                if len(ilp_sel) <= len(selected):
                    cov = np.zeros(num_tsubs, dtype=np.int8)
                    for j in ilp_sel:
                        for ti in block_tsub_lists[j]:
                            cov[ti] += 1
                    if cov.min() >= 1:
                        selected = ilp_sel
        except Exception:
            pass
    
    # Final pruning pass
    if selected:
        counts = np.zeros(num_tsubs, dtype=np.int8)
        for j in selected:
            for ti in block_tsub_lists[j]:
                counts[ti] += 1
        changed = True
        while changed:
            if time.monotonic() > deadline - 1.0:
                break
            changed = False
            new_sel = []
            for j in selected:
                red = True
                for ti in block_tsub_lists[j]:
                    if counts[ti] == 1:
                        red = False
                        break
                if red:
                    for ti in block_tsub_lists[j]:
                        counts[ti] -= 1
                    changed = True
                else:
                    new_sel.append(j)
            selected = new_sel
    
    return [list(all_kblocks[j]) for j in selected]
