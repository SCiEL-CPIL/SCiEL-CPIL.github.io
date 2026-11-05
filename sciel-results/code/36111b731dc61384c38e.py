import itertools
import time
import heapq
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # Precompute t-subset indices for each block
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    # 3. Greedy with lazy evaluation using priority queue
    covered = np.zeros(num_tsubs, dtype=bool)
    x0 = np.zeros(num_kblocks, dtype=float)
    selected_blocks = []
    
    # Lazy evaluation: maintain a heap of (negative_gain, block_idx, version)
    # We recompute gain on-the-fly when popping
    # Initial gains: number of t-subsets in each block
    initial_gains = np.array([len(idxs) for idxs in block_tsub_idxs], dtype=np.int64)
    
    # Heap entries: (-gain, block_idx)
    # We'll use a "lazy" approach: push all blocks, and when we pop, recompute actual gain
    # If actual gain != pushed gain, skip and re-push with correct gain
    heap = [(-initial_gains[j], j) for j in range(num_kblocks)]
    heapq.heapify(heap)
    
    # Track which blocks are already selected
    in_solution = np.zeros(num_kblocks, dtype=bool)
    
    while time.monotonic() < deadline - 30.0:  # Leave more time for polish + ILP
        if not heap:
            break
        
        neg_gain, b_idx = heapq.heappop(heap)
        pushed_gain = -neg_gain
        
        if in_solution[b_idx]:
            continue
        
        # Recompute actual marginal gain
        actual_gain = 0
        for idx in block_tsub_idxs[b_idx]:
            if not covered[idx]:
                actual_gain += 1
        
        if actual_gain != pushed_gain:
            # Stale entry, re-push with correct gain
            if actual_gain > 0:
                heapq.heappush(heap, (-actual_gain, b_idx))
            continue
        
        if actual_gain == 0:
            break
        
        # Select this block
        selected_blocks.append(b_idx)
        x0[b_idx] = 1.0
        in_solution[b_idx] = True
        
        # Update covered
        for idx in block_tsub_idxs[b_idx]:
            covered[idx] = True
            
        if np.all(covered):
            break
            
    # 4. Polish (Remove redundant blocks) - multiple passes
    counts = np.zeros(num_tsubs, dtype=int)
    for b_idx in selected_blocks:
        for idx in block_tsub_idxs[b_idx]:
            counts[idx] += 1
            
    changed = True
    while changed and time.monotonic() < deadline - 25.0:
        changed = False
        for i in range(len(selected_blocks)):
            b_idx = selected_blocks[i]
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_blocks.pop(i)
                x0[b_idx] = 0.0
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
                break  # Restart scan since indices changed
                
    # 5. Local search: try swapping blocks
    # For each selected block, try replacing it with an unselected block that covers
    # at least all the t-subsets it covers (or better)
    if time.monotonic() < deadline - 20.0:
        # Build coverage sets for quick lookup
        # For each selected block, find t-subsets only covered by this block
        # Try to replace with blocks that cover those critical t-subsets
        
        # First, identify "critical" t-subsets for each block
        # A t-subset is critical for block b if it's only covered by b
        # If we can find another block that covers all critical t-subsets of b,
        # we might be able to swap
        
        # Simpler approach: try removing each block and see if coverage is maintained
        # If not, try to add a different block instead
        for trial in range(3):  # Multiple passes
            if time.monotonic() > deadline - 20.0:
                break
            improved = False
            for i in range(len(selected_blocks)):
                if time.monotonic() > deadline - 20.0:
                    break
                b_idx = selected_blocks[i]
                
                # Try removing b_idx
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                
                # Check if all still covered
                all_still_covered = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 0:
                        all_still_covered = False
                        break
                
                if all_still_covered:
                    # Can remove
                    selected_blocks.pop(i)
                    x0[b_idx] = 0.0
                    improved = True
                    break
                else:
                    # Restore counts
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] += 1
                    # Try to find a replacement block
                    # Find t-subsets that would be uncovered
                    uncovered = []
                    for idx in block_tsub_idxs[b_idx]:
                        if counts[idx] == 1:  # Only covered by b_idx
                            uncovered.append(idx)
                    
                    if uncovered:
                        # Find blocks that cover all uncovered t-subsets
                        # This is expensive, so limit search
                        best_replace = -1
                        best_gain = 0
                        for cand in range(num_kblocks):
                            if in_solution[cand]:
                                continue
                            # Check if cand covers all uncovered
                            cand_covers = True
                            for uidx in uncovered:
                                if A[uidx, cand] == 0:
                                    cand_covers = False
                                    break
                            if cand_covers:
                                # Calculate net gain: how many new t-subsets does cand cover
                                new_gain = 0
                                for idx in block_tsub_idxs[cand]:
                                    if not covered[idx] or counts[idx] == 0:
                                        new_gain += 1
                                if new_gain > best_gain:
                                    best_gain = new_gain
                                    best_replace = cand
                        if best_replace >= 0:
                            # Do the swap
                            for idx in block_tsub_idxs[b_idx]:
                                counts[idx] -= 1
                            for idx in block_tsub_idxs[best_replace]:
                                counts[idx] += 1
                            selected_blocks[i] = best_replace
                            x0[b_idx] = 0.0
                            x0[best_replace] = 1.0
                            in_solution[b_idx] = False
                            in_solution[best_replace] = True
                            improved = True
                            break
                    # Restore if we didn't swap
                    # (already restored above)
            
            if not improved:
                break
                
    # 6. ILP Optimization
    if time.monotonic() < deadline - 10.0:
        try:
            remaining_time = deadline - time.monotonic() - 5.0
            if remaining_time > 5.0:
                c = np.ones(num_kblocks)
                constraints = LinearConstraint(A, 1, np.inf)
                integrality = np.ones(num_kblocks)
                bounds = Bounds(0, 1)
                
                res = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    x0=x0,
                    time_limit=min(remaining_time, 30.0)
                )
                
                if res.x is not None:
                    x_sol = np.round(res.x)
                    if np.all((x_sol == 0) | (x_sol == 1)):
                        coverage = A @ x_sol
                        if np.all(coverage >= 1):
                            new_count = np.sum(x_sol)
                            old_count = len(selected_blocks)
                            if new_count < old_count:
                                selected_blocks = [i for i in range(num_kblocks) if x_sol[i] == 1]
                                
        except Exception:
            pass

    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
