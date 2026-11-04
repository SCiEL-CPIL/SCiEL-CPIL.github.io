import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block for faster access later
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Phase 1: Greedy Set Cover with incremental gain updates
    # Precompute gains: gains[i] = number of uncovered t-subsets covered by block i
    # Use integer counts for speed
    uncovered_count = np.ones(num_t_subsets, dtype=np.int32)  # 1 if uncovered, 0 if covered
    # gains[i] = sum of uncovered_count[j] for j in block i's t-subsets
    # We maintain this incrementally
    
    # Initial gains: for each block, count how many t-subsets it covers (all start uncovered)
    gains = np.zeros(num_blocks, dtype=np.int32)
    for i in range(num_blocks):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        covered = A.indices[start:end]
        gains[i] = np.sum(uncovered_count[covered])
    
    selected_block_indices = []
    greedy_deadline = deadline - 45.0  # Reserve time for ILP and cleanup
    
    while True:
        if time.time() > greedy_deadline:
            break
            
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = int(np.argmax(gains))
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        
        # Update: these t-subsets are now covered
        # For each other block, if it covers any of these, decrement its gain
        # We need to update gains for all blocks that share t-subsets with b_idx
        
        # Efficient approach: for each newly covered t-subset, find all blocks that cover it
        # and decrement their gain by 1 (if the t-subset was previously uncovered)
        # But we need to be careful: only decrement if the t-subset was actually uncovered before
        
        # Since we set uncovered_count to 0, we can use that
        # For each t-subset j in covered_indices:
        #   if uncovered_count[j] was 1 (now 0), then for each block i that covers j: gains[i] -= 1
        
        # Build reverse index: for each t-subset, which blocks cover it
        # This is the transpose of A
        # Let's precompute this as a list of arrays
        # Actually, let's just iterate over the covered t-subsets and use A's structure
        
        # For speed, let's precompute the column-to-rows mapping
        # col_to_blocks[j] = list of block indices that cover t-subset j
        # This is expensive to build, so let's do it once
        pass
        
        # Actually, let me restructure: build the reverse mapping once
        # This is the transpose of A stored as CSR
        A_T = A.T.tocsr()
        
        # Now for each newly covered t-subset, we can find all blocks covering it
        # and decrement their gain
        for j in covered_indices:
            if uncovered_count[j] == 1:
                uncovered_count[j] = 0
                # Find all blocks that cover t-subset j
                block_start = A_T.indptr[j]
                block_end = A_T.indptr[j + 1]
                blocks_covering_j = A_T.indices[block_start:block_end]
                gains[blocks_covering_j] -= 1
            # If uncovered_count[j] was already 0, no update needed
        
        if np.all(uncovered_count == 0):
            break
            
    # Phase 2: Local Search Deletion
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    current_best = len(selected_block_indices)
    
    # Phase 3: Try ILP refinement on the current solution
    # Use a restricted set of blocks to make ILP more tractable
    # Strategy: Use blocks that appear in the greedy solution + some random candidates
    
    if time.time() < deadline - 15.0:
        try:
            # Build a reduced problem: use selected blocks + a random sample of others
            # to keep the ILP size manageable
            greedy_set = set(selected_block_indices)
            num_greedy = len(greedy_set)
            
            # Add some random blocks to potentially find better solutions
            num_extra = min(50, num_blocks - num_greedy)
            if num_extra > 0:
                other_blocks = [i for i in range(num_blocks) if i not in greedy_set]
                random_extra = np.random.choice(len(other_blocks), size=num_extra, replace=False)
                extra_blocks = [other_blocks[i] for i in random_extra]
            else:
                extra_blocks = []
            
            candidate_blocks = list(greedy_set) + extra_blocks
            num_candidates = len(candidate_blocks)
            
            # Build sub-matrix for candidate blocks
            # A_sub is (num_candidates, num_t_subsets)
            A_sub = A[candidate_blocks, :]
            
            # Objective: minimize sum(x)
            c = np.ones(num_candidates)
            
            # Constraints: A_sub^T x >= 1
            A_sub_T = A_sub.T.tocsr()
            
            constraint = LinearConstraint(A_sub_T, lb=1, ub=np.inf)
            
            # Bounds: x in [0, 1]
            lb = np.zeros(num_candidates)
            ub = np.ones(num_candidates)
            bounds = Bounds(lb, ub)
            
            # Integrality: all variables are binary
            integrality = np.ones(num_candidates)
            
            # Run MILP with a time limit
            time_remaining = deadline - 15.0 - time.time()
            ilp_time_limit = max(5.0, min(20.0, time_remaining))
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                       options={'time_limit': ilp_time_limit})
            
            if res.status == 0 and res.x is not None:
                ilp_solution = res.x.astype(int)
                ilp_block_positions = np.where(ilp_solution == 1)[0]
                
                if len(ilp_block_positions) > 0:
                    ilp_blocks = [candidate_blocks[i] for i in ilp_block_positions]
                    
                    # Verify validity
                    cover_check = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in ilp_blocks:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_check[covered] += 1
                    
                    if np.all(cover_check >= 1) and len(ilp_blocks) < current_best:
                        selected_block_indices = ilp_blocks
                        current_best = len(selected_block_indices)
                        
        except Exception:
            pass
            
    # Phase 4: Final Deletion Pass on the best solution found
    # Use vectorized approach for speed
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            # A block is redundant if all its t-subsets are covered by other blocks
            # i.e., cover_counts[c] >= 2 for all c in covered
            if np.all(cover_counts[covered] >= 2):
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
    
    # Phase 5: Additional local search - try swapping blocks
    # For each block in the solution, try replacing it with a block not in the solution
    # that covers all the same critical t-subsets
    if time.time() < deadline - 5.0:
        # Identify critical t-subsets (covered exactly once)
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] += 1
        
        critical_mask = (cover_counts == 1)
        num_critical = np.sum(critical_mask)
        
        if num_critical < len(selected_block_indices):
            # Try to remove blocks that don't cover any critical t-subsets
            # and see if we can still maintain coverage
            for i in range(len(selected_block_indices) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                # Check if this block covers any critical t-subset
                if not np.any(critical_mask[covered]):
                    # Try removing it
                    cover_counts[covered] -= 1
                    if np.all(cover_counts >= 1):
                        selected_block_indices.pop(i)
                        # Update critical mask
                        critical_mask = (cover_counts == 1)
                    else:
                        # Restore
                        cover_counts[covered] += 1
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
