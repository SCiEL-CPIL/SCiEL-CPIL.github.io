import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build incidence matrix A (sparse)
    # A[i, j] = 1 if t-subset i is contained in block j
    rows = []
    cols = []
    data = []
    
    for j, block in enumerate(all_blocks):
        block_set = set(block)
        for i, tsub in enumerate(all_t_subsets):
            # Check if tsub is subset of block
            # Optimization: only check if min(tsub) in block_set etc? 
            # For v=19, k=7, t=4, direct check is fast enough.
            if all(x in block_set for x in tsub):
                rows.append(i)
                cols.append(j)
                data.append(1)
                
    A = csr_matrix((data, (rows, cols)), shape=(num_t, num_blocks))
    
    # Helper: Run Greedy Set Cover to get initial feasible solution
    def run_greedy():
        uncovered = np.ones(num_t, dtype=bool)
        blocks_selected = []
        
        # Precompute block coverage indices for faster access
        # We already have A, but for greedy, we might want to iterate differently.
        # Let's use the sparse structure of A.
        # For each block j, the indices of t-subsets it covers are A.indices where A.indptr[j] to A.indptr[j+1]
        
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            best_block_idx = -1
            best_gain = -1
            
            # Iterate through all blocks to find the one covering most uncovered t-subsets
            # Optimization: Only consider blocks that cover at least one uncovered t-subset?
            # Simpler: Just check all. 50k blocks * 3876 checks is ~200M ops, might be slow in Python.
            # Better: Use the sparse matrix.
            
            # For each block j, gain = number of uncovered t-subsets in A[:, j]
            # We can vectorize this somewhat or just loop.
            
            # Let's try a slightly smarter greedy:
            # Pick a random uncovered t-subset, then pick a block covering it that maximizes gain?
            # Or just standard greedy.
            
            # To speed up greedy, we can precompute the list of t-subsets for each block.
            # But A is already built.
            
            # Let's just do a simple greedy with early stopping if time is tight.
            # Since we have A, we can calculate gain for each block.
            
            # Actually, for v=19, num_blocks=50k.
            # Looping 50k times and doing sparse slice operations might be okay.
            
            for j in range(num_blocks):
                # Get indices of t-subsets covered by block j
                start = A.indptr[j]
                end = A.indptr[j+1]
                if start == end:
                    continue
                covered_indices = A.indices[start:end]
                # Count how many of these are currently uncovered
                gain = np.sum(uncovered[covered_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = j
            
            if best_block_idx == -1:
                # Should not happen if uncovered.any() is true, unless no block covers them (impossible by definition)
                break
                
            blocks_selected.append(best_block_idx)
            
            # Update uncovered
            start = A.indptr[best_block_idx]
            end = A.indptr[best_block_idx+1]
            covered_indices = A.indices[start:end]
            uncovered[covered_indices] = False
            
        return blocks_selected

    # Step 1: Get a quick greedy solution to establish an upper bound
    greedy_solution = run_greedy()
    best_blocks_indices = greedy_solution
    best_count = len(best_blocks_indices)
    
    # Step 2: Solve ILP
    # Objective: minimize sum(x_j)
    c = np.ones(num_blocks)
    
    # Constraints: A x >= 1
    # LinearConstraint(A, lb=np.ones(num_t), ub=np.inf)
    # However, milp expects constraints in standard form or linear constraints.
    # A x >= 1  =>  -A x <= -1
    # Or use LinearConstraint(A, lb=np.ones(num_t), ub=np.inf)
    
    try:
        constraints = LinearConstraint(A, lb=np.ones(num_t), ub=np.inf)
        integrality = np.ones(num_blocks) # All variables are integer
        bounds = Bounds(lb=0, ub=1)
        
        # Set time limit for the solver
        # milp options: time_limit is in seconds
        # We have about 140s total. Greedy took some time. Let's give ILP 100s.
        time_remaining = deadline - time.time()
        solver_time_limit = max(10, time_remaining - 5) # Keep 5s for cleanup
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": solver_time_limit, "presolve": True})
        
        if res.status == 0 or res.status == 1: # 0: optimal, 1: feasible
            # res.x contains the solution
            # Round to nearest integer (should be already integer or close)
            x_sol = np.round(res.x).astype(int)
            # Filter out zero weights
            ilp_blocks_indices = np.where(x_sol > 0)[0]
            
            if len(ilp_blocks_indices) < best_count:
                best_blocks_indices = ilp_blocks_indices
                best_count = len(ilp_blocks_indices)
                
    except Exception as e:
        # If ILP fails or errors out, we still have the greedy solution
        pass

    # Step 3: Local Optimization on the best solution found
    # Remove redundant blocks
    if best_blocks_indices is not None and len(best_blocks_indices) > 0:
        # Convert to list of block indices
        current_indices = list(best_blocks_indices)
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
                
            # Build coverage count
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for idx in current_indices:
                start = A.indptr[idx]
                end = A.indptr[idx+1]
                covered = A.indices[start:end]
                coverage_count[covered] += 1
                
            # Check each block for redundancy
            redundant_indices = []
            for pos, idx in enumerate(current_indices):
                is_essential = False
                start = A.indptr[idx]
                end = A.indptr[idx+1]
                covered = A.indices[start:end]
                for sub_idx in covered:
                    if coverage_count[sub_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_indices.append(pos)
                    
            if redundant_indices:
                # Remove one redundant block
                idx_to_remove = random.choice(redundant_indices)
                current_indices.pop(idx_to_remove)
                changed = True
        
        best_blocks_indices = current_indices
        best_count = len(best_blocks_indices)

    # Construct final list of blocks
    if best_blocks_indices is None:
        # Fallback: return empty list or error? Task says "list of blocks".
        # If no solution found, return empty? No, must be valid.
        # Greedy should always find a solution.
        return []

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks_indices]
    
    # Verify validity (optional but good for safety, might be slow if exhaustive)
    # The greedy and ILP guarantee validity theoretically.
    # Let's do a quick check if time permits?
    # For now, trust the construction.
    
    return final_blocks
