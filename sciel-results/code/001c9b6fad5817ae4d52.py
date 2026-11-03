import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Build the incidence matrix A where A[i, j] = 1 if block j covers t-subset i
    # A is N_t x N_b
    # We use lil_matrix for efficient row-wise construction or just direct assignment
    A = lil_matrix((N_t, N_b), dtype=np.int8)
    
    # For each block, find which t-subsets it covers
    # Optimization: Instead of iterating all blocks and all t-subsets, 
    # iterate blocks and their t-subsets.
    for j, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            i = t_to_idx[ts]
            A[i, j] = 1
            
    A_csr = A.tocsr()
    
    # Phase 1: Minimize number of blocks
    # Minimize sum(x_j)
    # Subject to A @ x >= 1
    # x_j in {0, 1}
    
    c1 = np.ones(N_b, dtype=float)
    lb = np.zeros(N_b)
    ub = np.ones(N_b)
    
    # Constraint: A @ x >= 1
    # milp expects: lb <= A @ x <= ub
    # So: 1 <= A @ x <= inf
    cons1 = LinearConstraint(A_csr, lb=1.0, ub=np.inf)
    bounds1 = Bounds(lb, ub)
    
    # Solve Phase 1
    try:
        res1 = milp(c=c1, constraints=cons1, integrality=np.ones(N_b), bounds=bounds1)
        if not res1.success:
            # Fallback to greedy if ILP fails
            return _greedy_fallback(v, k, t, deadline, t0)
        
        x1 = res1.x
        # Round to binary
        x1 = (x1 > 0.5).astype(int)
        
        # Verify validity and count
        # In case of numerical issues, ensure validity
        # If invalid, add missing blocks greedily
        covered_counts = A_csr.dot(x1)
        missing = np.where(covered_counts < 1)[0]
        
        if len(missing) > 0:
            # Greedily add blocks to cover missing t-subsets
            current_blocks = set(np.where(x1 > 0)[0])
            for m_idx in missing:
                # Find a block covering m_idx not already in set
                candidates = A_csr[m_idx].nonzero()[1]
                for c_idx in candidates:
                    if c_idx not in current_blocks:
                        current_blocks.add(c_idx)
                        x1[c_idx] = 1
                        break
                if time.time() > deadline - 5.0:
                    break
            
            x1 = np.array([1 if i in current_blocks else 0 for i in range(N_b)], dtype=int)
            
    except Exception:
        return _greedy_fallback(v, k, t, deadline, t0)
    
    # Phase 2: Minimize redundancy (sum of pairwise overlaps)
    # Objective: minimize sum_{i<j} x_i x_j * overlap(i,j) ? No, that's quadratic.
    # Linear surrogate: Minimize sum_j x_j * (sum_{i != j} overlap(i,j) * x_i) is quadratic.
    # Instead, we can minimize the total "weight" of blocks, where weight is 1 + epsilon * (number of t-subsets covered by other selected blocks).
    # Or simpler: Minimize sum x_j subject to A x >= 1 AND minimizing a second objective: sum x_j * (number of t-subsets covered by x_j that are covered by >1 block).
    
    # Let's try a different Phase 2:
    # Fix the number of blocks to N1 (from Phase 1).
    # Minimize the "spread" or "redundancy".
    # A common heuristic is to minimize the sum of squares of coverage counts? No.
    # Let's try minimizing the sum of x_j * (number of t-subsets in block j that are covered by at least 2 blocks).
    # This is still hard to linearize directly without big-M or additional vars.
    
    # Alternative Phase 2: 
    # Use the solution from Phase 1 as a starting point.
    # Perform a local search to remove redundant blocks or swap for better structure.
    # Given time constraints, a simple local search on the ILP result is safer than a complex second ILP.
    
    # Let's check if we have time for a simple local search.
    if time.time() < deadline - 10.0:
        x1 = _local_search_refine(x1, A_csr, N_b, N_t, deadline)
        
    # Convert back to list of blocks
    final_blocks = [list(all_blocks[j]) for j in range(N_b) if x1[j] == 1]
    
    # Final safety check: Ensure validity
    # If the ILP solution was slightly off or local search broke it, repair.
    if final_blocks:
        # Quick check
        covered_mask = np.zeros(N_t, dtype=bool)
        for b in final_blocks:
            for ts in itertools.combinations(b, t):
                covered_mask[t_to_idx[ts]] = True
        if not np.all(covered_mask):
            # Repair
            missing = np.where(~covered_mask)[0]
            for m_idx in missing:
                candidates = A_csr[m_idx].nonzero()[1]
                for c_idx in candidates:
                    # Check if block is already in final_blocks
                    # This is O(N_b) check, but only for missing items
                    already = False
                    for fb in final_blocks:
                        if fb == list(all_blocks[c_idx]):
                            already = True
                            break
                    if not already:
                        final_blocks.append(list(all_blocks[c_idx]))
                        break
                if time.time() > deadline - 2.0:
                    break
    else:
        # Fallback if no blocks found
        final_blocks = _greedy_fallback(v, k, t, deadline, t0)
        
    return final_blocks

def _greedy_fallback(v, k, t, deadline, t0):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    block_to_t = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_t.append(covered)
        
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_to_t):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    covered_mask = np.zeros(N_t, dtype=bool)
    selected_blocks = []
    
    while not np.all(covered_mask):
        if time.time() > deadline - 2.0:
            break
        uncovered_idx = np.where(~covered_mask)[0]
        if len(uncovered_idx) == 0:
            break
            
        best_block = -1
        best_gain = -1
        
        # Sample candidates
        sample_size = min(5, len(uncovered_idx))
        rng = random.Random(int(time.time()*1000) % 10000)
        sampled_ts = rng.sample(uncovered_idx.tolist(), sample_size)
        
        candidates = set()
        for ti in sampled_ts:
            candidates.update(t_to_blocks[ti])
            
        for bi in candidates:
            if bi in selected_blocks:
                continue
            gain = 0
            for ti in block_to_t[bi]:
                if not covered_mask[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
                
        if best_block == -1:
            ti = rng.choice(uncovered_idx.tolist())
            best_block = rng.choice(t_to_blocks[ti])
            
        selected_blocks.append(best_block)
        for ti in block_to_t[best_block]:
            covered_mask[ti] = True
            
    return [list(all_blocks[bi]) for bi in selected_blocks]

def _local_search_refine(x_init, A_csr, N_b, N_t, deadline):
    x = x_init.copy()
    # Compute coverage counts
    counts = A_csr.dot(x)
    
    # Objective: Minimize sum(x) + penalty * (number of t-subsets with count > 1) ?
    # Actually, we just want to remove redundant blocks if possible.
    # A block is redundant if removing it doesn't leave any t-subset with count 0.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try removing each block
        for j in range(N_b):
            if x[j] == 1:
                # Check if block j is essential
                # Essential if there exists a t-subset i such that count[i] == 1 and A[i, j] == 1
                # We can check this by looking at counts and A[:, j]
                col_j = A_csr[:, j].toarray().flatten()
                # If any count is 1 where col_j is 1, it's essential
                if not np.any((counts == 1) & (col_j == 1)):
                    # Remove it
                    x[j] = 0
                    counts -= col_j
                    improved = True
                    # Restart loop to catch cascading redundancies
                    break
                    
    return x
