# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 maximizing sum 1/(512 - a).
    
    Strategy:
    1. Attempt to solve as a Mixed Integer Linear Programming (MILP) problem using SciPy's HiGHS solver.
       This models the Maximum Weight Independent Set problem exactly.
    2. If MILP is unavailable or fails, fall back to a Greedy heuristic followed by Local Search (1-swap and 2-swap moves).
    """
    deadline = time.time() + 145.0  # Safety margin
    
    N = 512
    vertices = list(range(1, N))
    n = len(vertices)
    
    # Precompute weights
    weights = np.array([1.0 / (N - a) for a in vertices])
    
    # Precompute conflict matrix (boolean) for fast access
    # conflicts[i, j] is True if gcd(vertices[i], vertices[j]) > 1
    conflicts = np.zeros((n, n), dtype=bool)
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            if gcd(vi, vj) > 1:
                conflicts[i, j] = True
                conflicts[j, i] = True

    # --- Attempt 1: MILP via SciPy ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        # Build constraint matrix A for x_i + x_j <= 1 for all conflicts
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if conflicts[i, j]:
                    rows.append(len(data))
                    cols.append(i)
                    data.append(1.0)
                    
                    rows.append(len(data))
                    cols.append(j)
                    data.append(1.0)
        
        if data:
            A = csr_matrix((data, (rows, cols)), shape=(len(data), n))
            b = np.ones(len(data))
        else:
            # No conflicts, take all
            return vertices
            
        c = -weights # Minimize negative weight
        
        constraints = LinearConstraint(A, -np.inf, b)
        bounds = Bounds(0, 1)
        integrality = np.ones(n) # All variables are binary
        
        # Set time limit via options if possible, but milp API might not expose it directly in all versions
        # We rely on the problem being small enough.
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            solution_indices = [i for i in range(n) if res.x[i] > 0.5]
            solution = [vertices[i] for i in solution_indices]
            return solution
            
    except Exception:
        pass

    # --- Attempt 2: Greedy + Local Search ---
    
    # Greedy Initialization
    # Sort vertices by weight descending
    order = np.argsort(-weights)
    
    available = np.ones(n, dtype=bool)
    current_set = set()
    current_score = 0.0
    
    for idx in order:
        if not available[idx]:
            continue
        # Check if idx is coprime to all in current_set
        # Since we maintain 'available' by removing conflicts, we just need to check if it's still available
        # But wait, 'available' is updated globally.
        # If we pick idx, we remove it and its conflicts from 'available'.
        
        # Is idx compatible with current_set?
        # Yes, because if it conflicted with any previously picked vertex, it would have been removed from 'available'.
        
        current_set.add(idx)
        current_score += weights[idx]
        
        # Remove conflicts
        available[idx] = False
        # Find conflicts of idx
        # conflicts[idx] is a boolean array
        available[conflicts[idx]] = False
        
    # Local Search
    # 1-swap: Remove one, add one
    # 2-swap: Remove two, add two (might be too slow, stick to 1-swap and add)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a new element
        # Candidates are those not in current_set and not conflicting with any in current_set
        # We can recompute available mask
        avail_mask = np.ones(n, dtype=bool)
        for idx in current_set:
            avail_mask[conflicts[idx]] = False
            avail_mask[idx] = False
            
        # Find best candidate to add
        best_add_idx = -1
        best_add_gain = 0.0
        for idx in range(n):
            if avail_mask[idx]:
                gain = weights[idx]
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_idx = idx
                    
        if best_add_idx != -1:
            current_set.add(best_add_idx)
            current_score += best_add_gain
            improved = True
            continue # Restart loop as availability changed
            
        # Try 1-swap: Remove i from set, Add j not in set
        # For each i in set, temporarily remove it
        # Then find best j to add that is compatible with set \ {i}
        
        set_list = list(current_set)
        for i in set_list:
            # Temporarily remove i
            temp_set = current_set - {i}
            temp_score = current_score - weights[i]
            
            # Find best j to add
            avail_mask = np.ones(n, dtype=bool)
            for k in temp_set:
                avail_mask[conflicts[k]] = False
                avail_mask[k] = False
            
            best_swap_j = -1
            best_swap_gain = 0.0
            for j in range(n):
                if j == i: continue # j is not in set, but i is. j can be i? No, distinct.
                if avail_mask[j]:
                    gain = weights[j] - weights[i] # Net change
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_j = j
                        
            if best_swap_j != -1 and best_swap_gain > 1e-9:
                # Perform swap
                current_set.remove(i)
                current_set.add(best_swap_j)
                current_score += best_swap_gain
                improved = True
                break # Restart loop
                
    # Convert set of indices to list of values
    result = [vertices[i] for i in current_set]
    return result

# EVOLVE-BLOCK-END
