"""Maximum independent set on square-sum conflict graph using ILP (scipy.optimize.milp)."""
import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, a cannot be in the set because a+a=2a is a square.
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict list
    # We need constraints x_i + x_j <= 1 for all conflicting pairs.
    # Note: The diagonal constraint (x_i + x_i <= 1 => 2x_i <= 1 => x_i <= 0.5 => x_i=0) 
    # is already handled by filtering valid_numbers.
    # So we only need to check pairs i < j where valid_numbers[i] + valid_numbers[j] is a square.
    
    conflict_pairs = []
    for i in range(num_valid):
        a = valid_numbers[i]
        for j in range(i + 1, num_valid):
            b = valid_numbers[j]
            if a + b in squares:
                conflict_pairs.append((i, j))
    
    # If no conflicts, all valid numbers can be included
    if not conflict_pairs:
        return sorted(valid_numbers)

    # Formulate ILP
    # Maximize sum(x)
    # Subject to: x_i + x_j <= 1 for all (i,j) in conflict_pairs
    #             0 <= x_i <= 1
    
    # Objective coefficients
    c = np.ones(num_valid)
    
    # Constraints
    # A_ub @ x <= b_ub
    # We can build the matrix A_ub efficiently if num_valid is small enough, 
    # but for ~230 variables and potentially thousands of constraints, 
    # a dense matrix is fine (230 * ~10000 = 2.3M floats ~ 18MB).
    
    # Let's estimate constraint count.
    # Each number conflicts with at most ~10 others (squares up to 512: 4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400,441,484).
    # 21 squares. So max degree ~21. Total edges ~ 230 * 21 / 2 ~ 2400.
    # Dense matrix is definitely fine.
    
    A_ub = np.zeros((len(conflict_pairs), num_valid))
    for k, (i, j) in enumerate(conflict_pairs):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
        
    b_ub = np.ones(len(conflict_pairs))
    
    # Integrality: all variables are integers
    integrality = np.ones(num_valid)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Linear Constraints
    constraints = LinearConstraint(A_ub, ub=b_ub)
    
    # Run MILP
    best_result = None
    best_size = 0
    
    # Time limit for scipy is in seconds
    # We reserve 5s for cleanup and fallback
    time_limit = deadline - time.time() - 5.0
    
    try:
        # HiGHS is the default solver for milp
        # options: time_limit, mip_rel_gap, etc.
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0, "mip_abs_gap": 0.0}
        )
        
        if res.success and res.x is not None:
            # res.x is a numpy array of floats, but should be close to 0 or 1
            x = np.round(res.x).astype(int)
            # Verify validity
            valid_mask = x > 0
            size = valid_mask.sum()
            
            # Double check constraints just in case of numerical issues
            valid = True
            selected_vals = [valid_numbers[i] for i in range(num_valid) if valid_mask[i]]
            for i in range(len(selected_vals)):
                for j in range(i, len(selected_vals)):
                    if selected_vals[i] + selected_vals[j] in squares:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_result = selected_vals
                best_size = size
            else:
                # If numerical error caused invalidity, fall back to greedy
                pass
                
    except Exception as e:
        pass # Fall back to greedy

    # Fallback: Greedy + Local Search
    if best_result is None or len(best_result) == 0:
        # Simple greedy by degree
        # Build adjacency for quick lookup
        adj = [set() for _ in range(num_valid)]
        for i, j in conflict_pairs:
            adj[i].add(j)
            adj[j].add(i)
            
        # Sort by degree
        order = sorted(range(num_valid), key=lambda x: len(adj[x]))
        
        selected = []
        selected_set = set()
        
        for v in order:
            if v in selected_set:
                continue
            # Check if v conflicts with any already selected
            conflict = False
            for u in selected_set:
                if u in adj[v]:
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
                selected_set.add(v)
        
        # Local Search: Try to add/remove/swap
        # Convert to mask for speed
        mask = 0
        for v in selected:
            mask |= (1 << v)
            
        # Precompute conflict masks
        conflict_mask = [0] * num_valid
        for i in range(num_valid):
            for j in adj[i]:
                conflict_mask[i] |= (1 << j)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding any vertex that doesn't conflict
            for v in range(num_valid):
                if not (mask & (1 << v)):
                    if (conflict_mask[v] & mask) == 0:
                        mask |= (1 << v)
                        improved = True
            
            # Try swapping: remove one, add two? Or just remove one if it allows adding more?
            # Simple swap: for each selected v, remove it, see if we can add any new vertex.
            # This is O(N^2) per iteration.
            # A better heuristic: 1-for-1 swap to increase size? No, size is fixed in 1-for-1.
            # We need 1-for-2 or 2-for-3.
            # Let's try: for each selected v, remove it. Then try to add any available vertex.
            # If we add one, size stays same. If we add two, size increases.
            # This is complex. Let's stick to simple addition first.
            # If no addition possible, try removing a high-degree vertex and seeing if neighbors become available.
            
            # Heuristic: Remove a vertex, then greedily add as many as possible from its neighbors.
            # If the number added > 1, we improved.
            
            current_size = mask.bit_count()
            
            for v in range(num_valid):
                if mask & (1 << v):
                    # Remove v
                    new_mask = mask & ~(1 << v)
                    
                    # Try to add vertices that were blocked by v
                    # Candidates are neighbors of v that are not in new_mask
                    # And don't conflict with rest of new_mask
                    candidates = conflict_mask[v] & ~new_mask
                    added = 0
                    while candidates:
                        lsb = candidates & (-candidates)
                        w = lsb.bit_length() - 1
                        if (conflict_mask[w] & new_mask) == 0:
                            new_mask |= lsb
                            added += 1
                        candidates ^= lsb
                    
                    if new_mask.bit_count() > current_size:
                        mask = new_mask
                        current_size = mask.bit_count()
                        improved = True
                        break # Restart loop with new mask

        result_indices = []
        for v in range(num_valid):
            if mask & (1 << v):
                result_indices.append(v)
        
        best_result = [valid_numbers[v] for v in result_indices]
        best_size = len(best_result)

    if not best_result:
        # Absolute fallback: return any valid number
        return [1]

    return sorted(best_result)

# EVOLVE-BLOCK-END
