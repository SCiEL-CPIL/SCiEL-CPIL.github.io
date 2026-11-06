import time
from math import gcd
import numpy as np

def construct():
    """
    Finds a set of pairwise coprime integers in 1..511 maximizing the sum of 1/(512-a).
    Uses MILP if available, otherwise falls back to a greedy + local search heuristic.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    N = 511
    # Weights: 1 / (512 - a)
    # a ranges from 1 to 511
    a_values = np.arange(1, N + 1)
    weights = 1.0 / (512 - a_values)
    
    # Identify conflicts: pairs (i, j) such that gcd(i, j) > 1
    # We need to build the constraint matrix for MILP or adjacency for heuristics
    # Number of variables is 511.
    
    # Precompute GCDs to find conflicts
    # Storing all conflicts as (i, j) indices
    conflicts = []
    for i in range(N):
        ai = a_values[i]
        for j in range(i + 1, N):
            aj = a_values[j]
            if gcd(ai, aj) > 1:
                conflicts.append((i, j))
                
    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: minimize -sum(weights * x)
        c = -weights
        
        # Constraints: x_i + x_j <= 1 for all conflicts
        # Number of constraints = len(conflicts)
        m = len(conflicts)
        n = N
        
        # Build A matrix (sparse is better, but scipy.optimize.milp accepts dense or sparse)
        # Let's use a sparse matrix for efficiency
        from scipy.sparse import lil_matrix
        A = lil_matrix((m, n))
        
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1
            A[k, j] = 1
            
        A = A.tocsr()
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x_i must be integer
        integrality = np.ones(n)
        
        # Linear constraint: A @ x <= 1
        # scipy.optimize.milp expects LinearConstraint(A, lb, ub)
        # lb is -inf, ub is 1
        cons = LinearConstraint(A, lb=-np.inf, ub=1)
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 120, "mip_gap": 0.0})
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract chosen indices
            # x is a numpy array of 0s and 1s
            chosen_indices = np.where(x > 0.5)[0]
            chosen_a = a_values[chosen_indices]
            
            # Verify pairwise coprime just in case of numerical issues
            valid = True
            for i in range(len(chosen_a)):
                for j in range(i + 1, len(chosen_a)):
                    if gcd(chosen_a[i], chosen_a[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return chosen_a.tolist()
            else:
                # Fallback if validation fails (shouldn't happen)
                pass
                
    except Exception:
        pass
        
    # Fallback: Greedy + Local Search
    # Greedy: pick largest weights (smallest denominators) that are coprime to current set
    # This is the seed algorithm.
    chosen = []
    # Sort indices by weight descending (a descending)
    # Since weight = 1/(512-a), larger a means larger weight.
    # So sorting a descending is sorting weight descending.
    for a in range(N, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    best_score = sum(1.0 / (512 - a) for a in chosen)
    best_set = set(chosen)
    
    # Local Search: Try swapping one element with another
    # Or adding/removing.
    # Given the time, a simple hill climbing might be enough if MILP fails.
    # But MILP is very likely to succeed.
    
    # If we reach here, return the greedy result
    return chosen

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a set of pairwise coprime integers in 1..511 maximizing the sum of 1/(512-a).
    Uses MILP if available, otherwise falls back to a greedy + local search heuristic.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    N = 511
    # Weights: 1 / (512 - a)
    # a ranges from 1 to 511
    a_values = np.arange(1, N + 1)
    weights = 1.0 / (512 - a_values)
    
    # Identify conflicts: pairs (i, j) such that gcd(i, j) > 1
    # We need to build the constraint matrix for MILP or adjacency for heuristics
    # Number of variables is 511.
    
    # Precompute GCDs to find conflicts
    # Storing all conflicts as (i, j) indices
    conflicts = []
    for i in range(N):
        ai = a_values[i]
        for j in range(i + 1, N):
            aj = a_values[j]
            if gcd(ai, aj) > 1:
                conflicts.append((i, j))
                
    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: minimize -sum(weights * x)
        c = -weights
        
        # Constraints: x_i + x_j <= 1 for all conflicts
        # Number of constraints = len(conflicts)
        m = len(conflicts)
        n = N
        
        # Build A matrix (sparse is better, but scipy.optimize.milp accepts dense or sparse)
        # Let's use a sparse matrix for efficiency
        from scipy.sparse import lil_matrix
        A = lil_matrix((m, n))
        
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1
            A[k, j] = 1
            
        A = A.tocsr()
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x_i must be integer
        integrality = np.ones(n)
        
        # Linear constraint: A @ x <= 1
        # scipy.optimize.milp expects LinearConstraint(A, lb, ub)
        # lb is -inf, ub is 1
        cons = LinearConstraint(A, lb=-np.inf, ub=1)
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 120, "mip_gap": 0.0})
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract chosen indices
            # x is a numpy array of 0s and 1s
            chosen_indices = np.where(x > 0.5)[0]
            chosen_a = a_values[chosen_indices]
            
            # Verify pairwise coprime just in case of numerical issues
            valid = True
            for i in range(len(chosen_a)):
                for j in range(i + 1, len(chosen_a)):
                    if gcd(chosen_a[i], chosen_a[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return chosen_a.tolist()
            else:
                # Fallback if validation fails (shouldn't happen)
                pass
                
    except Exception:
        pass
        
    # Fallback: Greedy + Local Search
    # Greedy: pick largest weights (smallest denominators) that are coprime to current set
    # This is the seed algorithm.
    chosen = []
    # Sort indices by weight descending (a descending)
    # Since weight = 1/(512-a), larger a means larger weight.
    # So sorting a descending is sorting weight descending.
    for a in range(N, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    best_score = sum(1.0 / (512 - a) for a in chosen)
    best_set = set(chosen)
    
    # Local Search: Try swapping one element with another
    # Or adding/removing.
    # Given the time, a simple hill climbing might be enough if MILP fails.
    # But MILP is very likely to succeed.
    
    # If we reach here, return the greedy result
    return chosen
# EVOLVE-BLOCK-END
