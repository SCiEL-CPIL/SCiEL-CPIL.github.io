"""
Program to find the minimal rank tensor decomposition for 3x4 * 4x5 matrix multiplication.
Approach: Model as sparse linear system, attempt small-rank search with integer/half-integer
entries, fall back to schoolbook (rank 60) if no better solution found within time budget.
"""

import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    
    # The schoolbook rank is n * m * p
    rank_schoolbook = n * m * p
    
    # Start with schoolbook solution as fallback
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank solution
    # The tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # For each (a,b,c) with T[a,b,c] = 1: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a bilinear (actually trilinear) constraint, making it hard.
    # Let's try a greedy/local search approach for small r values.
    
    start_time = time.time()
    time_budget = 100  # seconds, leaving time for finalization
    
    # Try ranks from 50 down to 55 (conservative range)
    # Actually, let's try a broader but still feasible range
    # Known lower bound for 3x4x5 is at least 42 (from literature), but let's try from 55 down
    
    best_solution = None
    
    for r in range(55, 49, -1):  # Try 55, 54, 53, 52, 51, 50
        if time.time() - start_time > time_budget:
            break
        
        # Try to find a solution with rank r
        # Use a randomized approach with local improvements
        solution = _try_find_decomposition(n, m, p, r, time_budget - (time.time() - start_time))
        if solution is not None:
            best_solution = (r, solution)
            break  # Found the lowest r in our range
    
    if best_solution is not None:
        r, (U, V, W) = best_solution
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_find_decomposition(n, m, p, r, time_budget):
    """
    Attempt to find a rank-r decomposition using a greedy + local search approach.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    start_time = time.time()
    
    # Initialize with random integer/half-integer values
    # Use values from {-1, -0.5, 0, 0.5, 1}
    possible_values = [-1, -0.5, 0, 0.5, 1]
    
    # Build the list of constraints
    # For each (i, j, k) in [n]x[m]x[p]:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    #   Constraint: sum_r U[a,r] * V[b,r] * W[c,r] = 1
    # For all other (a, b, c) combinations:
    #   Constraint: sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # There are n*m*p = 60 positive constraints
    # Total possible (a,b,c) triples: (n*m)*(m*p)*(p*n) = 12*20*15 = 3600
    # So 3600 - 60 = 3540 zero constraints
    
    # This is too many constraints to check all at once efficiently.
    # Let's use a different approach: start with schoolbook and try to merge columns.
    
    # Actually, let's try a simpler heuristic: start with random initialization and
    # use gradient-like updates to satisfy constraints.
    
    # Initialize U, V, W with small random values
    np.random.seed(42)
    U = np.random.uniform(-0.5, 0.5, (n*m, r))
    V = np.random.uniform(-0.5, 0.5, (m*p, r))
    W = np.random.uniform(-0.5, 0.5, (p*n, r))
    
    # Round to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Define the positive constraint indices
    pos_constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                pos_constraints.append((a, b, c))
    
    # For efficiency, only check a subset of zero constraints
    # Sample some zero constraints
    all_indices_a = list(range(n*m))
    all_indices_b = list(range(m*p))
    all_indices_c = list(range(p*n))
    
    zero_constraints = []
    pos_set = set(pos_constraints)
    for _ in range(200):  # Sample 200 zero constraints
        a = np.random.randint(n*m)
        b = np.random.randint(m*p)
        c = np.random.randint(p*n)
        if (a, b, c) not in pos_set:
            zero_constraints.append((a, b, c))
    
    # Iterative improvement
    for iteration in range(2000):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Check positive constraints
        pos_satisfied = True
        for (a, b, c) in pos_constraints:
            val = np.dot(U[a] * V[b], W[c])
            if abs(val - 1) > 1e-6:
                pos_satisfied = False
                break
        
        if pos_satisfied:
            # Check zero constraints
            zero_satisfied = True
            for (a, b, c) in zero_constraints:
                val = np.dot(U[a] * V[b], W[c])
                if abs(val) > 1e-6:
                    zero_satisfied = False
                    break
            
            if zero_satisfied:
                # Verify all zero constraints
                all_zero_ok = True
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            if (a, b, c) in pos_set:
                                continue
                            val = np.dot(U[a] * V[b], W[c])
                            if abs(val) > 1e-6:
                                all_zero_ok = False
                                break
                        if not all_zero_ok:
                            break
                    if not all_zero_ok:
                        break
                
                if all_zero_ok:
                    # Found a solution!
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return (U_list, V_list, W_list)
        
        # Random perturbation
        if np.random.rand() < 0.5:
            idx = np.random.randint(n*m)
            col = np.random.randint(r)
            U[idx, col] += np.random.choice([-1, 1]) * 0.5
            U[idx, col] = np.round(U[idx, col] * 2) / 2
        elif np.random.rand() < 0.5:
            idx = np.random.randint(m*p)
            col = np.random.randint(r)
            V[idx, col] += np.random.choice([-1, 1]) * 0.5
            V[idx, col] = np.round(V[idx, col] * 2) / 2
        else:
            idx = np.random.randint(p*n)
            col = np.random.randint(r)
            W[idx, col] += np.random.choice([-1, 1]) * 0.5
            W[idx, col] = np.round(W[idx, col] * 2) / 2
    
    return None


def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    r = U.shape[1]
    
    # Check positive constraints
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a] * V[b], W[c])
                if abs(val - 1) > 1e-6:
                    return False
    
    # Check zero constraints (sample)
    pos_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                pos_set.add((a, b, c))
    
    for _ in range(500):
        a = np.random.randint(n*m)
        b = np.random.randint(m*p)
        c = np.random.randint(p*n)
        if (a, b, c) not in pos_set:
            val = np.dot(U[a] * V[b], W[c])
            if abs(val) > 1e-6:
                return False
    
    return True
# EVOLVE-BLOCK-END
