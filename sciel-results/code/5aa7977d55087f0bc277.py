"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    # Check diagonal terms (must be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check off-diagonal terms (must be 0)
    # Coefficient of A[i,j] B[l,k] in C[i,k] is sum_r U[i,j,r] V[l,k,r] W[k,i,r]
    # This must be 0 if j != l
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    if j == l:
                        continue
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    if abs(val) > 1e-9:
                        return False
    return True

def _find_common_denominator(values, max_den=1000):
    """Find a common denominator that makes all values integers."""
    from fractions import Fraction
    fracs = []
    for v in values:
        f = Fraction(v).limit_denominator(max_den)
        if abs(v - f) > 1e-9:
            return None
        fracs.append(f)
    
    lcms = [1]
    for f in fracs:
        new_lcms = []
        for lcm in lcms:
            new_lcms.append(lcm * f.denominator // gcd(lcm, f.denominator))
        lcms = new_lcms
    
    return max(lcms) if lcms else None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Search for lower rank decompositions
    # Strategy: Random U, V (sparse integers), solve for W, check if W has small denominator.
    # Scale appropriately to get integer entries.
    
    max_attempts_per_rank = 2000
    
    # Start from known lower bounds or specific targets
    # Rank 18 is a known candidate for 2x4x4
    start_rank = 18
    if start_rank > rank_trivial:
        start_rank = rank_trivial - 1
        
    for rank in range(start_rank, 10, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            
            seed = attempt * 7919 + rank * 104729
            np.random.seed(seed)
            
            # Generate sparse U and V with entries in {-1, 0, 1}
            # Vary sparsity based on attempt
            sparsity = 0.2 + 0.4 * (attempt % 10) / 10.0
            U_r = np.where(np.random.rand(n * m, rank) < sparsity, np.random.choice([-1, 1], size=(n * m, rank)), 0)
            V_r = np.where(np.random.rand(m * p, rank) < sparsity, np.random.choice([-1, 1], size=(m * p, rank)), 0)
            
            U_np = U_r.astype(float)
            V_np = V_r.astype(float)
            
            W_np = np.zeros((p * n, rank))
            valid = True
            
            # For each (i, k), solve for W_{k, i, :}
            for i in range(n):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    
                    # Construct the system A W = 1
                    # A has shape (m, rank)
                    # A[j, r] = U[i, j, r] * V[j, k, r]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    # Solve A W_col = 1
                    try:
                        W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if solution is close to exact
                        if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-8:
                            valid = False
                            break
                            
                        W_np[k * n + i, :] = W_col
                    except np.linalg.LinAlgError:
                        valid = False
                        break
            
            if valid:
                # Check if W entries have small denominators
                all_w_values = W_np.flatten().tolist()
                common_den = _find_common_denominator(all_w_values, max_den=200)
                
                if common_den is not None:
                    # Scale W by common_den to make it integer
                    W_int = [[int(round(x * common_den)) for x in row] for row in W_np.tolist()]
                    
                    # U and V are already integers, so no scaling needed
                    U_int = [[int(x) for x in row] for row in U_np.tolist()]
                    V_int = [[int(x) for x in row] for row in V_np.tolist()]
                    
                    # Verify the decomposition
                    if _check_decomposition(U_int, V_int, W_int, n, m, p):
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        best_rank = rank
                        break  # Found a valid decomposition for this rank, move to next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
