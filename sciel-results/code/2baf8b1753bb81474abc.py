"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    Uses a heuristic search: start from schoolbook, try to reduce rank via coordinate descent
    and random restarts with integer/half-integer constraints.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook rank is n*m*p = 45
    # We know that for matrix multiplication, the rank is at least max(n*m, m*p, p*n) but can be lower.
    # For 3x3 by 3x5, let's try to find a lower rank.
    
    # Strategy: 
    # 1. Start with a known low-rank decomposition if possible (e.g., from literature or known bounds).
    # 2. Use coordinate descent to try to reduce rank further or find exact integer solutions.
    # 3. Since we need integer/half-integer entries, we can use a rounding strategy.
    
    # For now, let's implement a search that tries ranks from a lower bound up to 45.
    # Lower bound: The rank must be at least the maximum of the dimensions in some sense.
    # A simple lower bound is max(n, m, p) but that's not tight.
    # A better lower bound: rank >= max(n*m, m*p, p*n) / min(n, m, p) is not standard.
    # Let's just try ranks from 25 to 45 (since 45 is schoolbook, and we know some decompositions are lower).
    
    # Actually, let's use a more systematic approach:
    # We'll try to find a decomposition of rank R by solving a system of equations.
    # But that's hard. Instead, let's use the fact that for small matrices, we can try random initializations
    # and then refine.
    
    # Given the time constraint, let's implement a heuristic that:
    # 1. Tries to find a decomposition of rank R for R in a range.
    # 2. Uses random initialization with small integers/half-integers.
    # 3. Uses coordinate descent to minimize the error.
    # 4. Rounds to nearest integer/half-integer and checks if exact.
    
    deadline = time.time() + 100  # Leave 20s margin
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, return schoolbook as a fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    for R in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with small random values
            # Use values from {-1, -0.5, 0, 0.5, 1} for simplicity
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, R))
            
            # Coordinate descent to minimize error
            # Error = sum_{i,j,k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            # We'll update one matrix at a time.
            
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each (i, j, r), we want to choose U[i*m+j, r] to minimize the error.
                # This is a quadratic in U[i*m+j, r].
                # Let a = U[i*m+j, r], then the error contribution is:
                # sum_{k} (a * V[j*p+k, r] * W[k*n+i, r] + rest - 1)^2
                # = sum_k (a * B_k + C_k - 1)^2 where B_k = V[j*p+k, r] * W[k*n+i, r], C_k = sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
                # The optimal a is (sum_k (1 - C_k) * B_k) / (sum_k B_k^2)
                
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            # Compute the optimal value for U[i*m+j, r]
                            numerator = 0.0
                            denominator = 0.0
                            for k in range(p):
                                B = V[j*p+k, r] * W[k*n+i, r]
                                C = 0.0
                                for r2 in range(R):
                                    if r2 != r:
                                        C += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                                numerator += (1 - C) * B
                                denominator += B * B
                            if denominator > 1e-10:
                                U[i*m+j, r] = numerator / denominator
                            else:
                                U[i*m+j, r] = 0.0
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            numerator = 0.0
                            denominator = 0.0
                            for i in range(n):
                                B = U[i*m+j, r] * W[k*n+i, r]
                                C = 0.0
                                for r2 in range(R):
                                    if r2 != r:
                                        C += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                                numerator += (1 - C) * B
                                denominator += B * B
                            if denominator > 1e-10:
                                V[j*p+k, r] = numerator / denominator
                            else:
                                V[j*p+k, r] = 0.0
                
                for k in range(p):
                    for i in range(n):
                        for r in range(R):
                            numerator = 0.0
                            denominator = 0.0
                            for j in range(m):
                                B = U[i*m+j, r] * V[j*p+k, r]
                                C = 0.0
                                for r2 in range(R):
                                    if r2 != r:
                                        C += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                                numerator += (1 - C) * B
                                denominator += B * B
                            if denominator > 1e-10:
                                W[k*n+i, r] = numerator / denominator
                            else:
                                W[k*n+i, r] = 0.0
                
                # Check if we have an exact solution (within tolerance)
                # And if all entries are close to integers or half-integers
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    # Check if entries are close to integers or half-integers
                    def is_valid(x):
                        # Check if x is close to an integer or half-integer
                        for target in [int(x), int(x) + 0.5, int(x) - 0.5]:
                            if abs(x - target) < 1e-4:
                                return True
                        return False
                    
                    valid = True
                    for i in range(nm):
                        for r in range(R):
                            if not is_valid(U[i, r]):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for j in range(mp):
                            for r in range(R):
                                if not is_valid(V[j, r]):
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        for k in range(pn):
                            for r in range(R):
                                if not is_valid(W[k, r]):
                                    valid = False
                                    break
                                if not valid:
                                    break
                    
                    if valid:
                        # Found a valid solution
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Convert to lists
                        U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(nm)]
                        V_list = [[float(V_rounded[j, r]) for r in range(R)] for j in range(mp)]
                        W_list = [[float(W_rounded[k, r]) for r in range(R)] for k in range(pn)]
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = R
                        break
            if best_rank == R:
                break
        if best_rank == R:
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
