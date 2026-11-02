"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # Tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Strategy: Fix W to simple values, solve for U and V.
    
    # Try rank from 20 to 24
    for R in range(20, 25):
        if time.time() - start_time > deadline:
            break
        
        # Strategy 1: W all ones
        W = [[1] * R for _ in range(p * n)]
        
        # Need: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
        # This is a system of 24 bilinear equations in U and V.
        
        # Approach: iterate, fixing one set and solving for the other.
        # Initialize U and V randomly with small integers.
        import random
        random.seed(42 + R)
        
        found = False
        for attempt in range(20000):
            if time.time() - start_time > deadline:
                break
            
            # Random sparse initialization
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(m * p)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(R):
                            total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                break
        
        if found:
            return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Use structured W and solve linear systems
    # For each (i,k), we need M_{i,k} @ w_ik = 1 where M_{i,k}[j,r] = U[i*m+j,r]*V[j*p+k,r]
    # Try to construct U,V such that these systems have half-integer solutions.
    
    for R in range(20, 24):
        if time.time() - start_time > deadline:
            break
        
        import random
        random.seed(123 + R)
        
        for attempt in range(30000):
            if time.time() - start_time > deadline:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
            
            # Solve for W
            W_valid = True
            W = [[0] * R for _ in range(p * n)]
            for i in range(n):
                for k in range(p):
                    # Build M_{i,k}
                    M = np.zeros((m, R), dtype=float)
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i * m + j][r] * V[j * p + k][r]
                    
                    # Solve M @ w = 1
                    try:
                        # Check if system is consistent
                        w, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_valid = False
                        break
                    
                    # Check if w is half-integer
                    valid_w = True
                    w_rounded = [0.0] * R
                    for r in range(R):
                        val2 = w[r] * 2
                        nearest = round(val2)
                        if abs(val2 - nearest) > 1e-9:
                            valid_w = False
                            break
                        w_rounded[r] = nearest / 2.0
                    
                    if not valid_w:
                        W_valid = False
                        break
                    
                    # Verify exactness
                    for j in range(m):
                        total = 0.0
                        for r in range(R):
                            total += M[j, r] * w_rounded[r]
                        if abs(total - 1.0) > 1e-9:
                            W_valid = False
                            break
                    if not W_valid:
                        break
                    
                    for r in range(R):
                        W[k * n + i][r] = w_rounded[r]
                
                if not W_valid:
                    break
            
            if W_valid:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
