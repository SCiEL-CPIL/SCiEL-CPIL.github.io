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
    # Strategy: Search for low rank (14 to 24) using structured random initialization and linear solving for W.
    
    # Pre-compute indices for fast verification
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (12, R)
    # W shape: (p*n, R) = (8, R)
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let A_ik = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] for all i,j,k.
    
    for R in range(14, 25):
        if time.time() - start_time > deadline:
            break
        
        import random
        random.seed(42 + R)
        
        # Use a limited number of attempts for lower ranks, more for higher
        attempts = 5000 if R < 20 else 1000
        
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            
            # Random sparse initialization for U and V
            # Using {-1, 0, 1} to keep numbers small
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Solve for W
            # For each (i, k), we have a linear system in W[k*n+i, :]
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # Let M_ik be a matrix of size (m, R) where M_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # We need M_ik @ W_ik = 1 (vector of ones of size m)
            
            W = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M_ik
                    M_ik = np.zeros((m, R), dtype=float)
                    for j in range(m):
                        M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M_ik @ w = 1
                    try:
                        # Use lstsq to find minimum norm solution
                        w, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if w is half-integer
                    # w * 2 should be integer
                    w2 = w * 2
                    nearest = np.round(w2)
                    if not np.allclose(w2, nearest, atol=1e-9):
                        valid = False
                        break
                    
                    # Verify exactness with rounded values
                    w_rounded = nearest / 2.0
                    if not np.allclose(M_ik @ w_rounded, np.ones(m), atol=1e-9):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w_rounded
                
                if not valid:
                    break
            
            if valid:
                # Convert to lists of fractions/integers for the output
                U_out = [[int(x) for x in row] for row in U]
                V_out = [[int(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
                
                return {"U": U_out, "V": V_out, "W": W_out}
        
        # If not found, try next R
    
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
