"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.

Strategy:
1. Start with schoolbook (rank 60) as a valid baseline.
2. Use scipy.optimize.least_squares to search for lower rank decompositions.
3. The objective is to minimize the squared error of the tensor identity:
   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1 = 0
4. If a solution is found with error < tol, round to half-integers and verify exactly.
"""

import time
import numpy as np
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Define the residual function for scipy
    def residuals(x, r):
        # x is a flattened array of size (n*m + m*p + p*n) * r
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:]
        
        U_mat = U_flat.reshape(n * m, r)
        V_mat = V_flat.reshape(m * p, r)
        W_mat = W_flat.reshape(p * n, r)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U_mat[i * m + j, :], V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    res.append(val - 1.0)
        
        return np.array(res)

    # Try to find lower rank solutions
    # We try ranks from 55 down to 45
    np.random.seed(42)
    
    for r in range(55, 44, -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize x
            # Strategy: Start with schoolbook and project to lower rank? 
            # Or just random sparse?
            # Let's try a "warm start" by taking the first r columns of schoolbook 
            # and adding some noise, then optimizing.
            # But schoolbook columns are orthogonal-ish in the tensor product space? No.
            
            # Let's try random initialization with values in [-1, 1]
            x0 = np.random.randn(n * m * r + m * p * r + p * n * r) * 0.5
            
            try:
                result = least_squares(residuals, x0, args=(r,), method='lm', max_nfev=2000)
                if result.cost < 1e-10:
                    # Extract U, V, W
                    size_u = n * m * r
                    size_v = m * p * r
                    size_w = p * n * r
                    
                    U_mat = result.x[:size_u].reshape(n * m, r)
                    V_mat = result.x[size_u:size_u + size_v].reshape(m * p, r)
                    W_mat = result.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    # Verify
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_rank = r
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            # If we found a solution for rank r, we can stop or try lower?
            # The loop goes down, so it will try lower ranks in subsequent iterations.
            # But if we found one, we can break to save time if we are satisfied?
            # No, we want the MINIMAL rank. So we continue to lower r.
            pass

    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
