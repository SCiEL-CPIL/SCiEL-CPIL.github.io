"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    total = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(total, 1.0, atol=1e-8)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin
    
    # Target tensor T of shape (n, m, p) with all 1s
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    # Indices for U, V, W
    # U: (n, m, c) -> index i*m + j
    # V: (m, p, c) -> index j*p + k
    # W: (p, n, c) -> index k*n + i
    
    best_result = None
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    # Try to find a decomposition of rank R
    # We start from a lower bound. The rank of 3x4x4 is at least 34 (Landsberg bound?) 
    # Actually, for 3x4x4, the rank is known to be 44? 
    # Let's try from 40 up to 48.
    
    for R in range(40, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Multiple random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(restart * 1000 + R)
            
            # Initialize V and W randomly
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Alternating Least Squares
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # 1. Solve for U
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                # For each (i,j,k), this is linear in U[i,j,c].
                # We construct the matrix A_U of shape (n*m*p, n*m*R)
                # Row idx: (i,j,k)
                # Col idx: (i',j',c)
                # A_U[row, col] = V[j,k,c] * W[k,i,c] if i==i' and j==j' else 0
                
                A_U = np.zeros((n*m*p, n*m*R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            for c in range(R):
                                col_idx = (i*m + j)*R + c
                                A_U[row_idx, col_idx] = V[j*p + k, c] * W[k*n + i, c]
                
                # Solve A_U * U_flat = T_flat
                U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
                U = U_flat.reshape(n*m, R)
                
                # 2. Solve for V
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                # Linear in V[j,k,c].
                # Matrix A_V of shape (n*m*p, m*p*R)
                A_V = np.zeros((n*m*p, m*p*R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            for c in range(R):
                                col_idx = (j*p + k)*R + c
                                A_V[row_idx, col_idx] = U[i*m + j, c] * W[k*n + i, c]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
                V = V_flat.reshape(m*p, R)
                
                # 3. Solve for W
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                # Linear in W[k,i,c].
                # Matrix A_W of shape (n*m*p, p*n*R)
                A_W = np.zeros((n*m*p, p*n*R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_idx = i*m*p + j*p + k
                            for c in range(R):
                                col_idx = (k*n + i)*R + c
                                A_W[row_idx, col_idx] = U[i*m + j, c] * V[j*p + k, c]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
                W = W_flat.reshape(p*n, R)
                
                # Check error
                if it % 5 == 0:
                    U3 = U.reshape(n, m, R)
                    V3 = V.reshape(m, p, R)
                    W3 = W.reshape(p, n, R)
                    total = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    err = np.linalg.norm(total - T)
                    
                    if err < 1e-8:
                        # Found a solution
                        if R < best_rank:
                            best_rank = R
                            best_U = U
                            best_V = V
                            best_W = W
                        break
            else:
                continue
            break
        
        # If we found a solution for this R, we can stop if R is small enough? 
        # No, we want the minimal rank. But if we find one, we can return it 
        # and try to reduce further in subsequent calls? 
        # The evaluator calls construct() once. 
        # So we should try to find the smallest R possible.
        # If we find a valid R, we should continue to try R-1? 
        # The loop is ascending. So we find the smallest R that works.
        # Once we find a valid R, we can break out of the R loop? 
        # Yes, because we are iterating R from 40 upwards. 
        # If R=40 works, we return rank 40.
        # If R=40 fails, we try 41.
        
        if best_rank == R:
            break

    # If no better than schoolbook was found, return schoolbook
    if best_rank == n * m * p:
        # Generate schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
