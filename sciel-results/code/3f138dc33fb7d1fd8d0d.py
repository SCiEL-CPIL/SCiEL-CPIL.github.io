import numpy as np
import time
import random
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _solve_333(rank, max_restarts=50, time_limit=50.0):
    """
    Attempt to find a rank decomposition for 3x3x3 matrix multiplication.
    Returns (U, V, W) or None.
    """
    n, m, p = 3, 3, 3
    start_time = time.time()
    best_sol = None
    best_res = float('inf')
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # U: (rank, n, m), V: (rank, m, p), W: (rank, p, n)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(1234 + restart)
        # Initialize with small random values
        U = np.random.uniform(-1, 1, (rank, n, m))
        V = np.random.uniform(-1, 1, (rank, m, p))
        W = np.random.uniform(-1, 1, (rank, p, n))
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_r = x[:rank*n*m].reshape(rank, n, m)
            V_r = x[rank*n*m : rank*n*m + rank*m*p].reshape(rank, m, p)
            W_r = x[rank*n*m + rank*m*p :].reshape(rank, p, n)
            
            # T[i,j,k] = sum_c U[c,i,j] V[c,j,k] W[c,k,i]
            T = np.einsum('cij,cjk,cki->ijk', U_r, V_r, W_r)
            return (T - 1).flatten()
        
        try:
            # Use least_squares
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            res_norm = np.linalg.norm(result.fun)
            
            if res_norm < best_res:
                best_res = res_norm
                best_sol = result.x.copy()
                
                if res_norm < 1e-8:
                    break
                    
        except Exception:
            continue
            
    if best_sol is not None and best_res < 1e-8:
        U = best_sol[:rank*n*m].reshape(rank, n, m)
        V = best_sol[rank*n*m : rank*n*m + rank*m*p].reshape(rank, m, p)
        W = best_sol[rank*n*m + rank*m*p :].reshape(rank, p, n)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_rounded = np.einsum('cij,cjk,cki->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_rounded, 1, atol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Try to find a better decomposition for 3x3x3
    # If we find rank r for 3x3x3, we can use it for the first 3 columns of B
    # and add 9 terms for the 4th column.
    # Total rank = r + 9.
    # Schoolbook for 3x3x3 is 27. Total 36.
    # We want r < 27.
    
    found_333 = None
    for r in range(24, 27):
        if time.time() > deadline:
            break
        sol_333 = _solve_333(r, max_restarts=20, time_limit=20.0)
        if sol_333 is not None:
            found_333 = (r, sol_333)
            break
            
    if found_333 is not None:
        r_333, (U_333, V_333, W_333) = found_333
        # Construct 3x3x4 solution
        # U: (3*3, r_total), V: (3*4, r_total), W: (4*3, r_total)
        # The 3x3x3 part uses indices i in 0..2, j in 0..2, k in 0..2.
        # The 4th column (k=3) is computed with schoolbook:
        # C[i, 3] = sum_j A[i, j] B[j, 3]
        # This adds 9 terms.
        
        r_total = r_333 + 9
        
        U = np.zeros((n*m, r_total))
        V = np.zeros((m*p, r_total))
        W = np.zeros((p*n, r_total))
        
        # Fill in the 3x3x3 part
        # U_333 is (r_333, 3, 3) -> U[i*3+j, c] = U_333[c, i, j]
        for c in range(r_333):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, c] = U_333[c, i, j]
                    V[j*4+k, c] = V_333[c, j, k] # Wait, V index is j*p+k. p=4.
                    # V_333 is for 3x3x3, so k is 0..2.
                    # We need to map V_333[c, j, k] to V[j*4+k, c]
                    for k in range(3):
                        V[j*4+k, c] = V_333[c, j, k]
                    # W_333 is (r_333, 3, 3) -> W[k*3+i, c] = W_333[c, k, i]
                    for k in range(3):
                        for i in range(3):
                            W[k*3+i, c] = W_333[c, k, i]
                            
        # Fill in the 9 terms for the 4th column
        # Term index offset = r_333
        # We need to compute C[i, 3] = sum_j A[i, j] B[j, 3]
        # This is sum_{i,j} A[i,j] B[j,3] e_i e_3^T?
        # In the tensor form:
        # We need terms U_t, V_t, W_t such that
        # sum_t U_t[i,j] V_t[j,3] W_t[3,i] = delta_{i, i'} delta_{j, j'}?
        # Actually, we just need the sum to produce the correct C.
        # The tensor condition is sum_r U[r,i,j] V[r,j,k] W[r,k,i] = 1 for all i,j,k.
        # For k=3, we need sum_{t=r_333}^{r_total-1} U[t,i,j] V[t,j,3] W[t,3,i] = 1.
        # And for k=0,1,2, the sum over t=r_333.. must be 0.
        
        # Let's use the schoolbook terms for the 4th column.
        # For each i in 0..2, j in 0..2:
        # We want a term that contributes 1 to (i,j,3) and 0 elsewhere?
        # No, the term U[t,i,j] V[t,j,3] W[t,3,i] contributes to T[i,j,3].
        # We need sum_t ... = 1 for all i,j.
        # If we set U[t,i,j]=1, V[t,j,3]=1, W[t,3,i]=1 for specific i,j,
        # then for a fixed (i,j), only the term t corresponding to (i,j) contributes 1.
        # But we need to ensure it doesn't contribute to other k.
        # V[t,j,k] is 0 for k != 3.
        # W[t,k,i] is 0 for k != 3.
        # So these terms only affect k=3.
        
        t_offset = r_333
        idx = 0
        for i in range(3):
            for j in range(3):
                t = t_offset + idx
                U[i*3+j, t] = 1.0
                V[j*4+3, t] = 1.0
                W[3*3+i, t] = 1.0
                idx += 1
                
        # Verify the full solution
        U_full = U
        V_full = V
        W_full = W
        # Reshape to tensors
        U_t = U_full.T.reshape(r_total, 3, 3)
        V_t = V_full.T.reshape(r_total, 3, 4)
        W_t = W_full.T.reshape(r_total, 4, 3)
        
        T_check = np.einsum('cij,cjk,cki->ijk', U_t, V_t, W_t)
        
        if np.allclose(T_check, 1, atol=1e-9):
            # Convert to lists
            U_list = [[0]*r_total for _ in range(n*m)]
            V_list = [[0]*r_total for _ in range(m*p)]
            W_list = [[0]*r_total for _ in range(p*n)]
            
            for i in range(n*m):
                for c in range(r_total):
                    U_list[i][c] = float(U_full[i, c])
            for j in range(m*p):
                for c in range(r_total):
                    V_list[j][c] = float(V_full[j, c])
            for k in range(p*n):
                for c in range(r_total):
                    W_list[k][c] = float(W_full[k, c])
                    
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
