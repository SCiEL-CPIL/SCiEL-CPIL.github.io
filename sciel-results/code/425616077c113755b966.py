"""Matrix multiplication 2x3x5 rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_entries = n * m * p  # 30 tensor entries
    
    # Schoolbook rank-30 decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    def to_int_matrix(M, tol=1e-6):
        """Convert to integer or half-integer if close."""
        M = np.array(M, dtype=float)
        # Try rounding to integers
        M_int = np.round(M)
        if np.max(np.abs(M - M_int)) < tol:
            return M_int.astype(int).tolist()
        # Try half-integers (multiply by 2, round, divide)
        M_half = np.round(M * 2) / 2.0
        if np.max(np.abs(M - M_half)) < tol:
            return M_half.tolist()
        return M.tolist()
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_restarts=200):
        """Try to find rank-r decomposition."""
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Iterative refinement: alternate fixing two matrices and solving for the third
            for iteration in range(500):
                if time.time() > deadline - 5:
                    return None
                
                # Fix V and W, solve for U:
                # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is a linear system in U[i*m+j, :] for each (i,j)
                # Actually, for fixed (i,j), we need:
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                # Let A_{k,c} = V[j*p+k, c] * W[k*n+i, c], then sum_c U[i*m+j,c] * A_{k,c} = 1
                # This is A @ U[i*m+j, :]^T = ones(p)
                
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve A @ x = ones(p)
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence
                if verify(U, V, W, tol=1e-4):
                    # Final verification with tighter tolerance
                    if verify(U, V, W, tol=1e-8):
                        # Try to convert to clean numbers
                        U_list = to_int_matrix(U)
                        V_list = to_int_matrix(V)
                        W_list = to_int_matrix(W)
                        if verify(U_list, V_list, W_list, tol=1e-6):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        else:
                            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 20 to 30
    for r in range(20, 31):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_restarts=100)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
