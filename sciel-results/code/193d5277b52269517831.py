"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact (within tolerance)"""
        n, m, p = DIMENSIONS
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    
                    expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                    actual = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def to_half_integer_matrix(mat, tol=1e-6):
        """Convert float matrix to half-integer matrix if entries are close to half-integers"""
        n_rows, n_cols = mat.shape
        result = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                val = mat[i, j]
                rounded = np.round(val * 2) / 2.0
                if abs(val - rounded) > tol:
                    return None
                if rounded == int(rounded):
                    row.append(Fraction(int(rounded), 1))
                else:
                    num = int(round(rounded * 2))
                    row.append(Fraction(num, 2))
            result.append(row)
        return result
    
    def als_optimize(U, V, W, n_iter=50):
        """ALS optimization for tensor decomposition"""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                return U, V, W
            
            # Update U
            for a in range(n * m):
                i_a, j_a = divmod(a, m)
                M_p = np.zeros((p, r))
                for k in range(p):
                    b = j_a * p + k
                    c = k * n + i_a
                    M_p[k, :] = V[b, :] * W[c, :]
                
                lam = 1e-6
                A_mat = M_p.T @ M_p + lam * np.eye(r)
                b_vec = M_p.T @ np.ones(p)
                
                try:
                    U[a, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    U[a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update V
            for b in range(m * p):
                j_b, k_b = divmod(b, p)
                M_p = np.zeros((n, r))
                for i in range(n):
                    a = i * m + j_b
                    c = k_b * n + i
                    M_p[i, :] = U[a, :] * W[c, :]
                
                lam = 1e-6
                A_mat = M_p.T @ M_p + lam * np.eye(r)
                b_vec = M_p.T @ np.ones(n)
                
                try:
                    V[b, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    V[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update W
            for c in range(p * n):
                k_c, i_c = divmod(c, n)
                M_p = np.zeros((m, r))
                for j in range(m):
                    a = i_c * m + j
                    b = j * p + k_c
                    M_p[j, :] = U[a, :] * V[b, :]
                
                lam = 1e-6
                A_mat = M_p.T @ M_p + lam * np.eye(r)
                b_vec = M_p.T @ np.ones(m)
                
                try:
                    W[c, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    W[c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        return U, V, W
    
    def compute_error(U, V, W):
        """Compute the Frobenius norm of the residual"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        Approx = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            Approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        return np.linalg.norm(T - Approx)
    
    best_result = None
    
    for r in range(95, 100):
        if time.time() > deadline - 30:
            break
        
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            U = U / np.linalg.norm(U)
            V = V / np.linalg.norm(V)
            W = W / np.linalg.norm(W)
            
            U, V, W = als_optimize(U, V, W, n_iter=30)
            
            err = compute_error(U, V, W)
            
            if err < 1e-5:
                U_rounded = to_half_integer_matrix(U)
                V_rounded = to_half_integer_matrix(V)
                W_rounded = to_half_integer_matrix(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
                        best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        break
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
