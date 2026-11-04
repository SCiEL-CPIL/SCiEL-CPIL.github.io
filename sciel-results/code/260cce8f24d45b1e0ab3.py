"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3) : row of first matrix
    - j in [0, m) = [0, 3) : column of first matrix / row of second matrix  
    - k in [0, p) = [0, 7) : column of second matrix
    
    T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    
    The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    With W following the trace convention W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 140.0  # Leave safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 63
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i, c] * V_mat[j, c] * W_mat[k, c] for c in range(U_mat.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def convert_to_interface(U_mat, V_mat, W_mat):
        """Convert from (n x r), (m x r), (p x r) to interface format."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        U_list = [[0] * r for _ in range(n * m)]
        V_list = [[0] * r for _ in range(m * p)]
        W_list = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U_list[i * m + j][c] = U_mat[i, c]
        
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V_list[j * p + k][c] = V_mat[j, c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W_list[k * n + i][c] = W_mat[k, c]
        
        return U_list, V_list, W_list

    def round_to_half_integer(mat):
        """Round matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def als_solve(U, V, W, r, iterations=50):
        """
        Alternating Least Squares to minimize sum (1 - sum_c U[i,c]V[j,c]W[k,c])^2
        """
        n, m, p = DIMENSIONS
        for it in range(iterations):
            # Solve for U
            # For each i, we want sum_c U[i,c] * (V[j,c] * W[k,c]) = 1 for all j,k
            # This is a linear system for each i:
            # Let M_U be a matrix where row (j,k) is V[j,:] * W[k,:] (element-wise)
            # Then U[i,:] * M_U.T = 1_vector
            # M_U has shape (m*p, r)
            # We want to find U[i,:] such that U[i,:] @ M_U.T @ M_U = U[i,:] @ M_U.T @ 1
            # Actually, simpler:
            # sum_c U[i,c] V[j,c] W[k,c] = 1
            # This is equivalent to:
            # For fixed i, we have a linear system in U[i,:]:
            # sum_c U[i,c] (V[j,c] W[k,c]) = 1
            # Let A_U be a matrix of size (m*p) x r where A_U[j*p+k, c] = V[j,c] * W[k,c]
            # Then A_U @ U[i,:] = 1 (vector of ones of length m*p)
            
            A_U = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    A_U[j * p + k, :] = V[j, :] * W[k, :]
            
            # Solve A_U @ U[i] = 1
            # Use least squares
            for i in range(n):
                U[i, :] = np.linalg.lstsq(A_U, np.ones(m * p), rcond=None)[0]
            
            # Solve for V
            # A_V[j*p+k, c] = U[i,c] * W[k,c] ? No.
            # sum_c U[i,c] V[j,c] W[k,c] = 1
            # For fixed j:
            # sum_c V[j,c] (U[i,c] W[k,c]) = 1 for all i,k
            # Let A_V be a matrix of size (n*p) x r where A_V[i*p+k, c] = U[i,c] * W[k,c]
            # Then A_V @ V[j,:] = 1 (vector of ones of length n*p)
            
            A_V = np.zeros((n * p, r))
            for i in range(n):
                for k in range(p):
                    A_V[i * p + k, :] = U[i, :] * W[k, :]
            
            for j in range(m):
                V[j, :] = np.linalg.lstsq(A_V, np.ones(n * p), rcond=None)[0]
                
            # Solve for W
            # For fixed k:
            # sum_c W[k,c] (U[i,c] V[j,c]) = 1 for all i,j
            # Let A_W be a matrix of size (n*m) x r where A_W[i*m+j, c] = U[i,c] * V[j,c]
            # Then A_W @ W[k,:] = 1 (vector of ones of length n*m)
            
            A_W = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    A_W[i * m + j, :] = U[i, :] * V[j, :]
            
            for k in range(p):
                W[k, :] = np.linalg.lstsq(A_W, np.ones(n * m), rcond=None)[0]
                
        return U, V, W

    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for 3x3x7 is at least max(9, 21, 21) = 21
    # We try a range of ranks
    
    for r in range(21, 63):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        n_init = 3
        for init in range(n_init):
            if time.time() > deadline - 5:
                break
            
            U_init = np.random.randn(n, r) * 0.5 + 0.5
            V_init = np.random.randn(m, r) * 0.5 + 0.5
            W_init = np.random.randn(p, r) * 0.5 + 0.5
            
            U_sol, V_sol, W_sol = als_solve(U_init, V_init, W_init, r, iterations=20)
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_sol[i], V_sol[j] * W_sol[k])
                        err += (val - 1.0) ** 2
                        
            if err < 1e-6:
                # Try rounding to half-integers
                U_rounded = round_to_half_integer(U_sol)
                V_rounded = round_to_half_integer(V_sol)
                W_rounded = round_to_half_integer(W_sol)
                
                # Verify rounded solution
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    U_list, V_list, W_list = convert_to_interface(U_rounded, V_rounded, W_rounded)
                    best_rank = r
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    break
        
        if best_rank <= r:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
