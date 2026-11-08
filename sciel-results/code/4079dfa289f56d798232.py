"""Program for 3x4x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: must finish well before 150s
START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, safety margin

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = 3, 4, 6
    
    # Try to find a low-rank decomposition using optimization
    # The tensor for (C = A * B) has components T[i,k,j] = delta[i+k] where A is n x m, B is m x p
    # Rank-1 term c gives: U[i, c] * V[j, c] * W[k, c] = 1 when j is the middle index linking i and k
    # Actually: (C)_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{i,j,k} = 1 if the indices match, i.e., it's the standard multiplication tensor
    # T_{i,j,k} = delta for the bilinear form: A_{i,j} * B_{j,k}
    
    # The multiplication tensor has entries: T[i, j, k] = 1 for all i in [0,n), j in [m), k in [0,p)
    # where the bilinear map is: f(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T_{i,j,k} = 1 for all valid i,j,k (since every triple (i,j,k) contributes)
    
    # We want: sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k] = 1 for all i,j,k
    
    # Try small ranks and use iterative optimization
    best_rank = n * m * p  # schoolbook as fallback
    best_solution = None
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions using alternating minimization / optimization
    # We'll try ranks from a reasonable lower bound upward
    # The tensor has size n*m*p = 3*4*6 = 72 elements (all ones in the 3-mode fiber structure)
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Known results for small cases, but let's try to optimize
    
    for rank in range(max(n*m, m*p, p*n), n*m*p + 1):
        if _time_left() < 30.0:
            break
            
        # Multiple random restarts
        found = False
        n_restarts = min(50, int(max(1, _time_left() / 5.0)))
        
        for restart in range(n_restarts):
            if _time_left() < 10.0:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Scale to get closer to target
            U *= (1.0 / np.sqrt(rank)) * 3.0
            V *= (1.0 / np.sqrt(rank)) * 3.0
            W *= (1.0 / np.sqrt(rank)) * 3.0
            
            # Alternating minimization / fixed-point iteration
            # Minimize ||U ⊗ V ⊗ W - T||_F
            # We can do this via iterative projection (like tensor CP decomposition)
            
            max_iter = 2000
            tol = 1e-8
            prev_err = float('inf')
            
            # Target tensor T[i*m+j, k*n+i2, c_index] ... 
            # Actually let's reshape properly
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # The reconstruction: for each c, outer product U[:,c] (n*m), V[:,c] (m*p), W[:,c] (p*n)
            # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # Let's define the target explicitly as a 3-tensor T[i, j, k] = 1
            # And compute error as sum over all (i,j,k) of (sum_c ... - 1)^2
            
            for iteration in range(max_iter):
                if _time_left() < 5.0:
                    break
                    
                # Update U: for each (i*m+j), minimize over U[i*m+j, :]
                # U[:,c] appears in: sum_c U[imj, c] * V[jp+k, c] * W[kn+i, c] for each (j, k, i)
                # This is linear in U[imj, :] for fixed V, W
                # For a fixed (i, j): error contribution from terms involving U[i*m+j, :]
                # sum_k [sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1]^2
                # Let x = U[i*m+j, :], a_c(k) = V[j*p+k, c] * W[k*n+i, c]
                # minimize sum_k (sum_c x_c * a_c(k) - 1)^2
                # This is: ||A @ x - 1||^2 where A has rows indexed by k, columns by c
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                for i in range(n):
                    for j in range(m):
                        # Build A matrix (p x rank)
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        b_vec = np.ones(p)
                        # Solve: min ||A_mat @ x - b_vec||^2
                        try:
                            x_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            x_sol = np.random.randn(rank) * 0.1
                        U[i * m + j, :] = x_sol
                    
                    for i2 in range(n):
                        pass  # already handled above
            
            # Actually, let's do a cleaner alternating least squares
            for iteration in range(max_iter):
                if _time_left() < 5.0:
                    break
                    
                # Update U: for each row (i*m+j)
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_vec = np.ones(p)
                        try:
                            U[i * m + j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update V: for each row (j*p+k)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_vec = np.ones(n)
                        try:
                            V[j * p + k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update W: for each row (k*n+i)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_vec = np.ones(m)
                        try:
                            W[k * n + i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (val - 1.0) ** 2
                err = np.sqrt(err)
                
                if abs(err - prev_err) < tol or err < tol:
                    if err < 1e-6:
                        found = True
                    break
                prev_err = err
            
            if found:
                best_rank = rank
                best_solution = (U.copy(), V.copy(), W.copy())
                break
        
        if found:
            break
    
    # Convert to lists with rounded values for integer/half-integer entries
    U_best, V_best, W_best = best_solution
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(2.0 * x) / 2.0
    
    U_list = [[float(round_to_half(U_best[i, j])) for j in range(U_best.shape[1])] for i in range(U_best.shape[0])]
    V_list = [[float(round_to_half(V_best[i, j])) for j in range(V_best.shape[1])] for i in range(V_best.shape[0])]
    W_list = [[float(round_to_half(W_best[i, j])) for j in range(W_best.shape[1])] for i in range(W_best.shape[0])]
    
    # Verify the rounded solution is still valid
    n_rounded = U_best.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(n_rounded):
                    val += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U_list = [[float(U_sb[i, j]) for j in range(U_sb.shape[1])] for i in range(U_sb.shape[0])]
        V_list = [[float(V_sb[i, j]) for j in range(V_sb.shape[1])] for i in range(V_sb.shape[0])]
        W_list = [[float(W_sb[i, j]) for j in range(W_sb.shape[1])] for i in range(W_sb.shape[0])]
        n_rounded = U_sb.shape[1]
    
    return {"U": U_list, "V": V_list, "W": W_list}
