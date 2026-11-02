"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the matrix multiplication tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 iff exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Compute flattening ranks
    T1 = tensor.reshape(n * m, m * p * p * n)
    T2 = tensor.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    schoolbook_rank = n * m * p  # 30
    
    # Try to find a decomposition at rank = lower_bound
    # If successful, use it; otherwise fall back to schoolbook
    best_rank = schoolbook_rank
    best_U = best_V = best_W = None
    
    # Attempt rank reduction from lower_bound up to schoolbook_rank - 1
    for r in range(lower_bound, schoolbook_rank):
        # Try to find rank-r decomposition using numerical optimization
        # Initialize with random matrices and use gradient-free approach
        # Since the problem is small, we can try multiple random initializations
        
        success = False
        for trial in range(50):
            np.random.seed(trial)
            # Random initialization
            U_init = np.random.randn(n * m, r)
            V_init = np.random.randn(m * p, r)
            W_init = np.random.randn(p * n, r)
            
            # Use alternating least squares to refine
            # Fix V, W -> solve for U; fix U, W -> solve for V; fix U, V -> solve for W
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for iteration in range(200):
                # Fix V, W: for each a, T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # Group by a: for fixed a, we have a matrix M_a[b,c] = T[a,b,c]
                # M_a = sum_d U[a,d] (V[:,d] ⊗ W[:,d])
                # This is linear in U[a,:] if we define features F_d[b,c] = V[b,d]*W[c,d]
                
                # Build feature matrix: for each d, F_d is (m*p) x (p*n)
                # Then M_a = sum_d U[a,d] * F_d
                # So [M_a flattened] = U[a,:] @ F_d_stacked
                # This is a linear system: F_stacked @ U[a,:] = M_a_flattened
                
                # Build F_stacked: shape (m*p*p*n, r)
                # F_stacked[b*(p*n)+c, d] = V[b,d] * W[c,d]
                F_stacked = np.zeros((m * p * p * n, r))
                for b in range(m * p):
                    for c in range(p * n):
                        for d in range(r):
                            F_stacked[b * (p * n) + c, d] = V_cur[b, d] * W_cur[c, d]
                
                # For each a, solve F_stacked @ U[a,:] = T[a,:,:].flatten()
                # Use least squares
                U_new = np.zeros((n * m, r))
                for a in range(n * m):
                    target = tensor[a, :, :].flatten()
                    # Solve F_stacked @ x = target via least squares
                    x, residuals, rank_f, sv = np.linalg.lstsq(F_stacked, target, rcond=None)
                    U_new[a, :] = x
                U_cur = U_new
                
                # Fix U, W: for each b, T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For fixed b: M_b[a,c] = T[a,b,c] = sum_d V[b,d] (U[:,d] ⊗ W[:,d])
                # G_d[a,c] = U[a,d] * W[c,d]
                G_stacked = np.zeros((n * m * p * n, r))
                for a in range(n * m):
                    for c in range(p * n):
                        for d in range(r):
                            G_stacked[a * (p * n) + c, d] = U_cur[a, d] * W_cur[c, d]
                
                V_new = np.zeros((m * p, r))
                for b in range(m * p):
                    target = tensor[:, b, :].flatten()
                    x, residuals, rank_g, sv = np.linalg.lstsq(G_stacked, target, rcond=None)
                    V_new[b, :] = x
                V_cur = V_new
                
                # Fix U, V: for each c, T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For fixed c: H_c[a,b] = T[a,b,c] = sum_d W[c,d] (U[:,d] ⊗ V[:,d])
                # H_d[a,b] = U[a,d] * V[b,d]
                H_stacked = np.zeros((n * m * m * p, r))
                for a in range(n * m):
                    for b in range(m * p):
                        for d in range(r):
                            H_stacked[a * (m * p) + b, d] = U_cur[a, d] * V_cur[b, d]
                
                W_new = np.zeros((p * n, r))
                for c in range(p * n):
                    target = tensor[:, :, c].flatten()
                    x, residuals, rank_h, sv = np.linalg.lstsq(H_stacked, target, rcond=None)
                    W_new[c, :] = x
                W_cur = W_new
                
                # Check convergence
                error = 0.0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            val = sum(U_cur[a, d] * V_cur[b, d] * W_cur[c, d] for d in range(r))
                            error += (val - tensor[a, b, c]) ** 2
                error = np.sqrt(error)
                
                if error < 1e-8:
                    success = True
                    break
            
            if success:
                # Round to nearest half-integer if close
                U_rounded = np.round(U_cur * 2) / 2
                V_rounded = np.round(V_cur * 2) / 2
                W_rounded = np.round(W_cur * 2) / 2
                
                # Verify rounded version
                error_r = 0.0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            val = sum(U_rounded[a, d] * V_rounded[b, d] * W_rounded[c, d] for d in range(r))
                            error_r += (val - tensor[a, b, c]) ** 2
                error_r = np.sqrt(error_r)
                
                if error_r < 1e-6:
                    best_rank = r
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    if best_rank < schoolbook_rank and best_U is not None:
        rank = best_rank
        U = best_U
        V = best_V
        W = best_W
    else:
        # Fall back to schoolbook
        rank = schoolbook_rank
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
