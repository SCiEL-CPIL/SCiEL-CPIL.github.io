"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition (rank = n*m*p = 36)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_best, V_best, W_best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: For a given rank r, try random initializations and use ALS
    # But we need exact (half-integer) solutions, so we'll verify with exact arithmetic
    
    def verify_exact(U, V, W):
        """Verify that U,V,W give an exact decomposition of T with half-integer entries."""
        r = U.shape[1]
        # Check entries are half-integers
        for mat in [U, V, W]:
            for val in mat.flat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        # Check tensor reconstruction
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        return True
    
    def try_rank(r, max_iters=500, n_trials=10):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS: alternate fixing two factors and solving for the third
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                # Fix V, W, solve for U
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed V,W: T_a = U_a @ (V * W^T) reshaped... 
                # Actually: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # So for each a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # This is: vec(T[a,:,:]) = sum_r U[a,r] * vec(outer(V[:,r], W[:,r]))
                
                # Build matrix M_U where M_U[a, r] relates...
                # Let's do it properly:
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Fix V,W. For each a, we need:
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
                # This is a linear system in U[a,0..r-1]
                
                # Build A_U: shape (m*p * p*n, r)
                # A_U[b*p*n + c, r] = V[b,r] * W[c,r]
                A_U = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        for r_idx in range(r):
                            A_U[b * p*n + c, r_idx] = V[b, r_idx] * W[c, r_idx]
                
                # RHS: T reshaped
                rhs_U = T.reshape(n*m, m*p * p*n).T  # shape (m*p*p*n, n*m)
                
                # Solve A_U @ U.T = rhs_U
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_U, rhs_U, rcond=None)
                    U = U_new.T  # shape (n*m, r)
                except:
                    break
                
                # Similarly for V
                A_V = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        for r_idx in range(r):
                            A_V[a * p*n + c, r_idx] = U[a, r_idx] * W[c, r_idx]
                rhs_V = T.reshape(m*p, n*m * p*n).T
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_V, rhs_V, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Similarly for W
                A_W = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        for r_idx in range(r):
                            A_W[a * m*p + b, r_idx] = U[a, r_idx] * V[b, r_idx]
                rhs_W = T.reshape(p*n, n*m * m*p).T
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_W, rhs_W, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-6:
                    # Try to round to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    T_recon2 = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.max(np.abs(T_recon2 - T)) < 1e-9:
                        return U_round, V_round, W_round
            
            # Try rounding even if not fully converged
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            T_recon = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
            if np.max(np.abs(T_recon - T)) < 1e-9:
                return U_round, V_round, W_round
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, the tensor has 36 non-zero entries, so rank >= 1
    # A better lower bound: the tensor rank is at least the matrix rank of any slice
    # For 3x3x3, known min rank is 23. For 3x3x4, let's try from 20 up
    
    for r in range(15, best_rank + 1):
        if time.time() > deadline - 5:
            break
        result = try_rank(r, max_iters=200, n_trials=5)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break
    
    # Convert to lists
    U_list = [[float(U_best[i, j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[float(V_best[i, j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[float(W_best[i, j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
