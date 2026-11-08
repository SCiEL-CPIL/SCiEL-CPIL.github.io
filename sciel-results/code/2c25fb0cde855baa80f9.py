"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=2, p=4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, i in [0,m), j in [0,n)  -> a in [0, m*n)
    # b = j*p + k, j in [0,n), k in [0,p)  -> b in [0, n*p)
    # c = k*n + i, k in [0,p), i in [0,m)  -> c in [0, p*m)
    
    # Wait, from seed: U[i*m+j], V[j*p+k], W[k*n+i]
    # So m=2 (rows of A), n=4 (cols of A / rows of B), p=4 (cols of B)
    # A is m x n = 2 x 4
    # B is n x p = 4 x 4
    # C is m x p = 2 x 4
    
    # U index: i*m + j where i in [0,m)=2, j in [0,n)=4 -> 8 entries
    # V index: j*p + k where j in [0,n)=4, k in [0,p)=4 -> 16 entries  
    # W index: k*n + i where k in [0,p)=4, i in [0,m)=2 -> 8 entries? 
    # But task says W shape is (p*n, rank) = (16, rank). 
    # k*n+i: k in [0,4), i in [0,2) -> max is 3*4+1=13, so 14 values? No...
    # Actually k*n+i with n=4, m=2: k in 0..3, i in 0..1, so values 0..13, that's 14 distinct? 
    # No: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # So indices used are {0,1,4,5,8,9,12,13}, but W has shape (16,rank) so all 16 rows exist.
    
    # Build T
    T = np.zeros((m*n, n*p, p*m))  # (8, 16, 8)? But W is (16, rank)...
    
    # Hmm, the task says W has shape (p*n, rank) = (16, rank). But from the formula W[k*n+i], 
    # with k in [0,p)=4, i in [0,m)=2, we only use 8 of the 16 entries? That seems odd.
    # Let me re-read: "W[k*n+i,r]" - maybe n here is different?
    
    # Actually looking again at seed: W has shape (p*n, rank) and is indexed by W[k*n+i][column]
    # where k in range(p), i in range(n)? No, in seed: for i in range(n): for j in range(m): for k in range(p):
    # W[k*n+i][column] = 1
    # So i is in range(n)=4, not range(m)=2!
    
    # I think I mislabeled. Let me re-examine seed:
    # n, m, p = DIMENSIONS = [2, 4, 4]
    # So n=2, m=4, p=4
    # U[i*m+j] with i in range(n)=2, j in range(m)=4 -> 8 entries, shape (n*m, rank)=(8,rank) ✓
    # V[j*p+k] with j in range(m)=4, k in range(p)=4 -> 16 entries, shape (m*p, rank)=(16,rank) ✓
    # W[k*n+i] with k in range(p)=4, i in range(n)=2 -> 8 entries used, but shape (p*n,rank)=(8,rank) ✓
    
    # Ah! n=2, m=4, p=4. So A is n x m = 2 x 4, B is m x p = 4 x 4, C is n x p = 2 x 4.
    # W index: k*n+i, k in [0,4), i in [0,2), so indices 0..7, shape (8, rank). 
    # But task says W shape is (p*n, rank) = (4*2, rank) = (8, rank). Yes!
    
    n, m, p = 2, 4, 4
    # T[a, b, c] where:
    # a = i*m + j, i in [0,n), j in [0,m) -> a in [0, n*m) = [0,8)
    # b = j*p + k, j in [0,m), k in [0,p) -> b in [0, m*p) = [0,16)
    # c = k*n + i, k in [0,p), i in [0,n) -> c in [0, p*n) = [0,8)
    
    T = np.zeros((n*m, m*p, p*n))  # (8, 16, 8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now find rank-minimal decomposition: T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # i.e., T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from small to large
    for rank in range(1, best_rank + 1):
        # Try multiple random initializations
        for attempt in range(50):
            # Initialize with small random values
            np.random.seed(attempt)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(200):
                # Update U: for each a, U[a,:] should satisfy sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                # For fixed a, this is linear in U[a,:]: 
                # sum_t U[a,t] * (sum_b V[b,t]*W[c,t] * T[a,b,c] / something)... 
                # Actually, fix V and W, solve for U.
                # T[:,b,c] = sum_t U[:,t] * V[b,t] * W[c,t]
                # For each b,c: T[:,b,c] = U * (V[b,:]*W[c,:])
                # So U * (V[b,:]*W[c,:]) = T[:,b,c] for all b,c
                # This is overdetermined. Use least squares.
                
                # Stack all (b,c) pairs: we have 16*8=128 equations
                # A_matrix: (128, rank), bvec: (128, n*m)
                A_mat = np.zeros((m*p * p*n, rank))
                b_vec = np.zeros((m*p * p*n, n*m))
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A_mat[idx, :] = V[b, :] * W[c, :]
                        b_vec[idx, :] = T[:, b, c]
                        idx += 1
                # Solve U * A_mat.T = b_vec.T -> U = b_vec.T @ pinv(A_mat.T)
                # Or: A_mat @ U.T = b_vec.T -> U.T = pinv(A_mat) @ b_vec.T
                try:
                    U_new = np.linalg.lstsq(A_mat, b_vec.T, rcond=None)[0].T
                except:
                    break
                U = U_new
                
                # Update V
                A_mat = np.zeros((n*m * p*n, rank))
                b_vec = np.zeros((n*m * p*n, m*p))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_mat[idx, :] = U[a, :] * W[c, :]
                        b_vec[idx, :] = T[a, :, c]
                        idx += 1
                try:
                    V_new = np.linalg.lstsq(A_mat, b_vec.T, rcond=None)[0].T
                except:
                    break
                V = V_new
                
                # Update W
                A_mat = np.zeros((n*m * m*p, rank))
                b_vec = np.zeros((n*m * m*p, p*n))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[idx, :] = U[a, :] * V[b, :]
                        b_vec[idx, :] = T[a, b, :]
                        idx += 1
                try:
                    W_new = np.linalg.lstsq(A_mat, b_vec.T, rcond=None)[0].T
                except:
                    break
                W = W_new
                
                # Check convergence
                T_approx = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T - T_approx) / max(1, np.linalg.norm(T))
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                T_approx = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_approx, T, atol=1e-9):
                    best_rank = rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break
        
        if best_result is not None and best_rank <= rank:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
    
    return best_result
# EVOLVE-BLOCK-END
