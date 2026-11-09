import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    timeout = 110.0
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 18 to 90
    for rank in range(18, best_rank + 1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Multiple restarts
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + restart + rank) # Deterministic for reproducibility
            
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # ALS
            for it in range(200):
                if time.time() - start_time > timeout:
                    break
                    
                # Solve U
                M = np.zeros((rank, dim_v * dim_w))
                for r in range(rank):
                    M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Solve V
                M = np.zeros((rank, dim_u * dim_w))
                for r in range(rank):
                    M[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                T_flat = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Solve W
                M = np.zeros((rank, dim_u * dim_v))
                for r in range(rank):
                    M[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                T_flat = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
            # Check error
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
            err = np.linalg.norm(T_recon - T)
            if err < 1e-5:
                # Round
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_recon_int = np.zeros_like(T)
                for r in range(rank):
                    T_recon_int += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
                    
                if np.allclose(T_recon_int, T, atol=1e-9):
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            break
        if time.time() - start_time > timeout:
            break
            
    if best_U is None:
        best_rank = n * m * p
        best_U = np.zeros((dim_u, best_rank))
        best_V = np.zeros((dim_v, best_rank))
        best_W = np.zeros((dim_w, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, col] = 1
                    best_V[b, col] = 1
                    best_W[c, col] = 1
                    col += 1
    else:
        pass
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
