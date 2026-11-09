"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Build target tensor T[u, v, w] = 1 if (u,v,w) corresponds to valid (i,j,k)
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    def check_half_integer(U, V, W):
        """Round to nearest 0.5 and verify exact decomposition."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        try:
            T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None
    
    def structured_init(rank, seed):
        """Structured initialization exploiting the 3-term sum structure.
        
        For each output (i,k), we need sum_j A[i,j]*B[j,k].
        We initialize factors to approximate this structure.
        """
        rng = np.random.RandomState(seed)
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # Assign rank slots: for each (i,k) pair, allocate ~m slots for the j-sum
        # But we can share across k for the same i, and across i for the same k
        # Let's use a block structure: rank = n * p * something + shared
        
        # Simple structured init: for each (i,j,k) triple, create a factor
        # but allow sharing via random small perturbations
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        
        # Add random structure for remaining rank slots
        for c in range(col, rank):
            for r in range(dim_u):
                U[r, c] = rng.choice([-1, 0, 1])
            for r in range(dim_v):
                V[r, c] = rng.choice([-1, 0, 1])
            for r in range(dim_w):
                W[r, c] = rng.choice([-1, 0, 1])
        
        return U, V, W
    
    def random_init(rank, seed):
        rng = np.random.RandomState(seed)
        U = rng.randn(dim_u, rank) * 0.1
        V = rng.randn(dim_v, rank) * 0.1
        W = rng.randn(dim_w, rank) * 0.1
        return U, V, W
    
    def gradient_descent(U, V, W, steps, lr, seed=None):
        """Polish the decomposition using gradient descent."""
        for step in range(steps):
            if time.time() > deadline:
                break
            try:
                Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                Err = T - Rec
                
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
            except Exception:
                break
        
        return U, V, W
    
    # Try to find a lower rank decomposition
    # Lower bound: rank >= max(dim_u, dim_v, dim_w) = 30
    # Try ranks from 30 upward, but focus on feasible ranges
    
    best_result = None
    best_rank = n * m * p  # 60
    
    # Try structured + gradient approach for ranks below schoolbook
    for rank in range(30, min(60, best_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 100 + attempt
            if attempt < 2:
                U, V, W = structured_init(rank, seed)
            else:
                U, V, W = random_init(rank, seed)
            
            # Gradient polish
            U, V, W = gradient_descent(U, V, W, steps=150, lr=0.02)
            
            # Check half-integer rounding
            result = check_half_integer(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If we found something better, return it
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U_list, V_list, W_list = get_schoolbook()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
