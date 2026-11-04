"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 110.0  # safety margin
    
    # Try rank 25 first (theoretical lower bound)
    for rank in [25, 26, 27, 28, 30, 35, 40, 50, 60, 80, 100]:
        if time.time() > deadline - 30:
            break
        if not _try_rank(T, nm, mp, pn, rank, deadline):
            continue
        U, V, W = _get_best_decomp()
        # Convert to list of lists with rounding to half-integers
        U_list = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(nm)]
        V_list = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(mp)]
        W_list = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(pn)]
        
        # Verify exactness
        if _verify(U_list, V_list, W_list, T, nm, mp, pn):
            return {"U": U_list, "V": V_list, "W": W_list}
        # Try without rounding - check if already close to half-integers
        if _verify(U, V, W, T, nm, mp, pn):
            return {"U": [[U[i,c] for c in range(rank)] for i in range(nm)],
                    "V": [[V[i,c] for c in range(rank)] for i in range(mp)],
                    "W": [[W[i,c] for c in range(rank)] for i in range(pn)]}
    
    # Fallback: schoolbook
    rank = nm * mp * pn
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition using alternating minimization."""
    np.random.seed(42)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        # Random restart every 50 iterations
        if it % 50 == 0:
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
        
        for _ in range(5):
            # Update U: for each a, U[a,:] = (sum over b,d of T[a,b,d] * V[b,:]*W[d,:]) / (sum over b,d of V[b,:]^2 * W[d,:]^2)
            # This is a least squares problem per row
            # For efficiency, do it in batches
            # U_new[a, c] = sum_{b,d} T[a,b,d] V[b,c] W[d,c] / sum_{b,d} V[b,c]^2 W[d,c]^2
            
            # Vectorized approach
            # For each c: U[:,c] should minimize sum_{a,b,d} (T[a,b,d] - sum_c' U[a,c']V[b,c']W[d,c'])^2
            # This is a 3-way tensor decomposition - use ALS (Alternating Least Squres)
            
            # Update U
            for a in range(nm):
                # T[a,:,:] is (mp, pn)
                # We want U[a,:] to minimize ||T[a,:,:] - sum_c U[a,c] outer(V[:,c], W[:,c])||_F^2
                # This is a linear system in U[a,:]
                # Let's build the Gram matrix
                # G[c1, c2] = sum_{b,d} V[b,c1]*W[d,c1] * V[b,c2]*W[d,c2]
                # h[c] = sum_{b,d} T[a,b,d] * V[b,c] * W[d,c]
                # Solve G * U[a,:] = h
                # G = (V^T * W) element-wise... actually G[c1,c2] = sum_{b,d} V[b,c1]*V[b,c2]*W[d,c1]*W[d,c2]
                # = (V^T V) ⊙ (W^T W)  (Hadamard product)
                # h[c] = sum_{b,d} T[a,b,d] V[b,c] W[d,c]
                
                VT_V = V.T @ V  # (rank, rank)
                WT_W = W.T @ W  # (rank, rank)
                G = VT_W * WT_W  # Hadamard product
                # h[c] = sum_{b,d} T[a,b,d] V[b,c] W[d,c]
                # = sum_b (sum_d T[a,b,d] W[d,c]) V[b,c]
                # Let's compute: TW = T[a,:,:] @ W  -> (mp, rank), TW[b,c] = sum_d T[a,b,d] W[d,c]
                TW = T[a, :, :] @ W  # (mp, rank)
                h = V.T @ TW  # (rank,)
                
                # Solve G @ U[a,:] = h
                try:
                    U[a, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), h)
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), h, rcond=None)[0]
            
            # Update V similarly
            for b in range(mp):
                VT_V = V.T @ V
                WT_W = W.T @ W
                G = VT_V * WT_W
                # h[c] = sum_{a,d} T[a,b,d] U[a,c] W[d,c]
                # = sum_a (sum_d T[a,b,d] W[d,c]) U[a,c]
                # TU = T[:,b,:] @ U -> (nm, rank), TU[a,c] = sum_d T[a,b,d] U[a,c]... wait
                # Actually: h[c] = sum_{a,d} T[a,b,d] U[a,c] W[d,c]
                # = sum_d W[d,c] * (sum_a T[a,b,d] U[a,c])
                # Let's compute: for each c, sum_{a,d} T[a,b,d] U[a,c] W[d,c]
                # = sum_d W[d,c] * (T[:,b,:] * U)[:,c] summed over a... 
                # T[:,b,:] has shape (nm, pn). U has shape (nm, rank).
                # (T[:,b,:] * U) doesn't directly work. 
                # h[c] = sum_{a,d} T[a,b,d] U[a,c] W[d,c]
                # = sum_c' ... no, c is fixed.
                # = W[:,c]^T @ (T[:,b,:] * U[:,c])  where * is element-wise then sum over a
                # = W[:,c] @ (np.sum(T[:,b,:]*U[:,c,None], axis=0))
                # Let's do it differently:
                # h = np.array([np.sum(T[:,b,:]*U[:,c,None]*W[:,c][None,:]) for c in range(rank)])
                # More efficiently:
                # h[c] = sum_d W[d,c] * sum_a T[a,b,d] U[a,c]
                # Let M[c,d] = sum_a T[a,b,d] U[a,c] -> this is (T[:,b,:] * U) summed over a
                # Actually: for fixed b, T[:,b,:] is (nm, pn). U is (nm, rank).
                # sum_a T[a,b,d] U[a,c] for all c,d: this is (U.T @ T[:,b,:]) -> (rank, pn)
                UT_T = U.T @ T[:, b, :]  # (rank, pn)
                h = UT_T @ W  # (rank,)
                
                try:
                    V[b, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), h)
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), h, rcond=None)[0]
            
            # Update W similarly
            for d in range(pn):
                VT_V = V.T @ V
                WT_W = W.T @ W
                G = VT_V * WT_W
                # h[c] = sum_{a,b} T[a,b,d] U[a,c] V[b,c]
                # = sum_b V[b,c] * sum_a T[a,b,d] U[a,c]
                # T[:, :, d] is (nm, mp). U is (nm, rank), V is (mp, rank).
                # sum_a T[a,b,d] U[a,c] for all b,c: (U.T @ T[:,:,d]) -> (rank, mp)
                UT_T = U.T @ T[:, :, d]  # (rank, mp)
                h = UT_T @ V  # (rank,)
                
                try:
                    W[d, :] = np.linalg.solve(G + 1e-10 * np.eye(rank), h)
                except np.linalg.LinAlgError:
                    W[d, :] = np.linalg.lstsq(G + 1e-10 * np.eye(rank), h, rcond=None)[0]
        
        # Compute error
        # T_hat[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # err = ||T - T_hat||_F^2
        # Efficient: err = sum_{a,b,d} (T[a,b,d] - sum_c U[a,c]V[b,c]W[d,c])^2
        # = sum_{a,b,d} T[a,b,d]^2 - 2 sum_{a,b,d} T[a,b,d] sum_c U[a,c]V[b,c]W[d,c] + sum_{a,b,d} (sum_c U[a,c]V[b,c]W[d,c])^2
        # First term: sum T^2 = number of non-zeros = nm*mp... no, it's 100 (n*m*p)
        # Actually sum T[a,b,d]^2 = n*m*p = 100
        # Second term: 2 * sum_c (sum_{a,b,d} T[a,b,d] U[a,c] V[b,c] W[d,c])
        # = 2 * sum_c ( (T * U[:,c,None,None]) * V[c,None,:] ... ) 
        # Let's just compute directly for small tensors
        T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.sum((T - T_hat) ** 2)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-10:
            _set_best_decomp(U, V, W)
            return True
    
    _set_best_decomp(best_U, best_V, best_W)
    return best_err < 1e-8

def _set_best_decomp(U, V, W):
    global _best_U, _best_V, _best_W
    _best_U = U
    _best_V = V
    _best_W = W

def _get_best_decomp():
    return _best_U, _best_V, _best_W

def _verify(U, V, W, T, nm, mp, pn):
    """Check if the decomposition is exact."""
    if isinstance(U, list):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
    T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.max(np.abs(T - T_hat)) < 1e-9
# EVOLVE-BLOCK-END

