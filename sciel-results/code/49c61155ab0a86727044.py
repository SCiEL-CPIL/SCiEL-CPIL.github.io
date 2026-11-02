import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor: T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check invalid entries are zero
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and i == i2):
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > tol:
                            return False
        return True

    def try_rank(rank, max_restarts=1000, max_iters=300):
        """Try to find a rank decomposition using alternating least squares."""
        best_residual = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            scale = 1.0 / np.sqrt(rank)
            U *= scale
            V *= scale
            W *= scale
            
            for iteration in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, rank)
                target_flat = target.reshape(nm, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, rank)
                target_flat_v = target.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW, target_flat_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, rank)
                target_flat_w = target.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV, target_flat_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Compute residual efficiently
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.sum((T_calc - target) ** 2)
                
                if residual < 1e-10:
                    if verify_decomposition(U, V, W):
                        return U, V, W
                    break
                
                if residual < best_residual:
                    best_residual = residual
            
            if best_residual < 1e-10:
                if verify_decomposition(U, V, W):
                    return U, V, W
        
        return None, None, None

    # Try ranks from 20 upward
    for rank in range(20, 22):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_restarts=500 if rank == 20 else 1000, max_iters=500)
        if result[0] is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: Strassen-based rank 21
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 2 * b
                elif l == 1:
                    v_idx = 2 * b + 1
                elif l == 2:
                    v_idx = 6 + 2 * b
                elif l == 3:
                    v_idx = 7 + 2 * b
                V[v_idx][r] = val
                
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
