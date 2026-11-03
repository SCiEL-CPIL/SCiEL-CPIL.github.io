"""Tensor decomposition for 3x3x7 matrix multiplication using rank-revealing QR."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    deadline = time.time() + 120.0
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Compute mode-flattening ranks for lower bounds
    T1 = T.reshape(nm, mp * pn)
    r1 = np.linalg.matrix_rank(T1)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    r2 = np.linalg.matrix_rank(T2)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    r3 = np.linalg.matrix_rank(T3)
    lower_bound = max(r1, r2, r3)
    
    trivial_rank = n * m * p  # 63
    
    def verify_decomposition(U, V, W, T, tol=1e-9):
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_comp - T.astype(np.float64))
        return np.max(diff) < tol
    
    # Try a quick search for a better decomposition
    # If found, use it; otherwise fall back to trivial
    best_result = None
    
    # Try rank = lower_bound first (most likely to succeed if a better solution exists)
    if lower_bound < trivial_rank and time.time() < deadline - 10:
        # Try a simple structured search
        rng = np.random.RandomState(42)
        for R in range(lower_bound, min(lower_bound + 5, trivial_rank)):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                U = np.zeros((nm, R), dtype=np.float64)
                V = np.zeros((mp, R), dtype=np.float64)
                W = np.zeros((pn, R), dtype=np.float64)
                
                for r in range(R):
                    i = rng.randint(0, n)
                    j = rng.randint(0, m)
                    k = rng.randint(0, p)
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, r] = 1.0
                    V[b, r] = 1.0
                    W[c, r] = 1.0
                
                # Local search
                for _ in range(100):
                    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                    residual = T_comp - T.astype(np.float64)
                    if np.max(np.abs(residual)) < 1e-10:
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if verify_decomposition(U_r, V_r, W_r, T):
                            best_result = (R, U_r, V_r, W_r)
                            break
                        break
                    
                    # Random perturbation
                    mat = [U, V, W][rng.randint(0, 3)]
                    row = rng.randint(0, mat.shape[0])
                    col = rng.randint(0, mat.shape[1])
                    old = mat[row, col]
                    mat[row, col] = old + rng.choice([-1, 1]) * rng.uniform(0.1, 1)
                    
                    T_comp_new = np.einsum('ar,br,cr->abc', U, V, W)
                    residual_new = T_comp_new - T.astype(np.float64)
                    if np.max(np.abs(residual_new)) >= np.max(np.abs(residual)):
                        mat[row, col] = old
                
                if best_result is not None:
                    break
            
            if best_result is not None:
                break
    
    if best_result is not None:
        R, U, V, W = best_result
    else:
        R = trivial_rank
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
