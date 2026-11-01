"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find rank-20 decomposition using ALS with multiple seeds
    def try_als(rank, max_iters=300, seeds=[42, 123, 777, 2024]):
        for seed in seeds:
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            # Normalize initial vectors
            for r in range(rank):
                nU = np.linalg.norm(U[:,r])
                nV = np.linalg.norm(V[:,r])
                nW = np.linalg.norm(W[:,r])
                if nU > 0: U[:,r] /= nU
                if nV > 0: V[:,r] /= nV
                if nW > 0: W[:,r] /= nW
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                TW = T @ W  # (n*m, m*p, p*n) @ (p*n, rank) -> (n*m, m*p, rank)
                inner_U = np.einsum('abr,br->ar', TW, V)
                denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
                U = inner_U / (denom_U + 1e-15)
                
                # Update V
                TU = np.einsum('abc,ar->bcr', T, U)
                inner_V = np.einsum('bcr,cr->br', TU, W)
                denom_V = np.sum(U**2, axis=0) * np.sum(W**2, axis=0)
                V = inner_V / (denom_V + 1e-15)
                
                # Update W
                inner_W = np.einsum('bcr,br->cr', TU, V)
                denom_W = np.sum(U**2, axis=0) * np.sum(V**2, axis=0)
                W = inner_W / (denom_W + 1e-15)
                
                if it % 20 == 0 or it == max_iters - 1:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(recon - T))
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-10:
                        break
            
            if best_err < 1e-8:
                return best_U, best_V, best_W, best_err
        return None
    
    # Try rank 20
    result20 = try_als(20, max_iters=200)
    if result20 is not None:
        U, V, W, err = result20
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(recon - T)) < 1e-8:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 21 Strassen
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: C00 + C11
        [(1, 0, 1), (1, 1, -1)],     # M2: C10 - C11
        [(0, 1, 1), (1, 1, 1)],      # M3: C01 + C11
        [(0, 0, 1), (1, 0, 1)],      # M4: C00 + C10
        [(0, 0, -1), (0, 1, 1)],     # M5: -C00 + C01
        [(1, 1, 1)],                 # M6: C11
        [(0, 0, 1)]                  # M7: C00
    ]
    
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            for idx, val in U_coeffs[t].items():
                U[idx][r] = val
            for (j, k_local, val) in V_coeffs[t]:
                k_global = 2 * b + k_local
                v_idx = j * p + k_global
                V[v_idx][r] = val
            for (i, k_local, val) in W_coeffs[t]:
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
