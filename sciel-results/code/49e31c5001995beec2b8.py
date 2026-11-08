"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    nm, mp, np_ = n * m, m * p, p * n  # 9, 18, 18
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    # --- 1. Build the known valid Rank-46 Decomposition (Laderman x 2) ---
    
    # Laderman's 23 terms for 3x3 x 3x3
    # A indices: 0..8, B indices: 0..8, C indices: 0..8
    laderman_terms = [
        ([1,0,0,0,0,0,0,0,0], [1,0,0,-1,0,0,1,1,0], [1,0,0,0,0,0,0,0,1]),
        ([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([1,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1], [1,1,0,-1,-1,0,0,0,0], [1,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,1,0,1], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,1,0,0,0,0,0,0,1], [0,0,0,1,1,0,-1,-1,0], [0,1,0,0,0,0,0,1,0]),
        ([1,0,1,0,-1,0,1,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,-1]),
        ([0,0,1,0,-1,0,0,0,1], [1,0,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0,0]),
        ([1,0,0,0,0,0,1,0,0], [0,0,1,0,-1,0,0,-1,0], [0,0,0,1,0,0,0,0,0]),
        ([1,1,0,0,0,0,1,0,1], [0,0,1,0,0,-1,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,1,1,0,1,0,0,0,0], [0,0,0,0,1,-1,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,1,1,0,0,0,1,0,1], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]),
        ([1,0,1,0,1,0,0,0,0], [0,1,-1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,1,0,1,1,0,0,0,0], [0,0,0,0,1,-1,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,0,0,1,1,0,1,0,0], [0,0,0,0,0,1,-1,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,1,1,0,0,1,0], [0,0,0,0,0,0,0,1,-1], [0,0,0,0,0,1,0,0,0]),
        ([0,0,0,0,0,1,0,1,0], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,1,0,1,0], [0,0,0,0,1,-1,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,0,1,-1,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,1,0], [0,0,0,1,0,0,-1,0,0], [0,0,0,0,1,0,0,0,0]),
        ([1,0,0,1,0,0,1,0,0], [0,0,0,0,0,1,-1,0,0], [1,0,0,0,0,0,0,0,0]),
        ([1,0,0,1,0,0,1,0,0], [0,0,0,0,0,0,0,1,-1], [1,0,0,0,0,0,0,0,0])
    ]
    
    rank_46 = 46
    U46 = [[0] * rank_46 for _ in range(nm)]
    V46 = [[0] * rank_46 for _ in range(mp)]
    W46 = [[0] * rank_46 for _ in range(np_)]
    
    for t in range(23):
        alpha, beta, gamma = laderman_terms[t]
        # Block 1 (B1, C1)
        col1 = t
        for a_idx in range(9):
            U46[a_idx][col1] = alpha[a_idx]
        for j in range(3):
            for k in range(3):
                V46[j*6+k][col1] = beta[j*3+k]
        for i in range(3):
            for k in range(3):
                W46[k*3+i][col1] = gamma[i*3+k]
                
        # Block 2 (B2, C2)
        col2 = 23 + t
        for a_idx in range(9):
            U46[a_idx][col2] = alpha[a_idx]
        for j in range(3):
            for k in range(3):
                V46[j*6+3+k][col2] = beta[j*3+k]
        for i in range(3):
            for k in range(3):
                W46[(3+k)*3+i][col2] = gamma[i*3+k]
                
    def verify_decomp(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Target tensor
        T = np.zeros((nm, mp, np_))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    x = i * m + j
                    y = j * p + k
                    z = k * n + i
                    T[x, y, z] = 1.0
        
        decomp = np.zeros((nm, mp, np_))
        for c in range(rank):
            decomp += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
            
        return np.allclose(decomp, T, atol=1e-9)

    # Verify rank 46
    if verify_decomp(U46, V46, W46, rank_46):
        # Check half-integer
        U_c = np.round(np.array(U46) * 2) / 2
        V_c = np.round(np.array(V46) * 2) / 2
        W_c = np.round(np.array(W46) * 2) / 2
        if np.allclose(U_c, U46) and np.allclose(V_c, V46) and np.allclose(W_c, W46):
            best_result = {"U": U46, "V": V46, "W": W46}
            best_rank = rank_46
        else:
            best_result = None
            best_rank = float('inf')
    else:
        best_result = None
        best_rank = float('inf')

    # --- 2. Attempt to find a lower rank decomposition (e.g., rank 38) ---
    
    def unpack(x, r):
        U = x[:r*nm].reshape(r, nm).T
        V = x[r*nm:r*nm + r*mp].reshape(r, mp).T
        W = x[r*nm + r*mp:r*nm + r*mp + r*np_].reshape(r, np_).T
        return U, V, W

    def residual(x, r):
        U, V, W = unpack(x, r)
        res = np.zeros((nm, mp, np_))
        for c in range(r):
            res += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        # Target T is built once outside or passed? 
        # For efficiency, we should pass T_flat or build it once.
        # But least_squares expects residual(x). 
        # We can use a closure or global. 
        # Let's build T_flat once.
        return (res - T_target).flatten()

    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T_target[x, y, z] = 1.0
    T_flat = T_target.flatten()

    def residual_func(x, r):
        U, V, W = unpack(x, r)
        res = np.zeros((nm, mp, np_))
        for c in range(r):
            res += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return (res - T_target).flatten()

    ranks_to_try = [38, 37, 36]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            try:
                x0 = np.random.randn(r * (nm + mp + np_)) * 0.1
                # Use LM method
                res_opt = least_squares(lambda z: residual_func(z, r), x0, method='lm', max_nfev=500)
                err = np.linalg.norm(res_opt.fun)
                
                if err < 1e-6:
                    U, V, W = unpack(res_opt.x, r)
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness of rounded version
                    if verify_decomp(U_r.tolist(), V_r.tolist(), W_r.tolist(), r):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = r
                        break # Found a better one, break out of attempts
            except Exception:
                pass
        if best_rank < 46:
            break # Found rank < 46, no need to try lower ranks if we found 38? 
            # Actually, we should check if we found 38, 37, or 36.
            # If we found 38, we can stop if we don't want to try 37/36.
            # But 36 is better. So we should continue if we found 38?
            # The loop iterates 38, 37, 36.
            # If we find 38, we set best_rank=38.
            # Then we try 37. If we find 37, best_rank=37.
            # So we should NOT break the outer loop unless we find the absolute lowest we care about or time runs out.
            # But finding 36 is unlikely. 38 is the known bound.
            # I will break if I find 38 to save time, assuming 37/36 are very hard.
            if best_rank == 38:
                break

    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook if even rank 46 failed (should not happen)
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

# EVOLVE-BLOCK-END
