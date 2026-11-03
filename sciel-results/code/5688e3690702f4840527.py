import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb

    def check_decomp(U, V, W):
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=1e-9)

    # Try Rank 32 with Block Structure
    # Laderman part: 23 terms, Schoolbook part: 9 terms
    rank_32 = 32
    
    # Initialize U, V, W
    np.random.seed(42)
    U = np.zeros((n*m, rank_32), dtype=float)
    V = np.zeros((m*p, rank_32), dtype=float)
    W = np.zeros((p*n, rank_32), dtype=float)
    
    # Schoolbook part (cols 23-31)
    # Corresponds to B col 3, C col 3
    for idx in range(9):
        i = idx // 3
        j = idx % 3
        col = 23 + idx
        U[i*m+j, col] = 1.0
        V[j*p+3, col] = 1.0
        W[3*n+i, col] = 1.0
        
    # Laderman part (cols 0-22)
    # Initialize with random values in the non-zero blocks
    # V non-zero rows: 0,1,2,4,5,6,8,9,10
    # W non-zero rows: 0..8
    for col in range(23):
        U[:, col] = np.random.randn(n*m) * 0.1
        V[0, col] = np.random.randn() * 0.1
        V[1, col] = np.random.randn() * 0.1
        V[2, col] = np.random.randn() * 0.1
        V[4, col] = np.random.randn() * 0.1
        V[5, col] = np.random.randn() * 0.1
        V[6, col] = np.random.randn() * 0.1
        V[8, col] = np.random.randn() * 0.1
        V[9, col] = np.random.randn() * 0.1
        V[10, col] = np.random.randn() * 0.1
        W[0:9, col] = np.random.randn(9) * 0.1

    # ALS on the Laderman part only?
    # The tensor T is the sum of the Laderman part and Schoolbook part.
    # T_Laderman = T - T_Schoolbook
    T_school = np.einsum('at,bt,ct->abc', U[:, 23:], V[:, 23:], W[:, 23:], optimize=True)
    T_lad = T - T_school
    
    # Now find rank 23 decomposition of T_lad
    # T_lad is non-zero only in the blocks corresponding to B cols 0-2 and C cols 0-2.
    
    def try_laderman_als(max_iters=50):
        U_lad = U[:, :23].copy()
        V_lad = V[:, :23].copy()
        W_lad = W[:, :23].copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for a in range(n*m):
                # T_lad[a, :, :]
                # V_lad and W_lad
                M = np.einsum('bt,ct->bct', V_lad, W_lad).reshape(m*p * p*n, 23)
                # But we only care about the non-zero parts of V and W
                # V_lad has zeros in rows 3,7,11
                # W_lad has zeros in rows 9,10,11
                # So M will have zeros in those combinations.
                b_vec = T_lad[a, :, :].reshape(-1)
                try:
                    U_lad[a, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(m*p):
                if b in [3, 7, 11]: continue # Zero in Laderman part
                M = np.einsum('at,ct->act', U_lad, W_lad).reshape(n*m * p*n, 23)
                b_vec = T_lad[:, b, :].reshape(-1)
                try:
                    V_lad[b, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(p*n):
                if c in [9, 10, 11]: continue # Zero in Laderman part
                M = np.einsum('at,bt->abt', U_lad, V_lad).reshape(n*m * m*p, 23)
                b_vec = T_lad[:, :, c].reshape(-1)
                try:
                    W_lad[c, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            U[:, :23] = U_lad
            V[:, :23] = V_lad
            W[:, :23] = W_lad
            
            if check_decomp(U, V, W):
                return True
                
        return False

    if try_laderman_als():
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_decomp(U_s, V_s, W_s):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
