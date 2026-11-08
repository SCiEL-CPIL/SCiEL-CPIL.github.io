import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    dim_u = n * m
    dim_v = m * p
    dim_w = n * p

    # Build target tensor T with correct index conventions
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i  (Task convention)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i  # Corrected
                T[idx_u, idx_v, idx_w] = 1.0

    def try_rank(r):
        if time.time() - start_time > 110:
            return None
        np.random.seed(42 + r)
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        def compute_error(U, V, W):
            T_recon = np.zeros_like(T)
            for r_idx in range(r):
                T_recon += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            return np.linalg.norm(T_recon - T)

        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(5):
            if time.time() - start_time > 110:
                break
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            for step in range(200):
                if time.time() - start_time > 110:
                    break
                idx = np.random.randint(3)
                if idx == 0:
                    U_curr[np.random.randint(dim_u), np.random.randint(r)] += np.random.randn() * 0.1
                elif idx == 1:
                    V_curr[np.random.randint(dim_v), np.random.randint(r)] += np.random.randn() * 0.1
                else:
                    W_curr[np.random.randint(dim_w), np.random.randint(r)] += np.random.randn() * 0.1
                
                err = compute_error(U_curr, V_curr, W_curr)
                if err < best_err:
                    best_err = err
                    best_U = U_curr.copy()
                    best_V = V_curr.copy()
                    best_W = W_curr.copy()
                    
                    # Check exactness with half-integer rounding
                    U_rounded = np.round(best_U * 2) / 2.0
                    V_rounded = np.round(best_V * 2) / 2.0
                    W_rounded = np.round(best_W * 2) / 2.0
                    
                    T_recon_rounded = np.zeros_like(T)
                    for r_idx in range(r):
                        T_recon_rounded += np.einsum('i,j,k->ijk', U_rounded[:, r_idx], V_rounded[:, r_idx], W_rounded[:, r_idx])
                    
                    if np.allclose(T_recon_rounded, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
        return None

    # Try rank 27
    res_27 = try_rank(27)
    if res_27 is not None:
        return res_27

    # Fallback to Rank 28 (Strassen)
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(2, 2, 0, 0, 7)
    apply_strassen(0, 0, 2, 2, 14)
    apply_strassen(2, 2, 2, 2, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
