import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill (evaluator)
    
    n, m, p = 3, 4, 6
    
    # Fast vectorized verification
    def verify_fast(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :]
                    s = np.dot(prod, W[k*n+i, :])
                    expected = 1.0 if j == k else 0.0
                    err = abs(s - expected)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Build the rank-66 decomposition
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Convert to numpy
    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)
    
    err66 = verify_fast(U, V, W)
    if err66 > 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the tensor matrix M for rank analysis
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_idx = len(indices)
    M = np.zeros((n_idx, 66))
    for idx, (i, j, k) in enumerate(indices):
        M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Perform SVD on M to determine rank and find dependent columns
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        rank_M = np.sum(s_vals > tol)
        
        if rank_M < 66:
            if rank_M == 65:
                v_null = Vt_svd[-1]
                c_drop = np.argmax(np.abs(v_null))
                keep = [l for l in range(66) if l != c_drop]
                
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                
                err = verify_fast(U_new, V_new, W_new)
                if err < 1e-8:
                    rank = 65
                    best_U, best_V, best_W = U_new, V_new, W_new
            elif rank_M < 65:
                # Try to drop multiple columns
                # Get null space basis
                null_vecs = Vt_svd[rank_M:]
                # Try dropping columns with largest null-space coefficients
                null_norms = np.sum(null_vecs ** 2, axis=0)
                # Sort by null norm (largest first)
                drop_order = np.argsort(null_norms)[::-1]
                
                # Try dropping 2 columns
                if time.time() < deadline - 10:
                    for d1 in drop_order[:5]:
                        for d2 in drop_order[:5]:
                            if d1 == d2:
                                continue
                            keep = [l for l in range(66) if l != d1 and l != d2]
                            U_new = U[:, keep]
                            V_new = V[:, keep]
                            W_new = W[:, keep]
                            err = verify_fast(U_new, V_new, W_new)
                            if err < 1e-8:
                                rank = 64
                                best_U, best_V, best_W = U_new, V_new, W_new
                                break
                        if rank == 64:
                            break
                    # Try dropping 3 columns
                    if rank == 64 and time.time() < deadline - 5:
                        for d1 in drop_order[:4]:
                            for d2 in drop_order[:4]:
                                if d1 == d2:
                                    continue
                                for d3 in drop_order[:4]:
                                    if d3 == d1 or d3 == d2:
                                        continue
                                    keep = [l for l in range(66) if l not in (d1, d2, d3)]
                                    U_new = U[:, keep]
                                    V_new = V[:, keep]
                                    W_new = W[:, keep]
                                    err = verify_fast(U_new, V_new, W_new)
                                    if err < 1e-8:
                                        rank = 63
                                        best_U, best_V, best_W = U_new, V_new, W_new
                                        break
                                if rank == 63:
                                    break
                            if rank == 63:
                                break
    except Exception as e:
        pass
    
    # Additional local search: try to find better decompositions by perturbation
    # If we have time, try small random perturbations to see if we can find lower rank
    if rank > 63 and time.time() < deadline - 15:
        # Try a few random column combinations
        np.random.seed(42)
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            # Try dropping 2 random columns
            drop_cols = np.random.choice(66, size=2, replace=False)
            keep = [l for l in range(66) if l not in drop_cols]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            err = verify_fast(U_new, V_new, W_new)
            if err < 1e-8:
                rank = 64
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
