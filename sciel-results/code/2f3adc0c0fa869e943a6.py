"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Precompute T in different orientations for efficiency
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # For optimizing W
    T_vec_u = T.reshape(nm, mp * pn)  # For optimizing U
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # For optimizing V
    
    def try_rank_alt(r, n_trials=20, max_iter=600):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Structured initialization exploiting tensor structure
            if trial % 5 == 0:
                # Standard random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 5 == 1:
                # Sparse-like with ±1 entries
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(1, nm//3), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp//3), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn//3), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            elif trial % 5 == 2:
                # Block-structured: exploit i and k structure
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    i_d = d % n
                    k_d = (d // n) % p
                    j_start = (d // (n * p)) % m
                    # U: rows i*m+j for i=i_d, j in some range
                    for j in range(j_start, min(m, j_start + max(1, m//2))):
                        U[i_d * m + j, d] = 1.0
                    # V: rows j*p+k for j in some range, k=k_d
                    for j in range(j_start, min(m, j_start + max(1, m//2))):
                        V[j * p + k_d, d] = 1.0
                    # W: rows k*n+i for k=k_d, i=i_d
                    W[k_d * n + i_d, d] = 1.0
            elif trial % 5 == 3:
                # Hybrid: mix of random and structured
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(pn, r) * 0.3
                for d in range(r):
                    U[d % nm, d] += 1.0
                    V[d % mp, d] += 1.0
                    W[d % pn, d] += 1.0
            else:
                # Structured with full i,j,k coverage per rank element
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    i_d = d % n
                    j_d = (d // n) % m
                    k_d = (d // (n * m)) % p
                    U[i_d * m + j_d, d] = 1.0
                    V[j_d * p + k_d, d] = 1.0
                    W[k_d * n + i_d, d] = 1.0
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Vectorized construction of X for optimizing W
                # X[a*mp+b, d] = U[a,d]*V[b,d]
                X = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, r)
                
                # Solve for W
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W_new = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W_new = W
                
                # Vectorized construction of Y for optimizing U
                # Y[b*pn+c, d] = V[b,d]*W[c,d]
                Y = np.einsum('bd,cd->bcd', V, W_new).reshape(mp * pn, r)
                
                # Solve for U
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U_new = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U_new = U
                
                # Vectorized construction of Z for optimizing V
                # Z[a*pn+c, d] = U[a,d]*W[c,d]
                Z = np.einsum('ad,cd->acd', U_new, W_new).reshape(nm * pn, r)
                
                # Solve for V
                ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
                ZTT = Z.T @ T_vec_v.T
                try:
                    V_new = np.linalg.solve(ZTZ, ZTT).T
                except np.linalg.LinAlgError:
                    V_new = V
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_rec = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_rec - T)
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-12 and it > 20:
                    break
                prev_err = err
            
            # Final check with polish
            T_rec = tensor_from_factors(U, V, W)
            if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                    best_U, best_V, best_W = U_r, V_r, W_r
                else:
                    # Try rounding to integers
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    T_rec_i = tensor_from_factors(U_i, V_i, W_i)
                    if np.allclose(T_rec_i, T, atol=1e-8, rtol=1e-8):
                        best_U, best_V, best_W = U_i, V_i, W_i
                    else:
                        best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound for matrix multiplication is max(n, m, p) = 7
    # But practical lower bound is max(n*m, m*p, p*n) / something
    # Let's try a range of ranks, focusing on smaller ones first
    lower_bound = max(n, m, p)  # = 7, theoretical minimum
    # But we know schoolbook is 56, so let's try from a reasonable range
    # Try ranks: 7,8,...,28, then 29,...,56
    # But be smart: try smaller ranks with more trials
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        # More trials for smaller ranks (harder to find)
        n_trials = 25 if r < 20 else 20 if r < 30 else 15
        success = try_rank_alt(r, n_trials=n_trials)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
