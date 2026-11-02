"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
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
    
    # Try alternating minimization for various ranks
    def try_rank_alt(r, n_trials=10, max_iter=300):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization with structured patterns
            if trial % 3 == 0:
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 3 == 1:
                # Sparse initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    rows_u = np.random.choice(nm, size=min(3, nm), replace=False)
                    rows_v = np.random.choice(mp, size=min(3, mp), replace=False)
                    rows_w = np.random.choice(pn, size=min(3, pn), replace=False)
                    U[rows_u, d] = np.random.choice([-1, 1], size=len(rows_u))
                    V[rows_v, d] = np.random.choice([-1, 1], size=len(rows_v))
                    W[rows_w, d] = np.random.choice([-1, 1], size=len(rows_w))
            else:
                # Block-structured initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                block_size = max(1, r // 4)
                for d in range(r):
                    block = d // block_size
                    # Assign to specific blocks based on structure
                    for i in range(n):
                        for j in range(m):
                            if (i * m + j) % (block_size * 2) == block % (block_size * 2):
                                U[i * m + j, d] = 1
                    for j in range(m):
                        for k in range(p):
                            if (j * p + k) % (block_size * 2) == block % (block_size * 2):
                                V[j * p + k, d] = 1
                    for k in range(p):
                        for i in range(n):
                            if (k * n + i) % (block_size * 2) == block % (block_size * 2):
                                W[k * n + i, d] = 1
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Optimize W: Fix U and V
                X = np.zeros((nm * mp, r))
                for d in range(r):
                    X[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(X, T_vec_w.T, rcond=None)[0].T
                
                # Optimize U: Fix V and W
                Y = np.zeros((nm * pn, r))
                for d in range(r):
                    Y[:, d] = (V[:, d][:, None] * W[:, d][None, :]).flatten()
                
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(Y, T_vec_u.T, rcond=None)[0].T
                
                # Optimize V: Fix U and W
                Z = np.zeros((mp * pn, r))
                for d in range(r):
                    Z[:, d] = (U[:, d][:, None] * W[:, d][None, :]).flatten()
                
                ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
                ZTT = Z.T @ T_vec_v.T
                try:
                    V = np.linalg.solve(ZTZ, ZTT).T
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(Z, T_vec_v.T, rcond=None)[0].T
            
            # Check convergence
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
                    best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(nm, mp, pn) but more realistically n*p + m*p + n*m or similar
    # For 2x4x7: nm=8, mp=28, pn=14
    # Try from 30 to schoolbook
    for r in range(30, best_rank):
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=5)
        if success:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
