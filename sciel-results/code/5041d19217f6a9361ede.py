"""Matrix multiplication 2,3,9 - rank-minimal decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18

    # Schoolbook rank = n*m*p = 54
    schoolbook_rank = n * m * p

    # Build the target tensor: T[a, b, c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-10):
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = schoolbook_rank

    # Strategy 1: Block-diagonal decomposition (rank = n*m = 6)
    # For each row i of A, treat as 1x3 by 3x9 multiplication, which has rank 3.
    # Total rank = n * m = 6
    def block_diagonal():
        r_total = n * m
        U = np.zeros((nm, r_total))
        V = np.zeros((mp, r_total))
        W = np.zeros((pn, r_total))
        
        for i in range(n):
            for j in range(m):
                r = i * m + j  # rank index for this (i,j) pair
                # U: row i*m+j gets 1 at column r
                U[i * m + j, r] = 1.0
                # V: rows j*p+k for all k get 1 at column r
                V[j * p:(j + 1) * p, r] = 1.0
                # W: rows k*n+i for all k get 1 at column r
                for k in range(p):
                    W[k * n + i, r] = 1.0
        
        return U, V, W

    # Try block diagonal first
    U_bd, V_bd, W_bd = block_diagonal()
    if verify(U_bd, V_bd, W_bd):
        U_best, V_best, W_best = U_bd, V_bd, W_bd
        best_rank = n * m

    # Strategy 2: Try to find even lower rank via ALS
    # The tensor rank lower bound for n x m by m x p is at least max(n, m, p) for some cases
    # But for 2x3 by 3x9, let's try ranks below 6
    # Actually, the rank of the 2x3*3x9 tensor is at least the max of the ranks of its slices
    # Each slice for fixed i has rank 3, so total rank >= 6 if they're independent
    # But maybe there's overlap? Let's try anyway.
    
    def try_rank_als(r, max_iters=200, n_restarts=5, seed=42):
        rng = np.random.default_rng(seed)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            if restart == 0:
                # Structured init from perturbed schoolbook
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for idx in range(min(r, schoolbook_rank)):
                    i = idx // (m * p)
                    j = (idx // p) % m
                    k = idx % p
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                U += rng.standard_normal((nm, r)) * 0.1
                V += rng.standard_normal((mp, r)) * 0.1
                W += rng.standard_normal((pn, r)) * 0.1
            else:
                U = rng.standard_normal((nm, r)) * 0.5
                V = rng.standard_normal((mp, r)) * 0.5
                W = rng.standard_normal((pn, r)) * 0.5
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                M_U = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, r)
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                
                M_V = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, r)
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                
                M_W = np.einsum('ar,br->abr', U, V).reshape(nm * mp, r)
                for c_idx in range(pn):
                    rhs = T[:, :, c_idx].flatten()
                    W[c_idx, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(recon - T))
                if err < 1e-10:
                    return U, V, W
                if abs(prev_err - err) < 1e-15:
                    break
                prev_err = err
            
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Try ranks below the current best
    for r in range(best_rank - 1, max(best_rank - 10, 1), -1):
        if time.time() > deadline:
            break
        result = try_rank_als(r, max_iters=100, n_restarts=3, seed=42 + r)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
