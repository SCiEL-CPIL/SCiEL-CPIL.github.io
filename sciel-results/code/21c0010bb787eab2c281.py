"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p

    # Build the target tensor: T[a, b, c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i)
    # Otherwise 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

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

    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def try_rank_als(r, max_iters=300, n_restarts=8, seed=42):
        """Try to find rank-r decomposition using ALS with structured initialization."""
        rng = np.random.default_rng(seed)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Structured initialization: use schoolbook-like patterns with random mixing
            if restart == 0:
                # Start from a perturbed schoolbook
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
                # Add small random perturbation
                U += rng.standard_normal((nm, r)) * 0.1
                V += rng.standard_normal((mp, r)) * 0.1
                W += rng.standard_normal((pn, r)) * 0.1
            else:
                # Random initialization
                U = rng.standard_normal((nm, r)) * 0.5
                V = rng.standard_normal((mp, r)) * 0.5
                W = rng.standard_normal((pn, r)) * 0.5
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each a, solve (V^T ⊙ W^T) U[a,:] = T[a,:,:].flatten()
                # M_U[(b, c_idx), d] = V[b, d] * W[c_idx, d]
                M_U = (V[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(mp * pn, r)
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                
                # Update V
                M_V = (U[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(nm * pn, r)
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                
                # Update W
                M_W = (U[:, :, None] * V[None, :, :]).transpose(1, 2, 0).reshape(nm * mp, r)
                for c_idx in range(pn):
                    rhs = T[:, :, c_idx].flatten()
                    W[c_idx, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                
                # Check convergence
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

    # Try ranks from schoolbook_rank-1 downward, but limit the search
    # Focus on a few promising ranks
    ranks_to_try = list(range(schoolbook_rank - 1, max(schoolbook_rank - 15, 1), -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = try_rank_als(r, max_iters=200, n_restarts=5, seed=42 + r)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    # Convert to lists
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
