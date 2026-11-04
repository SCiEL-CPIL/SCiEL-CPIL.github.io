"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p

    # Build the target tensor as a 3D array T[a,b,c]
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook solution as fallback
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

    rng = np.random.default_rng(42)

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose T."""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        return err < tol, err

    def try_rank_als(r, max_iters=300, n_restarts=8):
        """Try to find rank-r decomposition using ALS with better convergence."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Structured initialization: use schoolbook pattern with random perturbations
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with a mix of structured and random
            if restart < 2:
                # Structured init based on block structure
                for idx in range(r):
                    i = idx % n
                    j = (idx // n) % m
                    k = (idx // (n*m)) % p
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                # Add small random perturbation
                U += 0.1 * rng.standard_normal((nm, r))
                V += 0.1 * rng.standard_normal((mp, r))
                W += 0.1 * rng.standard_normal((pn, r))
            else:
                U = rng.standard_normal((nm, r))
                V = rng.standard_normal((mp, r))
                W = rng.standard_normal((pn, r))
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                M_U = (V[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(mp * pn, r)
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                    except:
                        U[a, :] = rng.standard_normal(r)
                
                # Update V
                M_V = (U[:, :, None] * W[None, :, :]).transpose(1, 2, 0).reshape(nm * pn, r)
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                    except:
                        V[b, :] = rng.standard_normal(r)
                
                # Update W
                M_W = (U[:, :, None] * V[None, :, :]).transpose(1, 2, 0).reshape(nm * mp, r)
                for c_idx in range(pn):
                    rhs = T[:, :, c_idx].flatten()
                    try:
                        W[c_idx, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                    except:
                        W[c_idx, :] = rng.standard_normal(r)
                
                # Check convergence
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)
                
                if err < 1e-10:
                    return U, V, W
                
                # Early stopping if stuck
                if abs(prev_err - err) < 1e-12 and it > 50:
                    break
                prev_err = err
            
            # Final check
            valid, err = check_decomposition(U, V, W)
            if valid:
                return U, V, W
        
        return None

    # Try ranks from schoolbook_rank-1 downward, wider range
    # Focus on promising ranks: multiples of 3, and values near known bounds
    ranks_to_try = []
    for r in range(schoolbook_rank - 1, max(1, schoolbook_rank - 30), -1):
        ranks_to_try.append(r)
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = try_rank_als(r, max_iters=150, n_restarts=4)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break  # Found the lowest rank, stop

    # Convert to lists for output
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
