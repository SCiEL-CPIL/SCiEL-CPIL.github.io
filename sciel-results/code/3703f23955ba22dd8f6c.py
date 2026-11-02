"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Return the best exact tensor decomposition found within a time budget."""
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def _verify_fast(Ua, Va, Wa, rng, n_trials=5, tol=1e-9):
        """Vectorized verification: much faster than per-rank Python loop."""
        R = Ua.shape[1]
        # Reshape W to (p, n, R) so that W[k*n+i, r] -> W3[i, k, r]
        W3 = Wa.reshape(p, n, R).transpose(1, 0, 2)  # (n, p, R)
        
        for _ in range(n_trials):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B  # (n, p)
            
            # u_scalars: (R,) = Ua.T @ A.flatten()
            u_scalars = Ua.T @ A.flatten()
            v_scalars = Va.T @ B.flatten()
            # C_approx[i,k] = sum_r u_scalars[r] * v_scalars[r] * W3[i,k,r]
            C_approx = np.einsum('r,r,ikr->ik', u_scalars, v_scalars, W3)
            
            if not np.allclose(C_approx, C_true, atol=tol, rtol=0):
                return False
        return True
    
    def _verify_exact(U, V, W, seed=42, trials=3):
        """Verify exact decomposition using multiple random trials."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(seed)
        return _verify_fast(Ua, Va, Wa, rng, n_trials=trials, tol=1e-9)
    
    def _try_rank(R, rng, max_iters=2000):
        """Try to find a rank-R decomposition with small integer/half-integer entries."""
        # Search space: entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
        vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        val_arr = np.array(vals)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Random sparse initialization with small values
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            for r in range(R):
                # Each column gets a few non-zero entries
                nz_u = rng.randint(1, 4)
                nz_v = rng.randint(1, 4)
                nz_w = rng.randint(1, 4)
                
                idx_u = rng.choice(n * m, nz_u, replace=False)
                idx_v = rng.choice(m * p, nz_v, replace=False)
                idx_w = rng.choice(p * n, nz_w, replace=False)
                
                U[idx_u, r] = val_arr[rng.randint(0, len(vals), nz_u)]
                V[idx_v, r] = val_arr[rng.randint(0, len(vals), nz_v)]
                W[idx_w, r] = val_arr[rng.randint(0, len(vals), nz_w)]
            
            # Quick check with vectorized verification
            if _verify_fast(U, V, W, rng, n_trials=3, tol=1e-6):
                # Full verification with tighter tolerance
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify_exact(U_list, V_list, W_list, seed=rng.randint(10000), trials=5):
                    return U_list, V_list, W_list
        
        return None
    
    # Start with schoolbook as baseline
    best_result = _schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x4x7, rank is at least max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    # Try ranks from a reasonable lower bound up to schoolbook
    for R in range(28, min(best_rank, 50)):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(R * 1000 + 42)
        result = _try_rank(R, rng, max_iters=2000)
        
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = R
            break  # Found a valid lower rank, stop
    
    # If we found something better, return it; otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
