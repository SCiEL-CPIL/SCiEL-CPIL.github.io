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
    
    def _verify_exact(U, V, W, seed=42, trials=3):
        """Verify exact decomposition using multiple random trials."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        rng = np.random.RandomState(seed)
        for trial in range(trials):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_r = Ua[:, r]
                v_r = Va[:, r]
                w_r = Wa[:, r]
                u_scalar = np.dot(u_r, A.flatten())
                v_scalar = np.dot(v_r, B.flatten())
                C_approx += u_scalar * v_scalar * w_r.reshape(p, n).T
            if not np.allclose(C_approx, C_true, atol=1e-9):
                return False
        return True
    
    def _try_rank(R, rng, max_iters=5000):
        """Try to find a rank-R decomposition with small integer/half-integer entries."""
        # Search space: entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
        vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
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
                
                U[idx_u, r] = rng.choice(vals, nz_u)
                V[idx_v, r] = rng.choice(vals, nz_v)
                W[idx_w, r] = rng.choice(vals, nz_w)
            
            # Quick check: try a few random A, B
            Ua = U
            Va = V
            Wa = W
            
            ok = True
            for _ in range(3):
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_true = A @ B
                C_approx = np.zeros((n, p))
                for r in range(R):
                    u_scalar = np.dot(Ua[:, r], A.flatten())
                    v_scalar = np.dot(Va[:, r], B.flatten())
                    C_approx += u_scalar * v_scalar * Wa[:, r].reshape(p, n).T
                if not np.allclose(C_approx, C_true, atol=1e-6):
                    ok = False
                    break
            
            if ok:
                # Full verification
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
