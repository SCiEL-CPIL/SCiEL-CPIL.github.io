import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 30
    
    # Initialize with schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def get_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :R], np.multiply(V[v_idx, :R], W[w_idx, :R]))
                    err += (s - 1.0) ** 2
        return err

    def verify_exact(U, V, W, R, tol=1e-10):
        """Verify exact decomposition with tight tolerance."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :R], np.multiply(V[v_idx, :R], W[w_idx, :R]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_rational(U, V, W, R):
        """Try to snap values to nearby rational numbers with small denominators."""
        U_snap = U.copy()
        V_snap = V.copy()
        W_snap = W.copy()
        
        denominators = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
        
        for arr in [U_snap, V_snap, W_snap]:
            for idx in range(arr.size):
                val = arr.flat[idx]
                if abs(val) < 1e-15:
                    arr.flat[idx] = 0.0
                    continue
                best_rational = val
                best_dist = abs(val - round(val))
                
                for d in denominators:
                    num = round(val * d)
                    rational = num / d
                    dist = abs(val - rational)
                    if dist < best_dist:
                        best_dist = dist
                        best_rational = rational
                
                arr.flat[idx] = best_rational
        
        return U_snap, V_snap, W_snap

    def try_rank_r(R, max_time, seed=42):
        """Try to find a rank-R decomposition using structured search."""
        random.seed(seed)
        np.random.seed(seed)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Strategy 1: Start from schoolbook and remove R columns
        # Keep the most "important" columns (those with largest contributions)
        cols_to_remove = random.sample(range(rank_schoolbook), rank_schoolbook - R)
        cols_to_keep = [c for c in range(rank_schoolbook) if c not in cols_to_remove]
        
        U_init = U_school[:, cols_to_keep].copy()
        V_init = V_school[:, cols_to_keep].copy()
        W_init = W_school[:, cols_to_keep].copy()
        
        # Add small perturbations to break symmetry
        U_init += np.random.randn(n * m, R) * 0.01
        V_init += np.random.randn(m * p, R) * 0.01
        W_init += np.random.randn(p * n, R) * 0.01
        
        # Gradient descent with adaptive learning rate
        U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
        lr = 0.05
        start_time = time.time()
        deadline = start_time + max_time
        
        while time.time() < deadline:
            # Compute gradients
            grad_U = np.zeros_like(U_cur)
            grad_V = np.zeros_like(V_cur)
            grad_W = np.zeros_like(W_cur)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        s = np.dot(U_cur[u_idx, :R], np.multiply(V_cur[v_idx, :R], W_cur[w_idx, :R]))
                        ds = 2 * (s - 1.0)
                        
                        grad_U[u_idx, :R] += ds * np.multiply(V_cur[v_idx, :R], W_cur[w_idx, :R])
                        grad_V[v_idx, :R] += ds * np.multiply(U_cur[u_idx, :R], W_cur[w_idx, :R])
                        grad_W[w_idx, :R] += ds * np.multiply(U_cur[u_idx, :R], V_cur[v_idx, :R])
            
            # Update
            U_cur -= lr * grad_U
            V_cur -= lr * grad_V
            W_cur -= lr * grad_W
            
            err = get_error(U_cur, V_cur, W_cur, R)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                lr *= 0.98
            else:
                U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
                lr *= 0.9
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-10:
            U_snap, V_snap, W_snap = snap_to_rational(best_U, best_V, best_W, R)
            if verify_exact(U_snap, V_snap, W_snap, R):
                return U_snap, V_snap, W_snap, best_err
            if verify_exact(best_U, best_V, best_W, R):
                return best_U, best_V, best_W, best_err
        
        return best_U, best_V, best_W, best_err

    # Try to find rank 28 decomposition
    target_rank = 28
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Try multiple random seeds for diversity
    for seed in range(20):
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        search_time = min(time_left * 0.4, 10.0)
        
        U_sol, V_sol, W_sol, err = try_rank_r(target_rank, search_time, seed=seed)
        
        if err < 1e-10:
            U_snap, V_snap, W_snap = snap_to_rational(U_sol, V_sol, W_sol, target_rank)
            if verify_exact(U_snap, V_snap, W_snap, target_rank):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            if verify_exact(U_sol, V_sol, W_sol, target_rank):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Fallback: return schoolbook decomposition
    U_out = U_school.tolist()
    V_out = V_school.tolist()
    W_out = W_school.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
