"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank 72 as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U_list, V_list, W_list, r):
        """Verify decomposition using exact Fraction arithmetic."""
        # Convert to Fractions
        U_frac = [[Fraction(int(val * 2), 2) if val % 1 != 0 else Fraction(int(val)) for val in row] for row in U_list]
        V_frac = [[Fraction(int(val * 2), 2) if val % 1 != 0 else Fraction(int(val)) for val in row] for row in V_list]
        W_frac = [[Fraction(int(val * 2), 2) if val % 1 != 0 else Fraction(int(val)) for val in row] for row in W_list]
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1 if (j == jp and k == kp) else 0
                            val = sum(U_frac[i*m+j][c] * V_frac[jp*p+k][c] * W_frac[kp*n+i][c] for c in range(r))
                            if val != expected:
                                return False
        return True

    def try_rank_als(r, U_init=None, V_init=None, W_init=None, max_iters=100, seed=None):
        if time.time() > deadline - 10:
            return None
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        if U_init is not None:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
        else:
            # Sparse random initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                # Pick random non-zero indices
                u_idx = rng.randint(0, n * m)
                v_idx = rng.randint(0, m * p)
                w_idx = rng.randint(0, p * n)
                U[u_idx, c] = rng.choice([-1, 1])
                V[v_idx, c] = rng.choice([-1, 1])
                W[w_idx, c] = rng.choice([-1, 1])
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Fix U, V, solve for W
            W_new = np.zeros_like(W)
            for i in range(n):
                for kp in range(p):
                    M = np.zeros((m*m*p, r))
                    t = np.zeros(m*m*p)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for c in range(r):
                                    M[idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        W_new[kp*n+i, :] = w
                    except Exception:
                        pass
            W = W_new
            
            # Fix U, W, solve for V
            V_new = np.zeros_like(V)
            for jp in range(m):
                for k in range(p):
                    M = np.zeros((n*m*p, r))
                    t = np.zeros(n*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for kp in range(p):
                                for c in range(r):
                                    M[idx, c] = U[i*m+j, c] * W[kp*n+i, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        V_new[jp*p+k, :] = v
                    except Exception:
                        pass
            V = V_new
            
            # Fix V, W, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((m*p*p, r))
                    t = np.zeros(m*p*p)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for c in range(r):
                                    M[idx, c] = V[jp*p+k, c] * W[kp*n+i, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        U_new[i*m+j, :] = u
                    except Exception:
                        pass
            U = U_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (j == jp and k == kp) else 0.0
                                val = sum(U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+i, c] for c in range(r))
                                err = abs(val - expected)
                                if err > max_err:
                                    max_err = err
            if max_err < 1e-9:
                break
        
        # Check validity with rounding to half-integers
        U_list = [[round(val * 2) / 2 for val in row] for row in U]
        V_list = [[round(val * 2) / 2 for val in row] for row in V]
        W_list = [[round(val * 2) / 2 for val in row] for row in W]
        
        # Exact verification
        if verify_exact(U_list, V_list, W_list, r):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Get schoolbook solution
    sb = schoolbook()
    U_sb = np.array(sb["U"])
    V_sb = np.array(sb["V"])
    W_sb = np.array(sb["W"])
    
    # Try to reduce rank by zeroing columns
    # We try ranks 71 down to 64
    for r in range(71, 63, -1):
        if time.time() > deadline - 20:
            break
        
        # Strategy 1: Greedy removal from schoolbook
        # Try removing each column of schoolbook and re-optimizing
        found = False
        for col_to_remove in range(72):
            if time.time() > deadline - 15:
                break
            
            # Create masks
            U_mask = U_sb.copy()
            V_mask = V_sb.copy()
            W_mask = W_sb.copy()
            U_mask[:, col_to_remove] = 0
            V_mask[:, col_to_remove] = 0
            W_mask[:, col_to_remove] = 0
            
            # Remove the column to get r x n*m etc
            U_init = np.delete(U_mask, col_to_remove, axis=1)
            V_init = np.delete(V_mask, col_to_remove, axis=1)
            W_init = np.delete(W_mask, col_to_remove, axis=1)
            
            result = try_rank_als(r, U_init, V_init, W_init, max_iters=50, seed=col_to_remove)
            if result is not None:
                return result
        
        # Strategy 2: Random sparse restart
        for seed in [1, 2, 3]:
            if time.time() > deadline - 15:
                break
            result = try_rank_als(r, max_iters=50, seed=seed)
            if result is not None:
                return result

    return sb
# EVOLVE-BLOCK-END
