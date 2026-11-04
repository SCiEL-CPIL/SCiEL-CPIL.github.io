"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    def build_target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    def check_decomposition(U, V, W, T=None):
        if T is None:
            T = build_target_tensor()
        rank = U.shape[1]
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T, T_decomp, atol=1e-10)

    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Get schoolbook as guaranteed fallback
    U_school, V_school, W_school, rank_school = schoolbook_decomposition()

    # Try to find a lower-rank decomposition using structured search
    # Strategy: split k into two groups of 3, use separate decompositions for each
    def try_block_decomposition(r1, r2, seed=42, max_iters=500):
        """Try rank-r1 for first 3 columns, rank-r2 for last 3 columns."""
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        
        T1 = np.zeros((n * m, m * 3, 3 * n))
        T2 = np.zeros((n * m, m * 3, 3 * n))
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    T1[i * m + j, j * 3 + k, k * n + i] = 1.0
                    T2[i * m + j, j * 3 + k, k * n + i] = 1.0
        
        U1 = np.random.randn(n * m, r1) * 0.1
        V1 = np.random.randn(m * 3, r1) * 0.1
        W1 = np.random.randn(3 * n, r1) * 0.1
        U2 = np.random.randn(n * m, r2) * 0.1
        V2 = np.random.randn(m * 3, r2) * 0.1
        W2 = np.random.randn(3 * n, r2) * 0.1
        
        best_err = float('inf')
        best_state = None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute errors
            err1 = 0.0
            for r_idx in range(r1):
                err1 += np.sum((T1 - np.einsum('a,b,c->abc', U1[:, r_idx], V1[:, r_idx], W1[:, r_idx]))**2)
            err2 = 0.0
            for r_idx in range(r2):
                err2 += np.sum((T2 - np.einsum('a,b,c->abc', U2[:, r_idx], V2[:, r_idx], W2[:, r_idx]))**2)
            err = np.sqrt(err1 + err2)
            
            if err < best_err:
                best_err = err
                best_state = (U1.copy(), V1.copy(), W1.copy(), U2.copy(), V2.copy(), W2.copy())
            
            if err < 1e-10:
                break
            
            # Perturb randomly
            if np.random.rand() < 0.5:
                r_idx = np.random.randint(r1)
                f = np.random.choice(3)
                if f == 0: U1[:, r_idx] += np.random.randn(n * m) * 0.05
                elif f == 1: V1[:, r_idx] += np.random.randn(m * 3) * 0.05
                else: W1[:, r_idx] += np.random.randn(3 * n) * 0.05
            else:
                r_idx = np.random.randint(r2)
                f = np.random.choice(3)
                if f == 0: U2[:, r_idx] += np.random.randn(n * m) * 0.05
                elif f == 1: V2[:, r_idx] += np.random.randn(m * 3) * 0.05
                else: W2[:, r_idx] += np.random.randn(3 * n) * 0.05
        
        if best_err < 1e-10 and best_state is not None:
            U1, V1, W1, U2, V2, W2 = best_state
            U = np.hstack([U1, U2])
            V = np.zeros((m * p, r1 + r2))
            V[:m*3, :r1] = V1
            V[m*3:, r1:] = V2
            W = np.zeros((p * n, r1 + r2))
            W[:3*n, :r1] = W1
            W[3*n:, r1:] = W2
            return U, V, W
        
        return None

    # Try various rank splits
    for total_r in range(38, 54):
        if time.time() > deadline:
            break
        for r1 in range(max(1, total_r // 2 - 3), min(total_r - 1, total_r // 2 + 4)):
            r2 = total_r - r1
            if r2 < 1:
                continue
            result = try_block_decomposition(r1, r2, seed=1000 + total_r * 10 + r1, max_iters=100)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
