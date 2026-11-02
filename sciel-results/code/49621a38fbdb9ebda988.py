# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin
    
    n, m, p = 2, 2, 7
    
    # Build the target tensor T[i, j, k]
    # i: 0..3 (A index), j: 0..13 (B index), k: 0..13 (C index)
    # T[i, j, k] = 1 if A[i] * B[j] contributes to C[k], else 0
    # Convention:
    # A is 2x2, i = r_a * m + c_a = r_a * 2 + c_a
    # B is 2x7, j = r_b * p + c_b = r_b * 7 + c_b
    # C is 2x7, k = c_c * n + r_c = c_c * 2 + r_c (based on W[k*n+i] convention analysis)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        r_a, c_a = divmod(i, m)
        for j in range(m*p):
            r_b, c_b = divmod(j, p)
            if c_a == r_b:
                r_c = r_a
                c_c = c_b
                k = c_c * n + r_c
                T[i, j, k] = 1.0

    # Helper to verify a solution
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err < 1e-9

    # Helper to convert to list of lists with integer/half-integer entries if possible
    def to_list(mat):
        mat = np.array(mat)
        # Round to nearest 0.5
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded):
            return mat_rounded.tolist()
        else:
            return mat.tolist()

    # ALS Search
    def run_als(rank, max_iters=50, tol=1e-10, seed=0):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
            
            # Update U
            M = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                M[:, r] = np.kron(W[:, r], V[:, r])
            
            for i in range(n*m):
                target = T[i, :, :].flatten()
                u_i, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
                U[i, :] = u_i
            
            # Update V
            N = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                N[:, r] = np.kron(W[:, r], U[:, r])
                
            for j in range(m*p):
                target = T[:, j, :].flatten()
                v_j, _, _, _ = np.linalg.lstsq(N, target, rcond=None)
                V[j, :] = v_j
                
            # Update W
            P = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                P[:, r] = np.kron(V[:, r], U[:, r])
                
            for k in range(p*n):
                target = T[:, :, k].flatten()
                w_k, _, _, _ = np.linalg.lstsq(P, target, rcond=None)
                W[k, :] = w_k
                
            # Check error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < tol:
                break
                
        if best_err < 1e-6:
            return (best_U, best_V, best_W), best_err
        else:
            return None, best_err

    # Try to find low rank solutions
    # Theoretical lower bound for 2x2x7 is around 21 (based on WFTA/FFT ideas).
    # Let's try ranks from 21 up to 25.
    for target_rank in range(21, 26):
        if time.time() > deadline - 10:
            break
            
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            res, err = run_als(target_rank, max_iters=30, tol=1e-10, seed=seed)
            if res is not None:
                U, V, W = res
                # Verify
                if verify(U, V, W):
                    return {
                        "U": to_list(U),
                        "V": to_list(V),
                        "W": to_list(W)
                    }

    # Fallback: Construct a known valid solution (Rank 28: 3*7 + 4*1)
    # This is the schoolbook for the last column + Strassen for the first 3 pairs of columns.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen decomposition for 2x2x2
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_W_local = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    strassen_V_local = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # Apply Strassen for column pairs (0,1), (2,3), (4,5)
    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + (2 * b)
                elif loc_idx == 1: v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2: v_idx = 1 * p + (2 * b)
                elif loc_idx == 3: v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0: w_idx = (2 * b) * n + 0
                elif loc_idx == 1: w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2: w_idx = (2 * b) * n + 1
                elif loc_idx == 3: w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            col += 1
            
    # Schoolbook for the last column (index 6)
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    schoolbook_V_local = [
        {0: 1},
        {1: 1},
        {0: 1},
        {1: 1}
    ]
    
    schoolbook_W_local = [
        {0: 1},
        {0: 1},
        {1: 1},
        {1: 1}
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0: v_idx = 0 * p + 6
            elif loc_idx == 1: v_idx = 1 * p + 6
            V[v_idx][col] += coeff
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0: w_idx = 6 * n + 0
            elif loc_idx == 1: w_idx = 6 * n + 1
            W[w_idx][col] += coeff
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
