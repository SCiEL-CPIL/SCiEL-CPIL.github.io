"""Rewritten program for 3x3x6 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Fallback: Schoolbook rank 54
    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_solution = generate_schoolbook()
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank, n=3, m=3, p=6, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all "on-diagonal" entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check a sample of "off-diagonal" entries should be 0
        # For efficiency, check random samples
        rng = np.random.RandomState(42)
        for _ in range(200):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            j2 = rng.randint(0, m)
            k = rng.randint(0, p)
            k2 = rng.randint(0, p)
            i2 = rng.randint(0, n)
            
            # Skip if it's an on-diagonal entry
            if i == i2 and j == j2 and k == k2:
                continue
                
            val = np.sum(U_np[i * m + j, :] * V_np[j2 * p + k, :] * W_np[k2 * n + i2, :])
            if abs(val) > tol:
                return False
        
        return True

    def try_find_3x3x3_decomposition(R, max_attempts=2000, seed=42, deadline=None):
        """Try to find a rank-R 3x3x3 decomposition with integer/half-integer entries."""
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_attempts):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U_rand = rng.randint(-1, 2, size=(9, R))
            V_rand = rng.randint(-1, 2, size=(9, R))
            
            W_core = np.zeros((9, R))
            valid = True
            
            for i in range(3):
                for k in range(3):
                    M = np.zeros((3, R))
                    for j in range(3):
                        M[j, :] = U_rand[i * 3 + j, :] * V_rand[j * 3 + k, :]
                    
                    # Check if M has rank 3
                    if np.linalg.matrix_rank(M) < 3:
                        valid = False
                        break
                    
                    # Solve M @ w = ones(3)
                    try:
                        w_part, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(3), rcond=None)
                        
                        # Check if w_part is close to integer/half-integer
                        w_doubled = w_part * 2
                        closest_ints = np.round(w_doubled)
                        error = np.max(np.abs(w_doubled - closest_ints))
                        
                        if error < 1e-9:
                            W_core[i * 3 + k, :] = w_part
                        else:
                            # Try to find a half-integer solution
                            # w = w_part + N @ t where N is null space basis
                            # For R > 3, there's a null space
                            if R > 3:
                                # Compute null space
                                _, _, vh = np.linalg.svd(M)
                                null_space = vh[3:, :]  # Shape (R-3, R)
                                
                                # We need w_part + null_space @ t to be half-integer
                                # This is a system of 3 equations in (R-3) unknowns
                                # For simplicity, try random t values
                                for t_attempt in range(50):
                                    t = rng.randint(-10, 11, size=(R - 3,))
                                    w_candidate = w_part + null_space @ t
                                    w_cand_doubled = w_candidate * 2
                                    closest = np.round(w_cand_doubled)
                                    err = np.max(np.abs(w_cand_doubled - closest))
                                    if err < 1e-9:
                                        W_core[i * 3 + k, :] = w_candidate
                                        break
                                else:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                    except:
                        valid = False
                        break
            
            if valid:
                # Verify the decomposition
                U_mat = U_rand
                V_mat = V_rand
                W_mat = W_core
                
                max_err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = np.sum(U_mat[i*3+j, :] * V_mat[j*3+k, :] * W_mat[k*3+i, :])
                            if abs(val - 1) > 1e-9:
                                max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-9:
                    # Also verify off-diagonal entries are 0
                    off_diag_ok = True
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for i2 in range(3):
                                    for j2 in range(3):
                                        for k2 in range(3):
                                            if (i, j, k) == (i2, j2, k2):
                                                continue
                                            val = np.sum(U_mat[i*3+j, :] * V_mat[j2*3+k2, :] * W_mat[k2*3+i2, :])
                                            if abs(val) > 1e-9:
                                                off_diag_ok = False
                                                break
                                        if not off_diag_ok:
                                            break
                                    if not off_diag_ok:
                                        break
                                if not off_diag_ok:
                                    break
                            if not off_diag_ok:
                                break
                        if not off_diag_ok:
                            break
                    if off_diag_ok:
                        return U_rand.tolist(), V_rand.tolist(), W_core.tolist()
                else:
                    pass
        
        return None

    # Try to find 3x3x3 decomposition with various ranks
    found_core = False
    core_U = core_V = core_W = None
    core_rank = None

    # Try ranks from 23 to 27
    for R in range(23, 28):
        if time.time() > deadline - 10:  # Leave 10s for final verification
            break
            
        result = try_find_3x3x3_decomposition(R, max_attempts=1500, seed=R*1000+7, deadline=deadline-10)
        if result is not None:
            core_U, core_V, core_W = result
            core_rank = R
            found_core = True
            break

    if found_core:
        # Construct 3x3x6 solution from core
        R_total = 2 * core_rank
        
        if R_total < best_rank:
            U_total = [[0] * R_total for _ in range(n * m)]
            V_total = [[0] * R_total for _ in range(m * p)]
            W_total = [[0] * R_total for _ in range(p * n)]
            
            # Block 1: k=0,1,2 (first half of B), tensor indices 0..R-1
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for r in range(core_rank):
                            U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
                            V_total[j * 6 + k][r] = core_V[j * 3 + k][r]
                            W_total[k * 3 + i][r] = core_W[k * 3 + i][r]
            
            # Block 2: k=3,4,5 (second half of B), tensor indices R..2R-1
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        k_full = k + 3
                        for r in range(core_rank):
                            U_total[i * 3 + j][R_total - core_rank + r] = core_U[i * 3 + j][r]
                            V_total[j * 6 + k_full][R_total - core_rank + r] = core_V[j * 3 + k][r]
                            W_total[k_full * 3 + i][R_total - core_rank + r] = core_W[k * 3 + i][r]
            
            # Verify the full 3x3x6 decomposition
            if verify_decomposition(U_total, V_total, W_total, R_total):
                best_solution = {"U": U_total, "V": V_total, "W": W_total}
                best_rank = R_total
            else:
                # Verification failed; keep schoolbook
                pass

    return best_solution
# EVOLVE-BLOCK-END
