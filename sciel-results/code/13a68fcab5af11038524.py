"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 with search for rank 27."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def build_strassen_28():
        """Build the rank-28 Strassen decomposition for 2x4 x 4x4."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
        V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
        W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

        # Strassen 7 terms for 2x2 matrix multiplication
        strassen_terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        col = 0
        for b_col_block in range(2):
            for a_col_block in range(2):
                a_base = 2 * a_col_block
                b_row_base = 2 * a_col_block
                b_col_base = 2 * b_col_block
                c_col_base = 2 * b_col_block
                
                for u_t, v_t, w_t in strassen_terms:
                    for loc, coeff in u_t:
                        r = loc // 2
                        c = loc % 2
                        U[r * 4 + a_base + c, col] += coeff
                    
                    for loc, coeff in v_t:
                        r = loc // 2
                        c = loc % 2
                        V[(b_row_base + r) * 4 + b_col_base + c, col] += coeff
                    
                    for loc, coeff in w_t:
                        r = loc // 2
                        c = loc % 2
                        W[(c_col_base + c) * 2 + r, col] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    def validate_numpy(U, V, W):
        """Validate decomposition using numpy. Returns (valid, max_error)."""
        rank = U.shape[1]
        max_err = 0.0
        
        # Check all entries of the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s - 1.0))
        
        # Check off-diagonal entries should be 0
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    u_row = U[a]
                    v_row = V[b]
                    w_row = W[c]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s))
        
        return max_err < 1e-9, max_err

    # Build baseline rank-28
    base = build_strassen_28()
    best_result = {
        "U": base["U"].tolist(),
        "V": base["V"].tolist(),
        "W": base["W"].tolist()
    }
    best_rank = 28
    best_err = 0.0
    
    # Verify baseline
    valid, err = validate_numpy(base["U"], base["V"], base["W"])
    if not valid:
        # Should not happen, but just in case
        return best_result

    # Try to find rank 27
    target_rank = 27
    U28 = base["U"].copy()
    V28 = base["V"].copy()
    W28 = base["W"].copy()

    found = False
    
    # Precompute indices for faster ALS
    # For each a, list of (b, c) that are valid
    a_to_bc = {}
    for a in range(n*m):
        i1 = a // m
        j1 = a % m
        bc_list = []
        for b in range(m*p):
            j2 = b // p
            k1 = b % p
            if j2 != j1:
                continue
            for c in range(p*n):
                i2 = c // n
                k2 = c % n
                if i2 == i1 and k2 == k1:
                    bc_list.append((b, c))
        a_to_bc[a] = bc_list
    
    # For each b, list of (a, c) that are valid
    b_to_ac = {}
    for b in range(m*p):
        j2 = b // p
        k1 = b % p
        ac_list = []
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            if j1 != j2:
                continue
            for c in range(p*n):
                i2 = c // n
                k2 = c % n
                if i2 == i1 and k2 == k1:
                    ac_list.append((a, c))
        b_to_ac[b] = ac_list
        
    # For each c, list of (a, b) that are valid
    c_to_ab = {}
    for c in range(p*n):
        i2 = c // n
        k2 = c % n
        ab_list = []
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            if i1 != i2:
                continue
            for b in range(m*p):
                j2 = b // p
                k1 = b % p
                if j2 == j1 and k1 == k2:
                    ab_list.append((a, b))
        c_to_ab[c] = ab_list

    # Vectorized ALS for rank r
    def als_step(U, V, W, r, max_iter=50, tol=1e-12):
        """Perform ALS steps. Returns (U, V, W, error)."""
        err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            prev_err = err
            for factor in range(3):
                if factor == 0:
                    for a in range(n*m):
                        bc_list = a_to_bc[a]
                        if not bc_list:
                            U[a, :] = 0
                            continue
                        # Build M_mat: rows are (b, c), cols are r
                        # M[idx, :] = V[b, :] * W[c, :]
                        # rhs[idx] = 1.0
                        M_mat = np.zeros((len(bc_list), r))
                        rhs = np.ones(len(bc_list))
                        for idx, (b, c) in enumerate(bc_list):
                            M_mat[idx, :] = V[b, :] * W[c, :]
                        if np.any(M_mat != 0):
                            U[a, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            U[a, :] = 0
                elif factor == 1:
                    for b in range(m*p):
                        ac_list = b_to_ac[b]
                        if not ac_list:
                            V[b, :] = 0
                            continue
                        M_mat = np.zeros((len(ac_list), r))
                        rhs = np.ones(len(ac_list))
                        for idx, (a, c) in enumerate(ac_list):
                            M_mat[idx, :] = U[a, :] * W[c, :]
                        if np.any(M_mat != 0):
                            V[b, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            V[b, :] = 0
                else:
                    for c in range(p*n):
                        ab_list = c_to_ab[c]
                        if not ab_list:
                            W[c, :] = 0
                            continue
                        M_mat = np.zeros((len(ab_list), r))
                        rhs = np.ones(len(ab_list))
                        for idx, (a, b) in enumerate(ab_list):
                            M_mat[idx, :] = U[a, :] * V[b, :]
                        if np.any(M_mat != 0):
                            W[c, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            W[c, :] = 0
            
            # Check error
            valid, err = validate_numpy(U, V, W)
            if valid:
                return U, V, W, err
            if abs(prev_err - err) < tol:
                break
        return U, V, W, err

    # Search for rank 27
    for remove_col in range(28):
        if time.time() > deadline - 30:
            break
        
        keep_cols = [c for c in range(28) if c != remove_col]
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            seed = 12345 + remove_col * 1000 + attempt * 7919
            np.random.seed(seed)
            
            # Initialize from Strassen with removed column + perturbation
            scale = 0.1 + 0.05 * attempt
            U = U28[:, keep_cols].copy() + np.random.randn(n*m, target_rank) * scale
            V = V28[:, keep_cols].copy() + np.random.randn(m*p, target_rank) * scale
            W = W28[:, keep_cols].copy() + np.random.randn(p*n, target_rank) * scale
            
            U, V, W, err = als_step(U, V, W, target_rank, max_iter=100)
            
            if err < 1e-9:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            break
    
    if not found:
        # Try random initialization from scratch for rank 27?
        if time.time() < deadline - 20:
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                np.random.seed(99999 + attempt)
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                U, V, W, err = als_step(U, V, W, target_rank, max_iter=50)
                if err < 1e-9:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", rank)
