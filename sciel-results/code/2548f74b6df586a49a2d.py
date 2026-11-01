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
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Local 2x2 index mapping: (r,c) -> r*2+c
        # U local: A[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        # V local: B[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        # W local: C[i, j] where i in {0,1}, j in {0,1} -> flat: i*2+j
        
        # Strassen terms as (u_indices, v_indices, w_indices) with coefficients
        # Format: list of (local_index, coeff)
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
        # 4 block products: A_L*B_TL, A_R*B_BL -> C_L; A_L*B_TR, A_R*B_BR -> C_R
        for b_col_block in range(2):  # 0: left cols of C, 1: right cols of C
            for a_col_block in range(2):  # 0: left cols of A, 1: right cols of A
                # A block: columns 2*a_col_block + {0,1}
                # B block: rows 2*a_col_block + {0,1}, columns 2*b_col_block + {0,1}
                # C block: rows {0,1}, columns 2*b_col_block + {0,1}
                
                a_base = 2 * a_col_block  # column offset in A
                b_row_base = 2 * a_col_block  # row offset in B
                b_col_base = 2 * b_col_block  # column offset in B
                c_col_base = 2 * b_col_block  # column offset in C
                
                for u_t, v_t, w_t in strassen_terms:
                    # U: A[i, a_base + j] -> flat index i*4 + (a_base + j)
                    for loc, coeff in u_t:
                        r = loc // 2
                        c = loc % 2
                        U[r * 4 + a_base + c, col] += coeff
                    
                    # V: B[b_row_base + r, b_col_base + c] -> flat index (b_row_base+r)*4 + (b_col_base+c)
                    for loc, coeff in v_t:
                        r = loc // 2
                        c = loc % 2
                        V[(b_row_base + r) * 4 + b_col_base + c, col] += coeff
                    
                    # W: C[r, c_col_base + c] -> flat index (c_col_base+c)*2 + r
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
        # For efficiency, check all (a, b, c) triples where indices don't match
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
    
    # Precompute the "mask" of valid tensor entries for faster validation
    # valid_entries[i, j, k] is True if C[i,k] depends on A[i,j] B[j,k]
    # In our tensor, the entry T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # and 0 otherwise.
    
    # Vectorized ALS for rank r
    def als_step(U, V, W, r, max_iter=50, tol=1e-12):
        """Perform ALS steps. Returns (U, V, W, error)."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            prev_err = np.inf
            for factor in range(3):
                if factor == 0:
                    M_mat = np.zeros((m*p*p*n, r))
                    rhs = np.zeros(m*p*p*n)
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        # Fill rows for this a
                        # Row index in M_mat: b*p*n + c
                        # We need V[b] * W[c]
                        # And rhs is 1 if (i1, j1, k1) matches (i2, j2, k2)
                        # i2 = c // n, j2 = b // p, k2 = c % n
                        # Condition: i1==i2, j1==j2, k1==k2
                        # k1 = b % p
                        # So for fixed a (i1, j1), we look at b, c.
                        # j2 = b // p must be j1.
                        # k2 = c % n must be b % p.
                        # i2 = c // n must be i1.
                        
                        # Iterate b
                        for b in range(m*p):
                            j2 = b // p
                            k1 = b % p
                            if j2 != j1:
                                continue
                            # Iterate c
                            for c in range(p*n):
                                i2 = c // n
                                k2 = c % n
                                if i2 != i1 or k2 != k1:
                                    continue
                                # This is a valid entry
                                idx = b * p * n + c
                                M_mat[idx, :] = V[b, :] * W[c, :]
                                rhs[idx] = 1.0
                        # Solve for U[a]
                        # Note: M_mat has many zero rows.
                        # We can extract non-zero rows.
                        mask = np.any(M_mat != 0, axis=1)
                        if np.sum(mask) > 0:
                            M_sub = M_mat[mask, :]
                            rhs_sub = rhs[mask]
                            U[a, :], _, _, _ = np.linalg.lstsq(M_sub, rhs_sub, rcond=None)
                        else:
                            U[a, :] = 0
                elif factor == 1:
                    for b in range(m*p):
                        j2 = b // p
                        k1 = b % p
                        M_mat = np.zeros((n*m*p*n, r))
                        rhs = np.zeros(n*m*p*n)
                        for a in range(n*m):
                            i1 = a // m
                            j1 = a % m
                            if j1 != j2:
                                continue
                            for c in range(p*n):
                                i2 = c // n
                                k2 = c % n
                                if i2 != i1 or k2 != k1:
                                    continue
                                idx = a * p * n + c
                                M_mat[idx, :] = U[a, :] * W[c, :]
                                rhs[idx] = 1.0
                        mask = np.any(M_mat != 0, axis=1)
                        if np.sum(mask) > 0:
                            M_sub = M_mat[mask, :]
                            rhs_sub = rhs[mask]
                            V[b, :], _, _, _ = np.linalg.lstsq(M_sub, rhs_sub, rcond=None)
                        else:
                            V[b, :] = 0
                else:
                    for c in range(p*n):
                        i2 = c // n
                        k2 = c % n
                        M_mat = np.zeros((n*m*m*p, r))
                        rhs = np.zeros(n*m*m*p)
                        for a in range(n*m):
                            i1 = a // m
                            j1 = a % m
                            if i1 != i2:
                                continue
                            for b in range(m*p):
                                j2 = b // p
                                k1 = b % p
                                if j2 != j1 or k1 != k2:
                                    continue
                                idx = a * m * p + b
                                M_mat[idx, :] = U[a, :] * V[b, :]
                                rhs[idx] = 1.0
                        mask = np.any(M_mat != 0, axis=1)
                        if np.sum(mask) > 0:
                            M_sub = M_mat[mask, :]
                            rhs_sub = rhs[mask]
                            W[c, :], _, _, _ = np.linalg.lstsq(M_sub, rhs_sub, rcond=None)
                        else:
                            W[c, :] = 0
            
            # Check error
            valid, err = validate_numpy(U, V, W)
            if valid:
                return U, V, W, err
            if abs(prev_err - err) < tol:
                break
            prev_err = err
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
        # This is less likely to work but worth a shot if time permits
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
