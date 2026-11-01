"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 27 search via null-space and integer constraint."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    # Build the standard rank-32 schoolbook decomposition
    def build_schoolbook():
        rank = n * m * p  # 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-28 Strassen decomposition (based on 2x2 blocks)
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    k0 = 2 * b
                    k1 = 2 * b + 1
                    
                    # U side: A[i, j]
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[j, k]
                    V_contrib = {
                        0: [(j0, k0, 1), (j1, k1, 1)],
                        1: [(j0, k0, 1)],
                        2: [(j0, k1, 1), (j1, k1, -1)],
                        3: [(j1, k0, 1), (j0, k0, -1)],
                        4: [(j1, k1, 1)],
                        5: [(j0, k0, 1), (j0, k1, 1)],
                        6: [(j1, k0, 1), (j1, k1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, k]
                    # W flat index: k * n + i
                    W_contrib = {
                        0: {(0, k0): 1, (1, k1): 1},
                        1: {(1, k0): 1, (1, k1): -1},
                        2: {(0, k1): 1, (1, k1): 1},
                        3: {(0, k0): 1, (1, k0): 1},
                        4: {(0, k0): -1, (0, k1): 1},
                        5: {(1, k1): 1},
                        6: {(0, k0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[k_loc * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # Check diagonal entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries (should be 0)
        # Total entries: (n*m) * (m*p) * (p*n) = 8 * 16 * 8 = 1024
        # Diagonal entries: n * m * p = 32
        # Off-diagonal: 992
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b_idx in range(m*p):
                j2 = b_idx // p
                k1 = b_idx % p
                for c_idx in range(p*n):
                    k2 = c_idx // n
                    i2 = c_idx % n
                    
                    # Skip diagonal
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b_idx][r] * W[c_idx][r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def is_half_integer(x, tol=1e-6):
        """Check if x is a half-integer (i.e., 2*x is an integer)."""
        val = 2 * x
        rounded = round(val)
        return abs(val - rounded) < tol

    def check_half_integer_constraint(U, V, W):
        """Check if all entries in U, V, W are half-integers."""
        for row in U:
            for x in row:
                if not is_half_integer(x):
                    return False
        for row in V:
            for x in row:
                if not is_half_integer(x):
                    return False
        for row in W:
            for x in row:
                if not is_half_integer(x):
                    return False
        return True

    # Try to find a rank-27 decomposition by searching for a null vector in the space of rank-28 decompositions
    # Strategy: Start with rank-28 Strassen. Look for linear dependencies among the 28 tensor terms.
    # If we find a null vector c such that sum c_r T_r = 0, we can eliminate one term.
    # The new decomposition will have rank 27 with scaled terms.
    # We need to ensure the resulting entries are half-integers.
    
    def try_rank_27_search():
        if time.time() > deadline:
            return None
        
        base = build_strassen_28()
        rank28 = 28
        U28 = np.array(base["U"], dtype=float)
        V28 = np.array(base["V"], dtype=float)
        W28 = np.array(base["W"], dtype=float)
        
        # Construct the 28 tensor vectors in the full space
        dim_total = (n*m) * (m*p) * (p*n)  # 1024
        T_flat = np.zeros((rank28, dim_total))
        
        for r in range(rank28):
            idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        T_flat[r, idx] = U28[i, r] * V28[j, r] * W28[k, r]
                        idx += 1
        
        # We want to find a non-zero vector c in R^28 such that T_flat @ c = 0
        # This means the 28 tensors are linearly dependent.
        # If they are independent, rank 27 is not reachable by simple scaling from this specific rank-28 decomp.
        # However, other rank-28 decompositions might be dependent.
        
        # Check rank of T_flat
        try:
            rank_T = np.linalg.matrix_rank(T_flat, tol=1e-10)
        except:
            return None
            
        if rank_T < rank28:
            # Find null space
            U_svd, s, Vh = np.linalg.svd(T_flat)
            # Null space vectors are rows of Vh corresponding to zero singular values
            null_mask = s < 1e-10
            if np.any(null_mask):
                # Iterate through null vectors
                for r_null in range(len(null_mask)):
                    if not null_mask[r_null]:
                        continue
                    
                    c = Vh[r_null, :]
                    
                    # Normalize c so that max |c_i| = 1
                    max_c = np.max(np.abs(c))
                    if max_c == 0:
                        continue
                    c = c / max_c
                    
                    # We want to eliminate one term.
                    # Suppose we eliminate term k where c_k is large.
                    # Then we can write sum_{r != k} c_r (T_r / c_k) = T_k
                    # This doesn't directly give a rank-27 decomp unless we can absorb c_k into the other terms.
                    # Actually, the relation is sum_{r} c_r T_r = 0.
                    # This implies T_k = -sum_{r != k} (c_r / c_k) T_r.
                    # So the set {T_r | r != k} spans the same space as all {T_r}.
                    # But we need the SUM to be the same tensor.
                    # Original tensor T = sum_{r=0}^{27} T_r.
                    # If sum c_r T_r = 0, then T_k = -sum_{r != k} (c_r/c_k) T_r.
                    # Then T = sum_{r != k} T_r + T_k = sum_{r != k} T_r - sum_{r != k} (c_r/c_k) T_r
                    # = sum_{r != k} (1 - c_r/c_k) T_r.
                    # So the new coefficients for r != k are alpha_r = 1 - c_r/c_k.
                    # The new decomposition is sum_{r != k} alpha_r T_r.
                    # This has rank 27 (assuming alpha_r != 0).
                    
                    # We need alpha_r to be such that the resulting U, V, W entries are half-integers.
                    # U_new[:, r] = alpha_r * U28[:, r]
                    # V_new[:, r] = V28[:, r]
                    # W_new[:, r] = W28[:, r]
                    
                    # We can choose which term to eliminate (k) to make alpha_r "nice".
                    # Ideally, we want alpha_r to be integers or half-integers.
                    
                    best_alpha = None
                    best_k = -1
                    best_score = -1
                    
                    for k in range(rank28):
                        if abs(c[k]) < 1e-6:
                            continue
                        
                        alpha = 1.0 - c / c[k]
                        alpha[k] = 0.0
                        
                        # Check if alpha (excluding k) are "nice"
                        # We want alpha_r to be close to half-integers?
                        # Or maybe we can scale the whole thing?
                        # If we multiply U_r by alpha_r, entries become alpha_r * U28.
                        # U28 entries are in {-1, 0, 1}.
                        # So we need alpha_r to be in {0, 0.5, 1, 1.5, 2, ...} or {-0.5, -1, ...}
                        
                        valid = True
                        max_err = 0.0
                        for r in range(rank28):
                            if r == k:
                                continue
                            if abs(alpha[r]) < 1e-6:
                                continue # Skip zero coefficients, effectively rank < 27
                            # Check if alpha[r] is half-integer
                            if not is_half_integer(alpha[r], tol=1e-5):
                                valid = False
                                break
                            # Also check resulting U entries
                            # U28 entries are -1, 0, 1.
                            # alpha * U28 should be half-integer.
                            # If alpha is half-integer, product is half-integer.
                        
                        if valid:
                            # Score: prefer smaller alpha (simpler?) or just accept
                            # Let's just take the first valid one
                            best_alpha = alpha
                            best_k = k
                            break
                    
                    if best_alpha is not None:
                        # Construct new matrices
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        new_rank = rank28 - 1
                        col_map = []
                        for r in range(rank28):
                            if r != best_k:
                                col_map.append(r)
                        
                        for i in range(n*m):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(best_alpha[r_old] * U28[i, r_old])
                            U_new.append(row)
                        
                        for i in range(m*p):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(V28[i, r_old])
                            V_new.append(row)
                        
                        for i in range(p*n):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(W28[i, r_old])
                            W_new.append(row)
                        
                        # Validate
                        if validate(U_new, V_new, W_new, new_rank):
                            # Check half-integer constraint
                            if check_half_integer_constraint(U_new, V_new, W_new):
                                return {"U": U_new, "V": V_new, "W": W_new}
                            else:
                                # Try to fix? Maybe scale?
                                # If alpha_r is 0.5, and U28 is 1, entry is 0.5. OK.
                                # If alpha_r is 1.5, and U28 is 1, entry is 1.5. OK.
                                # Why would it fail?
                                # Maybe U28 has -1. -1 * 1.5 = -1.5. OK.
                                # So it should pass if alpha is half-integer.
                                pass
        
        return None

    # Fallback: Try to find a rank-27 decomposition via randomized search / integer programming
    # This is harder. For now, stick with the null space approach.
    # If that fails, return rank 28.

    best_result = None
    best_rank = 999

    # Try rank 27
    res27 = try_rank_27_search()
    if res27:
        best_result = res27
        best_rank = 27
    else:
        # Fallback to rank 28
        res28 = build_strassen_28()
        if validate(res28["U"], res28["V"], res28["W"], 28):
            best_result = res28
            best_rank = 28
        else:
            # Fallback to rank 32
            res32 = build_schoolbook()
            if validate(res32["U"], res32["V"], res32["W"], 32):
                best_result = res32
                best_rank = 32
            else:
                # Should not happen
                raise RuntimeError("Failed to find valid decomposition")

    return best_result

# EVOLVE-BLOCK-END
