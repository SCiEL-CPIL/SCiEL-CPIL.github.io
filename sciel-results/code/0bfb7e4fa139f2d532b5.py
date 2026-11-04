"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_decomposition():
    """Build the rank-42 Strassen-based decomposition."""
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = 2, 4, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # So T[i,j,k] should equal 1 if k is the output index for A row i and B col k, else 0
    
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We need T[i,j,k] = delta_{i//m, j//p} * (j % p == k % n) ... 
    # Actually: C[i, k] = sum_j A[i,j] * B[j,k]
    # In flattened: C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    # So the tensor T[i_flat, j_flat, k_flat] = 1 if i_flat = i*m+j, j_flat = j*p+k, k_flat = k*n+i for some valid i,j,k
    # More precisely: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # Let's compute the tensor from U,V,W
    # T = sum_r outer(U[:,r], V[:,r], W[:,r])
    # Use einsum: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    diff = np.max(np.abs(T_computed - T_target))
    return diff < 1e-10, diff

def _try_rank_reduction(U_init, V_init, W_init, target_rank, max_time=60.0):
    """
    Try to reduce rank using a parity-check / linear dependence approach.
    Each rank-1 term contributes a vector to the 48-dim tensor space.
    We check if any subset of terms is linearly sufficient to span the target.
    """
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    dim = n * m * p  # 48
    
    # Target tensor flattened
    T_target = np.zeros(dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index: we need a consistent flattening
                # T[i_flat, j_flat, k_flat] -> use lexicographic order
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                # Flatten (a_idx, b_idx, c_idx) to a single index
                # Total size: (n*m) * (m*p) * (p*n) = 8 * 24 * 12 = 2304
                # But we only care about the 48 non-zero entries
                flat_idx = a_idx * (m * p) * (p * n) + b_idx * (p * n) + c_idx
                # This is too large. Let me use a dict instead.
                pass
    
    # Actually, let's just use the direct tensor approach but be smarter about it.
    # Build the 48xR matrix where entry [entry_idx, r] = U[a,r]*V[b,r]*W[c,r]
    # for the 48 non-zero tensor entries.
    
    entries = []  # list of (a_idx, b_idx, c_idx)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                entries.append((a_idx, b_idx, c_idx))
    
    R = U_np.shape[1]
    # Build the 48 x R matrix M where M[e, r] = U[a_e, r] * V[b_e, r] * W[c_e, r]
    M = np.zeros((len(entries), R))
    for e, (a, b, c) in enumerate(entries):
        M[e, :] = U_np[a, :] * V_np[b, :] * W_np[c, :]
    
    # The target is the all-ones vector of length 48
    target_vec = np.ones(len(entries))
    
    current_rank = R
    best_rank = R
    
    def _check_subset(idx_keep):
        """Check if the subset of columns sums to target."""
        if len(idx_keep) == 0:
            return False
        M_sub = M[:, idx_keep]
        # We need sum of columns = target_vec
        col_sum = M_sub.sum(axis=1)
        return np.allclose(col_sum, target_vec, atol=1e-10)
    
    # Phase 1: Greedy single removal
    removed_any = True
    while removed_any:
        removed_any = False
        for r_remove in range(current_rank):
            if time.time() - start_time > max_time * 0.6:
                break
            idx_keep = [x for x in range(current_rank) if x != r_remove]
            if _check_subset(idx_keep):
                U_np = U_np[:, idx_keep]
                V_np = V_np[:, idx_keep]
                W_np = W_np[:, idx_keep]
                M = M[:, idx_keep]
                current_rank -= 1
                best_rank = min(best_rank, current_rank)
                removed_any = True
                break
        if time.time() - start_time > max_time * 0.6:
            break
    
    # Phase 2: Pair removal
    for r1 in range(current_rank):
        if time.time() - start_time > max_time * 0.8:
            break
        for r2 in range(r1 + 1, current_rank):
            if time.time() - start_time > max_time * 0.8:
                break
            idx_keep = [x for x in range(current_rank) if x != r1 and x != r2]
            if _check_subset(idx_keep):
                U_np = U_np[:, idx_keep]
                V_np = V_np[:, idx_keep]
                W_np = W_np[:, idx_keep]
                M = M[:, idx_keep]
                current_rank -= 2
                best_rank = min(best_rank, current_rank)
                break
        if time.time() - start_time > max_time * 0.8:
            break
    
    # Phase 3: Triple removal
    if time.time() - start_time < max_time * 0.9:
        from itertools import combinations
        for r1, r2, r3 in combinations(range(current_rank), 3):
            if time.time() - start_time > max_time * 0.95:
                break
            idx_keep = [x for x in range(current_rank) if x not in (r1, r2, r3)]
            if _check_subset(idx_keep):
                U_np = U_np[:, idx_keep]
                V_np = V_np[:, idx_keep]
                W_np = W_np[:, idx_keep]
                M = M[:, idx_keep]
                current_rank -= 3
                best_rank = min(best_rank, current_rank)
                break
    
    if best_rank < 42:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }
    return None

def _standard_decomposition():
    """Build the standard rank-48 decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x3_block_decomposition():
    """
    Try a different block partitioning: split the 4x6 matrix into 2x3 blocks.
    2x4 * 4x6 = 2x(2+2) * (2+2)x(3+3)
    This gives 4 subproblems of 2x2 * 2x3, each of rank 5 (known optimal for 2x2*2x3).
    Total rank: 4 * 5 = 20? No, that's not right because of the shared middle dimension.
    
    Actually, let's think of it as:
    A (2x4) * B (4x6)
    Split B into columns: B = [B1 | B2] where B1 is 4x3, B2 is 4x3
    Split A into rows: A = [A1; A2] where A1 is 2x2, A2 is 2x2
    
    C = A1*B1 + A2*B2 (but this isn't quite right either)
    
    Let me use a different approach: view 4x6 as 2 blocks of 2x3.
    A (2x4) * B (4x6):
    Split A into [A_left (2x2) | A_right (2x2)]
    Split B into [[B_tl (2x3); B_bl (2x3)] [B_tr (2x3); B_br (2x3)]]
    
    C = A_left * B_tl + A_right * B_tr  (for left half of C, 2x3)
    C = A_left * B_bl + A_right * B_br  (for right half of C, 2x3)
    
    Each 2x2 * 2x3 multiplication has rank at most 5 (since 2*2*3=12 schoolbook, 
    but we can do better). Actually the bilinear complexity of 2x2*2x3 is 5.
    
    So total rank could be 4 * 5 = 20? That seems too low. Let me verify.
    
    Actually, I think the issue is that we're double-counting. Let me just try it and see.
    """
    n, m, p = 2, 4, 6
    # Use 2x2 * 2x3 blocks
    # A is 2x4, split into two 2x2 blocks: A = [A0 | A1]
    # B is 4x6, split into four 2x3 blocks:
    #   B = [[B00, B01],
    #        [B10, B11]]
    # where B00, B01 are top 2 rows (cols 0-2, 3-5)
    #       B10, B11 are bottom 2 rows (cols 0-2, 3-5)
    #
    # C (2x6) = [A0*B00 + A1*B10 | A0*B01 + A1*B11]
    #
    # Each 2x2 * 2x3 product can be done in rank 5.
    # We have 4 such products, but they share the same A blocks.
    #
    # For 2x2 * 2x3, the rank-5 decomposition uses:
    # (a11+b11)(b11) type terms... Let me look up the known result.
    # Actually, for 2x2 * 2x3, the bilinear complexity is 5.
    #
    # But we can't just multiply ranks because the A matrices are shared.
    # Let me just construct it explicitly.
    
    # For 2x2 * 2x3, known rank-5 decomposition:
    # Given X (2x2) = [[x11, x12], [x21, x22]] and Y (2x3) = [[y11,y12,y13],[y21,y22,y23]]
    # Z = X*Y (2x3)
    #
    # One known rank-5 scheme:
    # m1 = (x11+x21+x22)*(y11+y21+y22)  -- no, this doesn't work directly
    #
    # Actually, let me just use the schoolbook for 2x2*2x3 which is rank 6,
    # and see if we can do better.
    # Schoolbook: 2*2*3 = 12 terms, but many can be combined.
    # Actually for 2x2 * 2x3, the minimum rank is 5.
    
    # Let me use a simpler approach: just try the 2x2*2x2 Strassen (rank 7 per block)
    # but with 2x3 blocks instead.
    #
    # Actually, let me just return the standard 42-rank decomposition and focus
    # on improving the rank reduction.
    pass
    return _strassen_decomposition()

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # Start with the known rank-42 decomposition
    result_42 = _strassen_decomposition()
    
    # Verify the initial decomposition
    valid, diff = _verify_decomposition(result_42["U"], result_42["V"], result_42["W"])
    
    if not valid:
        # Fallback to standard rank-48
        result_std = _standard_decomposition()
        valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
        if valid_std:
            return result_std
        return result_42
    
    best_result = result_42
    best_rank = 42
    
    # Try rank reduction with generous time budget
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        reduced = _try_rank_reduction(
            result_42["U"], result_42["V"], result_42["W"],
            target_rank=41,
            max_time=min(80.0, deadline - elapsed - 10.0)
        )
        if reduced is not None:
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Final verification
    valid_final, _ = _verify_decomposition(best_result["U"], best_result["V"], best_result["W"])
    if not valid_final:
        result_std = _standard_decomposition()
        valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
        if valid_std:
            return result_std
    
    return best_result

# EVOLVE-BLOCK-END
