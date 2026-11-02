"""Rank-43 Laderman decomposition for 4x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 2x2: 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Block-level Strassen: 7 block products, each 2x2x2 Strassen (7 scalar) = 49
    # Block indices: A[br,bc], B[br,bc], C[br,bc] for br,bc in {0,1}
    # A block (br,bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # B block (br,bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # C block (br,bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1

    # Strassen 2x2 coefficients for A side: (local_row, local_col, coeff)
    S_A = [
        [(0,0,1),(1,1,1)],   # M1: A00+A11
        [(1,0,1),(1,1,1)],   # M2: A10+A11
        [(0,0,1)],            # M3: A00
        [(1,1,1)],            # M4: A11
        [(0,0,1),(0,1,1)],   # M5: A00+A01
        [(1,0,1),(0,0,-1)],  # M6: A10-A00
        [(0,1,1),(1,1,-1)],  # M7: A01-A11
    ]
    # Strassen 2x2 coefficients for B side
    S_B = [
        [(0,0,1),(1,1,1)],   # M1: B00+B11
        [(0,0,1)],            # M2: B00
        [(1,0,1),(1,1,-1)],  # M3: B10-B11
        [(0,1,1),(0,0,-1)],  # M4: B01-B00
        [(1,1,1)],            # M5: B11
        [(0,0,1),(0,1,1)],   # M6: B00+B01
        [(1,0,1),(1,1,1)],   # M7: B10+B11
    ]
    # C combination: for each C block (cbr,cbc), list of (strassen_idx, coeff)
    S_C = [
        (0, 0, [(0,1),(3,1),(4,-1),(6,1)]),  # C00 = M1+M4-M5+M7
        (0, 1, [(2,1),(4,1)]),               # C01 = M3+M5
        (1, 0, [(1,1),(3,1)]),               # C10 = M2+M4
        (1, 1, [(0,1),(1,-1),(2,1),(5,1)]),  # C11 = M1-M2+M3+M6
    ]

    for bi in range(7):
        # Block-level: which A blocks and B blocks are involved
        # A block combination for Strassen product bi
        a_blocks = [
            [(0,0,1),(1,1,1)],  # bi=0: A00+A11
            [(1,0,1),(1,1,1)],  # bi=1: A10+A11
            [(0,0,1)],           # bi=2: A00
            [(1,1,1)],           # bi=3: A11
            [(0,0,1),(0,1,1)],  # bi=4: A00+A01
            [(1,0,1),(0,0,-1)], # bi=5: A10-A00
            [(0,1,1),(1,1,-1)], # bi=6: A01-A11
        ]
        b_blocks = [
            [(0,0,1),(1,1,1)],  # bi=0: B00+B11
            [(0,0,1)],           # bi=1: B00
            [(1,0,1),(1,1,-1)], # bi=2: B10-B11
            [(0,1,1),(0,0,-1)], # bi=3: B01-B00
            [(1,1,1)],           # bi=4: B11
            [(0,0,1),(0,1,1)],  # bi=5: B00+B01
            [(1,0,1),(1,1,1)],  # bi=6: B10+B11
        ]
        c_blocks = [
            [(0,0,1),(1,1,1),(0,1,-1),(1,0,1)],  # bi=0: C00+C11-C01+C10
            [(1,0,1)],                              # bi=1: C10
            [(0,1,1)],                              # bi=2: C01
            [(1,1,1)],                              # bi=3: C11
            [(0,1,1),(1,1,-1)],                    # bi=4: C01-C11
            [(1,1,1)],                              # bi=5: C11
            [(0,0,1)],                              # bi=6: C00
        ]

        for s in range(7):
            r = bi * 7 + s
            # Fill U: for each A block involved, fill the 2x2 local Strassen pattern
            for (abr, abc, a_coeff) in a_blocks[bi]:
                for (lr, lc, l_coeff) in S_A[s]:
                    row = 2 * abr + lr
                    col = 2 * abc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            # Fill V: for each B block involved
            for (bbr, bbc, b_coeff) in b_blocks[bi]:
                for (lr, lc, l_coeff) in S_B[s]:
                    row = 2 * bbr + lr
                    col = 2 * bbc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            # Fill W: for each C block involved
            for (cbr, cbc, c_coeff) in c_blocks[bi]:
                for (lr, lc, l_coeff) in S_C[bi][2] if False else []:
                    pass
                # Actually S_C is per-block, not per-strassen-index. Let me fix.
                # W for this (bi, s): the C combination depends on bi, and the local pattern depends on s
                # W[c_row, c_col, r] = c_coeff * (local C pattern for strassen index s)
                # But C doesn't have a "local pattern" in the same way. 
                # Actually, the C side just sums the products. Each product bi contributes 
                # to certain C blocks with certain coefficients. The local 2x2 within each 
                # C block is just 1 (identity) because we're not doing further decomposition on C.
                # Wait, that's not right either. Let me think again.
                
                # Actually, in the recursive Strassen, each "block product" M_i = A_block * B_block
                # is itself a 2x2 matrix product computed via Strassen (7 scalar mults).
                # The result M_i is a 2x2 matrix that gets added to C blocks.
                # So for each scalar product within M_i (indexed by s), the contribution to C
                # is: c_coeff * (the s-th Strassen output pattern for 2x2)
                
                # The 2x2 Strassen output patterns (which entries of the 2x2 result each M_s contributes to):
                # M1 -> [[1,0],[0,1]] (diagonal)
                # M2 -> [[0,0],[1,0]]
                # M3 -> [[0,1],[0,0]]
                # M4 -> [[0,0],[0,1]]
                # M5 -> [[0,1],[0,1]]
                # M6 -> [[0,0],[1,1]]
                # M7 -> [[1,0],[0,0]]
                
                # Actually let me just use the standard 2x2 Strassen C combination directly.
                pass

    # Let me redo this more carefully using the standard approach.
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Standard 2x2 Strassen:
    # 7 products, each computing a 2x2 result
    # A_local[si][lr][lc] = coefficient of A[lr,lc] in product si
    # B_local[si][lr][lc] = coefficient of B[lr,lc] in product si
    # C_local[si][lr][lc] = coefficient of product si in C[lr,lc]

    A_2x2 = [
        [[1,0],[0,1]],  # M1: A00+A11
        [[0,0],[1,1]],  # M2: A10+A11
        [[1,0],[0,0]],  # M3: A00
        [[0,0],[0,1]],  # M4: A11
        [[1,1],[0,0]],  # M5: A00+A01
        [[0,0],[1,-1]], # M6: A10-A00
        [[0,1],[0,-1]], # M7: A01-A11
    ]
    B_2x2 = [
        [[1,0],[0,1]],  # M1: B00+B11
        [[1,0],[0,0]],  # M2: B00
        [[0,1],[0,-1]], # M3: B10-B11
        [[1,-1],[0,0]], # M4: B01-B00
        [[0,0],[0,1]],  # M5: B11
        [[1,1],[0,0]],  # M6: B00+B01
        [[0,1],[0,1]],  # M7: B10+B11
    ]
    C_2x2 = [
        [[1,0],[0,1]],  # M1 contributes to C00 and C11
        [[0,0],[1,0]],  # M2 contributes to C10
        [[0,1],[0,0]],  # M3 contributes to C01
        [[0,0],[0,1]],  # M4 contributes to C11
        [[0,1],[0,1]],  # M5 contributes to C01 and C11
        [[0,0],[1,1]],  # M6 contributes to C10 and C11
        [[1,0],[0,0]],  # M7 contributes to C00
    ]

    # Block-level combinations:
    # For each block product bi, which A blocks, B blocks, C blocks are involved
    block_A = [
        [(0,0,1),(1,1,1)],  # bi=0
        [(1,0,1),(1,1,1)],  # bi=1
        [(0,0,1)],           # bi=2
        [(1,1,1)],           # bi=3
        [(0,0,1),(0,1,1)],  # bi=4
        [(1,0,1),(0,0,-1)], # bi=5
        [(0,1,1),(1,1,-1)], # bi=6
    ]
    block_B = [
        [(0,0,1),(1,1,1)],  # bi=0
        [(0,0,1)],           # bi=1
        [(1,0,1),(1,1,-1)], # bi=2
        [(0,1,1),(0,0,-1)], # bi=3
        [(1,1,1)],           # bi=4
        [(0,0,1),(0,1,1)],  # bi=5
        [(1,0,1),(1,1,1)],  # bi=6
    ]
    block_C = [
        [(0,0,1),(1,1,1),(0,1,-1),(1,0,1)],  # bi=0: C00+C11-C01+C10
        [(1,0,1)],                              # bi=1: C10
        [(0,1,1)],                              # bi=2: C01
        [(1,1,1)],                              # bi=3: C11
        [(0,1,1),(1,1,-1)],                    # bi=4: C01-C11
        [(1,1,1)],                              # bi=5: C11
        [(0,0,1)],                              # bi=6: C00
    ]

    for bi in range(7):
        for s in range(7):
            r = bi * 7 + s
            # U
            for (abr, abc, a_coeff) in block_A[bi]:
                for lr in range(2):
                    for lc in range(2):
                        val = A_2x2[s][lr][lc]
                        if val != 0:
                            row = 2 * abr + lr
                            col = 2 * abc + lc
                            U[row * m + col, r] += a_coeff * val

            # V
            for (bbr, bbc, b_coeff) in block_B[bi]:
                for lr in range(2):
                    for lc in range(2):
                        val = B_2x2[s][lr][lc]
                        if val != 0:
                            row = 2 * bbr + lr
                            col = 2 * bbc + lc
                            V[row * p + col, r] += b_coeff * val

            # W
            for (cbr, cbc, c_coeff) in block_C[bi]:
                for lr in range(2):
                    for lc in range(2):
                        val = C_2x2[s][lr][lc]
                        if val != 0:
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row, r] += c_coeff * val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    return bool(np.allclose(result, 1.0, atol=1e-10))


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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


def _tensor_matrix(U_arr, V_arr, W_arr, n, m, p):
    """Build the tensor matrix M where M[ijp, r] = U[i,j,r]*V[j,k,r]*W[k,i,r]"""
    rank = U_arr.shape[1]
    M = np.zeros((n * m * p, rank), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
    return M


def _rank_reduce(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linearly dependent columns in the tensor matrix.
    Uses SVD to identify the numerical rank, then greedily removes dependent columns.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    M = _tensor_matrix(U_arr, V_arr, W_arr, n, m, p)
    
    # Get numerical rank via SVD
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
    except Exception:
        return None
    
    tol = 1e-9
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        return None
    
    # Greedy removal: sort by norm, try to remove smallest first
    kept = list(range(rank))
    removed = []
    
    norms = np.linalg.norm(M, axis=0)
    order = np.argsort(norms)
    
    for r in order:
        if time.time() > deadline:
            break
        if r not in kept:
            continue
        others = [k for k in kept if k != r]
        if not others:
            break
        try:
            x, residuals, rank_m, s_vals = np.linalg.lstsq(M[:, others], M[:, r], rcond=None)
            residual = np.linalg.norm(M[:, r] - M[:, others] @ x)
            if residual < 1e-7:
                kept.remove(r)
                removed.append(r)
        except Exception:
            pass
    
    if not removed:
        return None
    
    new_rank = len(kept)
    U_new = U_arr[:, kept].tolist()
    V_new = V_arr[:, kept].tolist()
    W_new = W_arr[:, kept].tolist()
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, new_rank):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return None


def _rank_reduce_structured(U, V, W, n, m, p, deadline):
    """
    More aggressive rank reduction: try all pairs of columns to remove,
    checking if the remaining set still spans the tensor.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    M = _tensor_matrix(U_arr, V_arr, W_arr, n, m, p)
    
    # First pass: greedy single-column removal
    kept = list(range(rank))
    norms = np.linalg.norm(M, axis=0)
    order = np.argsort(norms)
    
    for r in order:
        if time.time() > deadline:
            break
        if r not in kept:
            continue
        others = [k for k in kept if k != r]
        if not others:
            break
        try:
            x, residuals, rank_m, s_vals = np.linalg.lstsq(M[:, others], M[:, r], rcond=None)
            residual = np.linalg.norm(M[:, r] - M[:, others] @ x)
            if residual < 1e-7:
                kept.remove(r)
        except Exception:
            pass
    
    current_rank = len(kept)
    
    # Second pass: try removing additional columns with looser tolerance
    improved = True
    while improved and time.time() < deadline:
        improved = False
        norms = np.linalg.norm(M[:, kept], axis=0)
        order = np.argsort(norms)
        
        for idx in order:
            if time.time() > deadline:
                break
            r = kept[idx]
            others = [k for k in kept if k != r]
            if not others:
                break
            try:
                x, residuals, rank_m, s_vals = np.linalg.lstsq(M[:, others], M[:, r], rcond=None)
                residual = np.linalg.norm(M[:, r] - M[:, others] @ x)
                if residual < 1e-5:
                    kept.remove(r)
                    improved = True
                    break
            except Exception:
                pass
    
    if len(kept) >= rank:
        return None
    
    new_rank = len(kept)
    U_new = U_arr[:, kept].tolist()
    V_new = V_arr[:, kept].tolist()
    W_new = W_arr[:, kept].tolist()
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, new_rank):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return None


def _rank_reduce_perturbation(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by perturbing the decomposition slightly and then
    finding a lower-rank approximation. This can sometimes find reductions
    that pure linear algebra on the exact tensor misses.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    M = _tensor_matrix(U_arr, V_arr, W_arr, n, m, p)
    
    # Try various target ranks
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        # SVD-based low-rank approximation
        try:
            U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
        except Exception:
            break
        
        # Check if target_rank approximation is close enough
        if target_rank >= len(s):
            continue
        approx_error = np.sum(s[target_rank:] ** 2)
        if approx_error > 1e-10:
            continue
        
        # Extract the top target_rank components
        U_trunc = U_svd[:, :target_rank]
        s_trunc = s[:target_rank]
        Vt_trunc = Vt_svd[:target_rank, :]
        
        # Reconstruct M_approx
        M_approx = U_trunc @ np.diag(s_trunc) @ Vt_trunc
        
        # Now we need to factor M_approx back into U, V, W
        # This is the hard part - we need to find U, V, W such that
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = M_approx[i,j,k]
        
        # One approach: use the structure. Since M_approx is a sum of 
        # target_rank rank-1 tensors, each rank-1 tensor corresponds to
        # a product u_r ⊗ v_r ⊗ w_r.
        
        # For each r, extract the 3-way tensor slice
        for r in range(target_rank):
            # The r-th rank-1 component is U_svd[:,r] * s[r] * Vt_svd[r,:]
            # This gives us M_slice[i,j,k] = U_svd[i*mp+jp+k, r] * s[r] * Vt_svd[r, i*mp+jp+k]
            # But this isn't directly a rank-1 tensor in the (i,j,k) space.
            # We need to reshape it properly.
            pass
        
        # This approach is complex. Let me try a simpler heuristic.
        # Instead, just check if we can drop columns from the original.
        break
    
    return None


def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS

    # Build the rank-49 Strassen decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        return _schoolbook_decomposition(n, m, p)

    # Try to reduce rank
    best_result = result
    best_rank = rank
    
    if time.time() < deadline - 5.0:
        reduced = _rank_reduce(U, V, W, n, m, p, deadline - 3.0)
        if reduced is not None:
            best_result = reduced
            best_rank = len(reduced["U"][0])
    
    # Try structured reduction if we have time
    if time.time() < deadline - 5.0 and best_rank > 43:
        reduced2 = _rank_reduce_structured(U, V, W, n, m, p, deadline - 3.0)
        if reduced2 is not None and len(reduced2["U"][0]) < best_rank:
            best_result = reduced2
            best_rank = len(reduced2["U"][0])
    
    return best_result

# EVOLVE-BLOCK-END
