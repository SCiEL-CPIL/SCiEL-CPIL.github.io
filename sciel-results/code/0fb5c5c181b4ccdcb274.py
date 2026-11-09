import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 110.0  # 110s safety margin (evaluator kills at 150s)

    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18

    # Build the matrix multiplication tensor T of shape (U_dim, V_dim, W_dim)
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def verify(U, V, W):
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_rec, T)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.int64)
        V = np.zeros((V_dim, rank), dtype=np.int64)
        W = np.zeros((W_dim, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2x2(U, V, W, i0, i1, j0, j1, k0, k1, col_offset):
        a00 = i0 * m + j0; a01 = i0 * m + j1
        a10 = i1 * m + j0; a11 = i1 * m + j1
        b00 = j0 * p + k0; b01 = j0 * p + k1
        b10 = j1 * p + k0; b11 = j1 * p + k1
        c00 = k0 * n + i0; c01 = k0 * n + i1
        c10 = k1 * n + i0; c11 = k1 * n + i1
        col = col_offset
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += 1
        col += 1
        # M3 = A00(B10-B00)
        U[a00, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M4 = A11(B01-B11)
        U[a11, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B10)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b10, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        return col

    # 1x2x2 block: 4 schoolbook terms, but check for rank-3 possibility
    # For A[i,j0..j1] * B[j0..j1, k0..k1] -> C[i, k0..k1]
    # This is a rank-4 tensor (4 independent terms) unless there's structure.
    # Actually for a single row i, the 4 terms are:
    #   U[i*m+j0, r]=1, V[j0*p+k0, r]=1, W[k0*n+i, r]=1
    #   U[i*m+j1, r]=1, V[j0*p+k1, r]=1, W[k1*n+i, r]=1
    #   U[i*m+j0, r]=1, V[j1*p+k0, r]=1, W[k0*n+i, r]=1
    #   U[i*m+j1, r]=1, V[j1*p+k1, r]=1, W[k1*n+i, r]=1
    # These 4 are linearly independent (each has a unique V or W index), so rank is 4.
    # No improvement possible for 1x2x2.

    def schoolbook_1x2x2(U, V, W, i, j0, j1, k0, k1, col_offset):
        col = col_offset
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        return col

    # Build the 66-rank Strassen-based decomposition
    def build_strassen_66():
        total_rank = 6 * 7 + 6 * 4  # 42 + 24 = 66
        U = np.zeros((U_dim, total_rank), dtype=np.int64)
        V = np.zeros((V_dim, total_rank), dtype=np.int64)
        W = np.zeros((W_dim, total_rank), dtype=np.int64)
        col = 0
        # Top 2 rows (i=0,1)
        for j0 in [0, 2]:
            j1 = j0 + 1
            for k0 in [0, 2, 4]:
                k1 = k0 + 1
                col = strassen_2x2x2(U, V, W, 0, 1, j0, j1, k0, k1, col)
        # Bottom 1 row (i=2)
        for j0 in [0, 2]:
            j1 = j0 + 1
            for k0 in [0, 2, 4]:
                k1 = k0 + 1
                col = schoolbook_1x2x2(U, V, W, 2, j0, j1, k0, k1, col)
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook as ultimate fallback
    best = schoolbook()
    best_rank = n * m * p

    # Try the Strassen-based 66-rank decomposition
    strassen_result = build_strassen_66()
    if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
        best = strassen_result
        best_rank = 66
    else:
        # Fallback: try a different partitioning
        # Alternative: top 1 row schoolbook, bottom 2 rows Strassen
        alt_rank = 6 * 4 + 6 * 7  # 24 + 42 = 66
        U = np.zeros((U_dim, alt_rank), dtype=np.int64)
        V = np.zeros((V_dim, alt_rank), dtype=np.int64)
        W = np.zeros((W_dim, alt_rank), dtype=np.int64)
        col = 0
        # Top 1 row (i=0)
        for j0 in [0, 2]:
            j1 = j0 + 1
            for k0 in [0, 2, 4]:
                k1 = k0 + 1
                col = schoolbook_1x2x2(U, V, W, 0, j0, j1, k0, k1, col)
        # Bottom 2 rows (i=1,2)
        for j0 in [0, 2]:
            j1 = j0 + 1
            for k0 in [0, 2, 4]:
                k1 = k0 + 1
                col = strassen_2x2x2(U, V, W, 1, 2, j0, j1, k0, k1, col)
        if verify(U, V, W):
            best = {"U": U, "V": V, "W": W}
            best_rank = alt_rank

    # Now try to find a lower-rank decomposition via combinatorial search
    # Strategy: try to express the 66-term decomposition with fewer terms
    # by finding linear dependencies or alternative factorizations.
    
    # Approach: For the 1x2x2 blocks (24 terms), check if we can share
    # factors across blocks. The key insight: in the full 3x4x6 problem,
    # different blocks share U/V/W indices.
    
    # Try a direct search for rank < 66 by attempting to find a valid
    # decomposition with rank 65, 64, etc.
    # This is extremely hard in general, but we can try a structured approach.
    
    # Alternative structured approach: use a 2x2x2 Strassen for the top 2 rows
    # but with a different column partitioning.
    # Partition columns of B into {0,1,2}, {3,4,5} instead of pairs.
    # Then for top 2 rows: 2x4x3 and 2x4x3 blocks.
    # 2x4x3: partition cols of A into {0,1} and {2,3}
    #   2x2x3 + 2x2x3
    #   2x2x3: partition cols of B into {0,1} and {2}
    #     2x2x2 (rank 7) + 2x2x1 (rank 4)
    #   So 2x2x3 has rank 11
    #   2x4x3 has rank 22
    #   Two 2x4x3 blocks: 44
    # Bottom row: 1x4x6
    #   Partition into 1x2x2 blocks: 6 blocks * rank 4 = 24
    # Total: 44 + 24 = 68. Worse.
    
    # Try: partition cols of B into {0,1,2,3}, {4,5}
    # Top 2 rows: 2x4x4 + 2x4x2
    #   2x4x4: 2x2x2*4 = 4*7 = 28 (using Strassen for each 2x2x2)
    #   Wait, 2x4x4: partition cols of A into {0,1} and {2,3}, cols of B into {0,1} and {2,3}
    #   4 blocks of 2x2x2, each rank 7: total 28
    #   2x4x2: partition cols of A into {0,1} and {2,3}
    #   2 blocks of 2x2x2, each rank 7: total 14
    # Top 2 rows total: 28 + 14 = 42
    # Bottom row: 1x4x6
    #   Partition cols of A into {0,1} and {2,3}, cols of B into {0,1,2}, {3,4,5}
    #   1x2x3 * 2 + 1x2x3 * 2
    #   1x2x3: rank 6 (3 for each of 2 columns of B... no)
    #   Actually 1x2x3: 6 terms (2*3), can we do better?
    #   For a single row, 1x2x3 is just 6 independent terms. Rank 6.
    #   So 4 blocks of 1x2x3: 4 * 6 = 24
    # Total: 42 + 24 = 66. Same.
    
    # Try another approach: can we do 3x4x6 as a combination of larger Strassen?
    # 3x4x6 = 2x4x6 + 1x4x6
    # 2x4x6: partition into 2x2x6 + 2x2x6
    #   2x2x6: partition cols of B into 3 groups of 2
    #   3 blocks of 2x2x2: 3 * 7 = 21
    #   So 2x2x6 has rank 21
    #   2x4x6: 2 * 21 = 42
    # 1x4x6: partition into 1x2x6 + 1x2x6
    #   1x2x6: partition cols of B into 3 groups of 2
    #   3 blocks of 1x2x2: 3 * 4 = 12
    #   1x4x6: 2 * 12 = 24
    # Total: 42 + 24 = 66. Same.
    
    # The 66-rank solution seems to be the natural Strassen-based answer.
    # Let's try a randomized search to see if we can find rank 65 or lower.
    
    if time.time() < DEADLINE - 10:
        # Try a limited randomized search for lower rank
        # This is very unlikely to succeed for tensor rank, but worth trying.
        try:
            # Attempt: modify the 66-rank decomposition slightly to see
            # if we can merge two terms.
            # Two terms r1 and r2 can be merged if there exist scalars
            # such that the resulting tensor still matches T.
            # This requires U[:,r1], V[:,r1], W[:,r1] to be "close" to
            # U[:,r2], V[:,r2], W[:,r2] in a specific sense.
            
            # Check if any two columns in the factor matrices are linearly
            # dependent in a way that allows merging.
            U_mat = best["U"].astype(np.float64)
            V_mat = best["V"].astype(np.float64)
            W_mat = best["W"].astype(np.float64)
            
            # Check rank of U_mat, V_mat, W_mat
            # If any has rank < 66, we might be able to reduce.
            # But for a valid decomposition, each matrix should have full rank
            # in the relevant sense.
            
            # Try to find if any pair of terms can be replaced by a single term
            # with half-integer coefficients.
            # For terms r and s, we need:
            #   u_r * v_r * w_r + u_s * v_s * w_s = u_new * v_new * w_new
            # for all (i,j,k) where the original tensor has a 1.
            
            # This is a very constrained problem. Let's try a few random pairs.
            rng = np.random.RandomState(42)
            found_better = False
            
            for attempt in range(200):
                if time.time() > DEADLINE - 5:
                    break
                r = rng.randint(66)
                s = rng.randint(66)
                if r == s:
                    continue
                # Check if we can replace terms r and s with a single term
                # This requires u_r*v_r*w_r + u_s*v_s*w_s to be rank-1.
                # Compute the tensor contribution of terms r and s.
                T_rs = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r]) \
                     + np.einsum('i,j,k->ijk', U_mat[:, s], V_mat[:, s], W_mat[:, s])
                # Check if T_rs has tensor rank 1
                # Flatten and check
                T_rs_flat = T_rs.flatten()
                nonzero = np.nonzero(T_rs_flat)[0]
                if len(nonzero) > 12:  # Too many nonzeros for rank 1
                    continue
                # A rank-1 tensor in 12x24x18 has at most min(12,24,18)=12 nonzeros
                # in a specific pattern. Check if it's rank 1.
                # Reshape and use SVD-like approach
                M1 = T_rs.reshape(12, 24*18)
                sv = np.linalg.svd(M1, compute_uv=False)
                if len(sv) == 0 or sv[0] < 1e-10:
                    continue
                if len(sv) > 1 and sv[1] / sv[0] < 1e-8:
                    # Might be rank 1
                    # Try to extract the rank-1 factors
                    u_new = M1[:, 0]
                    # Normalize
                    norm = np.linalg.norm(u_new)
                    if norm < 1e-10:
                        continue
                    u_new = u_new / norm
                    # Find v_new and w_new from the residual
                    # This is complex; skip for now
                    pass
        except Exception:
            pass
    
    # Final verification and cleanup
    if not verify(best["U"], best["V"], best["W"]):
        # Ultimate fallback: schoolbook
        best = schoolbook()
        best_rank = n * m * p

    # Ensure integer/half-integer entries
    U_final = best["U"]
    V_final = best["V"]
    W_final = best["W"]
    
    # Check for half-integers: multiply by 2 and see if all even
    # If not, the entries are already integers
    result = {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
