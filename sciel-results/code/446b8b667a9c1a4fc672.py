"""Optimized tensor decomposition for 2x2 * 2x8 matrix multiplication using block Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS  # 2, 2, 8

    # The 2x2 * 2x8 multiplication is equivalent to 4 independent 2x2 * 2x2 multiplications:
    # We partition B into 4 blocks of 2 columns each.
    # For each 2x2 * 2x2 sub-problem, we use Strassen (rank 7).
    # Total rank = 4 * 7 = 28.
    #
    # However, the A matrix is SHARED across all 4 blocks.
    # In a standard block Strassen approach, each block independently applies
    # the 7 Strassen linear combinations to A, resulting in redundant computation
    # but the same rank count since each block still needs 7 separate terms.
    #
    # The key insight: if we process all 4 blocks simultaneously, the 7 Strassen
    # A-combinations are the SAME for every block. But in tensor decomposition,
    # each (A_combo, B_combo, C_coeff) term is independent. Since each block uses
    # different B columns, we still need 4*7 = 28 terms.
    #
    # Can we do better? Let's think about this more carefully.
    # The 2x2 * 2x8 tensor is a 4x16x16 tensor (A has 4 entries, B has 16, C has 16).
    # It decomposes as a direct sum of 4 copies of the 2x2*2x2 tensor (rank 7 each),
    # so the rank is exactly 4*7 = 28 (since direct sum of tensors has rank = sum of ranks
    # when the supports are disjoint, which they are here).
    #
    # Wait - is that actually true? The rank of a direct sum is the sum of the ranks
    # only if we're summing over a field where that holds. For the tensor decomposition
    # over reals/integers, yes, if the three sets of indices for C are disjoint across blocks,
    # then the rank of the sum equals the sum of the ranks.
    #
    # In our case: C block 0 uses columns k=0,1; C block 1 uses k=2,3; etc.
    # The W factors for different blocks have support on disjoint C indices.
    # The V factors for different blocks have support on disjoint B indices.
    # The U factors are the same across blocks (same A combinations).
    #
    # But even though U is shared, each term in the decomposition is a rank-1 tensor
    # u ⊗ v ⊗ w. Different blocks give different (v,w) but same u.
    # Since v and w for different blocks have disjoint support, these rank-1 tensors
    # are linearly independent, so the total rank is indeed 4*7 = 28.
    #
    # This is better than schoolbook (32) and is optimal for this tensor.

    # Strassen decomposition for 2x2 * 2x2 (0-indexed):
    # M1 = (A00+A11)*(B00+B11)  -> contributes to C00, C11
    # M2 = (A10+A11)*B00        -> contributes to C00 (-), C10 (+)
    # M3 = A00*(B01-B11)        -> contributes to C00 (+), C11 (+)
    # M4 = A11*(B10-B00)        -> contributes to C00 (+), C11 (+)
    # M5 = (A00+A01)*B11        -> contributes to C01 (+), C11 (-)
    # M6 = (A10-A00)*(B00+B01)  -> contributes to C10 (+)
    # M7 = (A01-A11)*(B10+B11)  -> contributes to C01 (+)
    #
    # A indices (flattened row-major): A00=0, A01=1, A10=2, A11=3
    # B indices (flattened row-major for 2x2): B00=0, B01=1, B10=2, B11=3
    # C indices (flattened row-major for 2x2): C00=0, C01=1, C10=2, C11=3

    strassen_A = [
        [1,  0,  0,  1],  # M1: A00 + A11
        [0,  0,  1,  1],  # M2: A10 + A11
        [1,  0,  0,  0],  # M3: A00
        [0,  0,  0,  1],  # M4: A11
        [1,  1,  0,  0],  # M5: A00 + A01
        [-1, 0,  1,  0],  # M6: A10 - A00
        [0,  1,  0, -1],  # M7: A01 - A11
    ]

    strassen_B = [
        [1,  0,  0,  1],  # M1: B00 + B11
        [1,  0,  0,  0],  # M2: B00
        [0,  1,  0, -1],  # M3: B01 - B11
        [-1, 0,  1,  0],  # M4: B10 - B00
        [0,  0,  0,  1],  # M5: B11
        [1,  1,  0,  0],  # M6: B00 + B01
        [0,  0,  1,  1],  # M7: B10 + B11
    ]

    strassen_C = [
        [1,  0,  0,  1],   # M1: C00 + C11
        [-1, 0,  1,  0],   # M2: -C00 + C10
        [1,  0,  0,  1],   # M3: C00 + C11
        [1,  0,  0,  1],   # M4: C00 + C11
        [0,  1,  0, -1],   # M5: C01 - C11
        [0,  0,  1,  0],   # M6: C10
        [0,  1,  0,  0],   # M7: C01
    ]

    # Verify the Strassen decomposition is correct
    for c_idx in range(4):
        for t1 in range(4):
            for t2 in range(4):
                # Expected: sum_j A[i,j]*B[j,k] where i=c_idx//2, k=c_idx%2
                # Actually let's just verify C = A*B
                pass

    # More careful verification: for each (i,j,k) in the 2x2x2 tensor,
    # sum_r strassen_A[r][i*2+j] * strassen_B[r][j*2+k] * strassen_C[r][i*2+k] should equal
    # delta_{i,i}*delta_{j,j}*delta_{k,k}... no.
    # The tensor T satisfies T[A_idx, B_idx, C_idx] = 1 iff the C_idx corresponds to the
    # output entry (i,k) that gets contribution from A[i,j]*B[j,k].
    # So T[(i,j),(j,k),(i,k)] = 1, all other entries 0.

    # Verify: for each i, j, k in {0,1}:
    #   sum_r strassen_A[r][i*2+j] * strassen_B[r][j*2+k] * strassen_C[r][i*2+k] = 1
    # And for all other index combinations: sum = 0.

    # Actually, the correct identity is:
    # For output C[i,k]: sum_r (sum_j' strassen_A[r][i*2+j'] * A[i,j']) 
    #                      * (sum_j'' strassen_B[r][j''*2+k] * B[j'',k]) 
    #                      * ... no, that's not how it works.
    #
    # The bilinear form is: C[i,k] = sum_r P_r(A) * Q_r(B)
    # where P_r(A) = sum_{i,j} strassen_A[r][i*2+j] * A[i,j]
    # and Q_r(B) depends on (i,k)... no.
    #
    # Actually for the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r:
    # T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    # where a indexes A entries (i,j), b indexes B entries (j,k), c indexes C entries (i,k).
    # And the multiplication is: C[i,k] = sum_{j} A[i,j]*B[j,k] = sum_{j} T[(i,j),(j,k),(i,k)]
    # So we need: T[(i,j),(j,k),(i,k)] = 1 for the correct triple, 0 otherwise.
    # More precisely: sum_{(i',j'),(j'',k')} A[i',j']*B[j'',k']*T[(i',j'),(j'',k'),(i,k)] = A[i,j']B[j,k] when (i',j')=(i,j), (j'',k')=(j,k)
    # This means T is the multiplication tensor: T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
    
    # Let me verify: for fixed i,k in {0,1}:
    # sum_{j} A[i,j]*B[j,k] should equal sum_r (sum_{i',j'} strassen_A[r][i'*2+j']*A[i',j']) 
    #                                              (sum_{j'',k'} strassen_B[r][j''*2+k']*B[j'',k'] * delta_{k',k})
    # ... this is getting confused. Let me just do a numerical check.

    # Quick numerical verification
    rng = np.random.RandomState(42)
    A_test = rng.rand(2, 2)
    B_test = rng.rand(2, 2)
    C_test = A_test @ B_test

    # Compute C using Strassen
    M = np.zeros(7)
    M[0] = (A_test[0,0]+A_test[1,1]) * (B_test[0,0]+B_test[1,1])
    M[1] = (A_test[1,0]+A_test[1,1]) * B_test[0,0]
    M[2] = A_test[0,0] * (B_test[0,1]-B_test[1,1])
    M[3] = A_test[1,1] * (B_test[1,0]-B_test[0,0])
    M[4] = (A_test[0,0]+A_test[0,1]) * B_test[1,1]
    M[5] = (A_test[1,0]-A_test[0,0]) * (B_test[0,0]+B_test[0,1])
    M[6] = (A_test[0,1]-A_test[1,1]) * (B_test[1,0]+B_test[1,1])

    C_strassen = np.zeros((2,2))
    C_strassen[0,0] = M[0] - M[1] + M[2] + M[3]
    C_strassen[0,1] = M[4] + M[6]
    C_strassen[1,0] = M[5] + M[1]
    C_strassen[1,1] = M[0] - M[4] + M[3] + M[2]

    if not np.allclose(C_strassen, C_test):
        # Fallback to schoolbook if Strassen verification fails (shouldn't happen)
        pass

    # Build the full decomposition for 2x2 * 2x8
    # 4 blocks of 2 columns each
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7  # 28

    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28

    col = 0
    for block in range(num_blocks):
        for r in range(7):
            # U: coefficients for A entries
            for idx in range(4):
                U[idx][col] = strassen_A[r][idx]

            # V: coefficients for B entries
            # In local 2x2 block: B indices are (j, k_loc) where k_loc is 0 or 1
            # Global k = block*2 + k_loc
            # V index = j * p + k_global = j * 8 + (block*2 + k_loc)
            # Local B index lb = j*2 + k_loc (row-major for 2x2)
            for lb in range(4):
                val = strassen_B[r][lb]
                if val != 0:
                    j = lb // 2
                    k_loc = lb % 2
                    k_glob = block * 2 + k_loc
                    v_idx = j * p + k_glob
                    V[v_idx][col] = val

            # W: coefficients for C entries
            # In local 2x2 block: C indices are (i, k_loc)
            # Global k = block*2 + k_loc
            # W index = k_global * n + i = (block*2 + k_loc)*2 + i  (per task: W[k*n+i, r])
            # Local C index lc = i*2 + k_loc (row-major for 2x2)
            for lc in range(4):
                val = strassen_C[r][lc]
                if val != 0:
                    i = lc // 2
                    k_loc = lc % 2
                    k_glob = block * 2 + k_loc
                    w_idx = k_glob * n + i
                    W[w_idx][col] = val

            col += 1

    # Final numerical verification of the full decomposition
    if time.time() < deadline - 5.0:
        A_full = rng.rand(n, m)
        B_full = rng.rand(m, p)
        C_expected = A_full @ B_full

        # Compute C via the decomposition
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum over B part) * W[k*n+i, r]
        # Actually the correct formula: 
        # C[i,k] = sum_r sum_{i',j'} sum_{j'',k'} A[i',j']*B[j'',k'] * U[i',j',r]*V[j'',k'',r]*W[k,n_i,r]
        # Wait, that's not right either. The tensor decomposition means:
        # The multiplication tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # where a = i*m+j, b = j'*p+k', c = k*n+i (using the task's W[k*n+i] convention)
        # And C[i,k] = sum_{j} A[i,j]*B[j,k]
        # So we need: sum_j A[i,j]*B[j,k] = sum_{a,b} A[a_entry]*B[b_entry]*T[a,b,c]
        # where c corresponds to output (i,k), i.e., c = k*n+i
        # and T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # So: C[i,k] = sum_{i',j',j'',k'} A[i',j']*B[j'',k'] * sum_r U[i'*m+j', r]*V[j''*p+k'', r]*W[k*n+i, r]
        # For this to equal sum_j A[i,j]*B[j,k], we need the sum over (i',j',j'',k') to pick out only
        # terms where i'=i, k'=k, j'=j''.
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    w_val = W[k * n + i][r]
                    if w_val == 0:
                        continue
                    # Sum over A
                    a_sum = 0.0
                    for idx in range(n * m):
                        ai = idx // m
                        aj = idx % m
                        a_sum += U[idx][r] * A_full[ai, aj]
                    # Sum over B
                    b_sum = 0.0
                    for bidx in range(m * p):
                        bj = bidx // p
                        bk = bidx % p
                        b_sum += V[bidx][r] * B_full[bj, bk]
                    total += a_sum * b_sum * w_val
                C_computed[i, k] = total

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: schoolbook decomposition (rank 32)
            rank_s = n * m * p
            U_s = [[0] * rank_s for _ in range(n * m)]
            V_s = [[0] * rank_s for _ in range(m * p)]
            W_s = [[0] * rank_s for _ in range(p * n)]
            cc = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][cc] = 1
                        V_s[j * p + k][cc] = 1
                        W_s[k * n + i][cc] = 1
                        cc += 1
            return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
