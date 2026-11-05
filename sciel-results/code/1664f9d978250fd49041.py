"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110

    # Strategy:
    # A (4x5) = [A11 (4x4) | A12 (4x1)]
    # B (5x5) = [B11 (4x4); B12 (4x1)]  (top 4 rows, col 4) + B22 (1x1) at (4,4)
    # C (4x5) = [C11 (4x4) | C12 (4x1)]
    #
    # C11 = A11 * B11  -> Strassen 4x4x4 with Strassen 2x2x2 inside: rank 7*7=49
    # C12 = A11*B12 + A12*B22 -> schoolbook: 4*4*1 + 4*1*1 = 16+4 = 20
    # Total rank: 69

    rank_total = 69
    U = np.zeros((n * m, rank_total))  # (20, 69)
    V = np.zeros((m * p, rank_total))  # (25, 69)
    W = np.zeros((p * n, rank_total))  # (20, 69)

    col = 0

    # --- Part 1: Strassen 4x4x4 (rank 49) ---
    # A11 = A[0:4, 0:4], B11 = B[0:4, 0:4], C11 = C[0:4, 0:4]
    # Split into 2x2 blocks:
    # A11 = [[P, Q], [R, S]] where P=A[0:2,0:2], Q=A[0:2,2:4], R=A[2:4,0:2], S=A[2:4,2:4]
    # B11 = [[T, U], [V, Wb]] where T=B[0:2,0:2], U=B[0:2,2:4], V=B[2:4,0:2], Wb=B[2:4,2:4]
    # C11 = [[X, Y], [Z, Ab]] where X=C[0:2,0:2], Y=C[0:2,2:4], Z=C[2:4,0:2], Ab=C[2:4,2:4]
    #
    # Strassen:
    # M1 = (P+S)(T+Wb)
    # M2 = (R+S)T
    # M3 = P(U-Wb)
    # M4 = S(V-T)
    # M5 = (P+Q)Wb
    # M6 = (R-P)(T+U)
    # M7 = (Q-S)(V+Wb)
    #
    # X = M1 + M4 - M5 + M7
    # Y = M3 + M5
    # Z = M2 - M4 + M6
    # Ab = M1 - M2 + M3 - M6
    #
    # Each Mi is a 2x2x2 product. Use Strassen (rank 7) for each.
    # For a 2x2x2 Strassen with matrices X2(2x2) and Y2(2x2) producing Z2(2x2):
    # s1 = (X2[0,0]+X2[1,1])(Y2[0,0]+Y2[1,1])
    # s2 = (X2[1,0]+X2[1,1])Y2[0,0]
    # s3 = X2[0,0](Y2[0,1]-Y2[1,1])
    # s4 = X2[1,1](Y2[1,0]-Y2[0,0])
    # s5 = (X2[0,0]+X2[0,1])Y2[1,1]
    # s6 = (X2[1,0]-X2[0,0])(Y2[0,0]+Y2[0,1])
    # s7 = (X2[0,1]-X2[1,1])(Y2[1,0]+Y2[1,1])
    # Z2[0,0] = s1+s4-s5+s7
    # Z2[0,1] = s3+s5
    # Z2[1,0] = s2-s4+s6
    # Z2[1,1] = s1-s2+s3-s6
    #
    # Each scalar multiplication s_i = a*b contributes one rank-1 term.
    # For s_i = (sum of A entries with coeffs) * (sum of B entries with coeffs):
    # We need to expand this into rank-1 terms. But since we're building the tensor
    # decomposition, each "virtual multiplication" s_i = a_i * b_i where a_i is a
    # linear combination of A entries and b_i is a linear combination of B entries.
    #
    # Actually, for the tensor decomposition, each rank-1 term corresponds to:
    # U[row_a, col] = coeff_a, V[row_b, col] = coeff_b, W[row_c, col] = coeff_c
    # where the product U*V*W = 1 for the (i,j,k) entry.
    #
    # The way this works: for each scalar multiplication in the algorithm,
    # we have a linear form in A and a linear form in B. The product gives
    # a bilinear form that contributes to specific C entries.
    #
    # For s = (sum_{(i,j)} alpha_{ij} * A[i,j]) * (sum_{(j',k)} beta_{j'k} * B[j',k])
    # This contributes alpha_{ij} * beta_{j'k} * gamma to C[i',k'] for each (i',k').
    # But in the tensor decomposition, we can't have a single rank-1 term for
    # a product of sums. We need to expand.
    #
    # Actually, the correct way: each rank-1 term in the tensor decomposition
    # is of the form u ⊗ v ⊗ w where u is a vector in R^{nm}, v in R^{mp}, w in R^{pn}.
    # The tensor T = sum_c u_c ⊗ v_c ⊗ w_c.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the Strassen algorithm, each "multiplication" s = a * b where
    # a = sum_{(i,j)} alpha_{ij} A[i,j] and b = sum_{(j',k)} beta_{j'k} B[j',k]
    # The contribution to C is: s contributes to C[i',k'] with coefficient gamma_{i'k'}
    # So the bilinear form is: (sum alpha_{ij} A[i,j]) * (sum beta_{j'k} B[j',k]) * gamma_{i'k'}
    # for each (i',k').
    #
    # To express this as rank-1 terms, we need:
    # For each (i,j) with alpha_{ij} != 0 and each (j',k) with beta_{j'k} != 0:
    #   But that would be a product of many terms, not rank-1.
    #
    # The correct approach: the tensor decomposition rank-1 terms correspond to
    # individual scalar multiplications in the algorithm. If the algorithm has
    # R scalar multiplications, the rank is R.
    #
    # In Strassen for 2x2x2, there are 7 scalar multiplications. Each one is:
    # s_m = a_m * b_m where a_m is a specific entry (or sum) of the 2x2 matrix A
    # and b_m is a specific entry (or sum) of the 2x2 matrix B.
    #
    # Wait, no. In Strassen, s1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]).
    # This is ONE scalar multiplication of two numbers. In the tensor decomposition,
    # this corresponds to... hmm.
    #
    # Actually, I think the correct interpretation is:
    # The matrix multiplication tensor T_{ijk} = delta_{ij'}delta_{jk'} for the
    # appropriate indices. For a 2x2x2 multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 if the indices are consistent, 0 otherwise.
    # Specifically, T[i,j,k] = 1 for all i,j,k in {0,1} (it's the all-ones tensor
    # for the 2x2x2 case? No...)
    #
    # Let me be more careful. The 2x2x2 matrix multiplication tensor is:
    # T[i,j,k] where i,k are output indices, j is the contraction index.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones 2x2x2 tensor).
    # Wait no, that's not right either.
    #
    # The tensor is defined as T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * T_{ijk}
    # where T_{ijk} is the coefficient. For matrix multiplication C = AB:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{ijk} = 1 for all valid i,j,k.
    #
    # For 2x2x2: T_{ijk} = 1 for all i,j,k in {0,1}. That's 8 entries, all 1.
    # The rank of this tensor is 7 (Strassen).
    #
    # Strassen gives 7 rank-1 decompositions:
    # s1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # s2 = (A10+A11)B00 -> contributes to C10, C11
    # s3 = A00(B01-B11) -> contributes to C01, C11
    # s4 = A11(B10-B00) -> contributes to C10, C00
    # s5 = (A00+A01)B11 -> contributes to C01, C00
    # s6 = (A10-A00)(B00+B01) -> contributes to C10, C01
    # s7 = (A01-A11)(B10+B11) -> contributes to C11, C00
    #
    # Each s_m is a product of two linear forms. In the tensor decomposition:
    # s_m = (sum_{(i,j)} alpha^m_{ij} A_{ij}) * (sum_{(j,k)} beta^m_{jk} B_{jk})
    # This contributes to C_{ik} with coefficient gamma^m_{ik}:
    # T_{ijk} += alpha^m_{ij} * beta^m_{jk} * gamma^m_{ik}
    #
    # But for this to be a valid rank-1 term, we need:
    # U[i*m+j, m] = alpha^m_{ij}
    # V[j*p+k, m] = beta^m_{jk}
    # W[k*n+i, m] = gamma^m_{ik}
    # And the product U*V*W summed over m should give T_{ijk}.
    #
    # For a single rank-1 term m:
    # sum_m U[i*m+j, m] * V[j*p+k, m] * W[k*n+i, m] = T_{ijk}
    #
    # For Strassen 2x2x2, the 7 terms are:
    # Term 1: U has alpha^1 at (0,0) and (1,1), V has beta^1 at (0,0) and (1,1),
    #          W has gamma^1 at (0,0) and (1,1)
    # But wait, U is indexed by (i,j) pair, so U[0*m+0] = 1, U[1*m+1] = 1
    # V is indexed by (j,k) pair, so V[0*p+0] = 1, V[1*p+1] = 1
    # W is indexed by (k,i) pair, so W[0*n+0] = 1, W[1*n+1] = 1
    #
    # Then for T[i,j,k]:
    # sum_m U[i*m+j, m] * V[j*p+k, m] * W[k*n+i, m]
    # For term 1: U[i*m+j, 1] * V[j*p+k, 1] * W[k*n+i, 1]
    # This is nonzero only if (i,j) is (0,0) or (1,1), AND (j,k) is (0,0) or (1,1),
    # AND (k,i) is (0,0) or (1,1).
    #
    # This is getting complicated. Let me just enumerate all the terms explicitly
    # for the full 4x5x5 decomposition.

    # Let me take a cleaner approach: directly construct the rank-1 terms
    # for the Strassen-based decomposition.

    # For the 4x4x4 Strassen (with Strassen 2x2x2 inside):
    # We have 7 outer Strassen multiplications, each being a 2x2x2 product.
    # Each 2x2x2 product uses 7 inner Strassen multiplications.
    # Total: 49 scalar multiplications.

    # Each scalar multiplication s = a * b where:
    # a = sum of some A entries with coefficients
    # b = sum of some B entries with coefficients
    # s contributes to some C entries with coefficients.

    # For the tensor decomposition, each such scalar multiplication gives
    # multiple rank-1 terms (one for each pair of (A entry, B entry) that
    # appear in the sums, combined with the appropriate C coefficient).

    # Actually, I realize the issue: a product of two linear forms is NOT
    # a single rank-1 term in the tensor. It's a sum of rank-1 terms.
    # (sum_i a_i x_i)(sum_j b_j y_j) = sum_{i,j} a_i b_j x_i y_j
    # Each (i,j) pair gives one rank-1 term.

    # So if s = (A00 + A11) * (B00 + B11), this expands to:
    # A00*B00 + A00*B11 + A11*B00 + A11*B11
    # That's 4 rank-1 terms, not 1!

    # This means Strassen doesn't directly give a rank-7 decomposition in
    # the tensor sense. The rank-7 result is about the number of scalar
    # multiplications in the algorithm, but each scalar multiplication of
    # linear forms expands to multiple rank-1 tensor terms.

    # Hmm, but the known result is that the 2x2x2 matrix multiplication
    # tensor has rank 7. So there MUST be a way to write it as 7 rank-1 terms.

    # Let me look up the explicit Strassen decomposition of the 2x2x2 tensor.
    #
    # The 2x2x2 tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We want T = sum_{m=1}^{7} u_m ⊗ v_m ⊗ w_m
    #
    # Strassen's decomposition:
    # m=1: u = e_{00}+e_{11}, v = e_{00}+e_{11}, w = e_{00}+e_{11}
    #   (where e_{ij} is the basis vector for position (i,j))
    # m=2: u = e_{10}+e_{11}, v = e_{00}, w = e_{10}+e_{11}
    # m=3: u = e_{00}, v = e_{01}-e_{11}, w = e_{01}+e_{11}
    # m=4: u = e_{11}, v = e_{10}-e_{00}, w = e_{10}+e_{00}
    # m=5: u = e_{00}+e_{01}, v = e_{11}, w = e_{01}+e_{00}
    # m=6: u = e_{10}-e_{00}, v = e_{00}+e_{01}, w = e_{10}+e_{01}
    # m=7: u = e_{01}-e_{11}, v = e_{10}+e_{11}, w = e_{11}+e_{00}

    # Let me verify term 1:
    # (e00+e11) ⊗ (e00+e11) ⊗ (e00+e11)
    # = e00⊗e00⊗e00 + e00⊗e00⊗e11 + e00⊗e11⊗e00 + e00⊗e11⊗e11
    #   + e11⊗e00⊗e00 + e11⊗e00⊗e11 + e11⊗e11⊗e00 + e11⊗e11⊗e11
    #
    # This gives 1 for (i,j,k) where i is in {0,1}, j in {0,1}, k in {0,1}
    # AND the specific combinations match. Actually, this gives 1 for:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # Wait, that's all 8 entries! That can't be right for just one term.

    # I think I'm confusing the tensor structure. Let me reconsider.
    #
    # For matrix multiplication C = AB where A is 2x2 and B is 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # The tensor T is indexed by (i, j, k) where:
    # i = row of A (and C), j = col of A / row of B, k = col of B (and C)
    # T[i,j,k] = 1 if this term appears in C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # In the factorization T = sum_m U_m ⊗ V_m ⊗ W_m:
    # U_m is a vector in R^{2*2} = R^4 (indexed by (i,j))
    # V_m is a vector in R^{2*2} = R^4 (indexed by (j,k))
    # W_m is a vector in R^{2*2} = R^4 (indexed by (k,i))
    #
    # T[i,j,k] = sum_m U_m[i,j] * V_m[j,k] * W_m[k,i]
    #
    # For Strassen, the 7 terms are:
    # m=1: U = [1,0,0,1] (A00,A01,A10,A11), V = [1,0,0,1] (B00,B01,B10,B11), W = [1,0,0,1] (C00,C01,C10,C11 indexed by (k,i))
    #   Wait, W is indexed by (k,i), so W[k,i].
    #   W = [1,0,0,1] means W[0,0]=1, W[1,1]=1
    #
    # Let me verify: for (i,j,k) = (0,0,0):
    # U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # For (i,j,k) = (0,0,1):
    # U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # But T[0,0,1] should be 1 (it's A[0,0]*B[0,1] in C[0,1]).
    # So term 1 alone doesn't give the right answer. The sum of all 7 terms should.

    # OK so I need to carefully construct all 7 terms and verify they sum to the
    # all-ones tensor. Let me just use the known Strassen formulas and build
    # the U, V, W vectors for each term.

    # For 2x2x2 Strassen, the 7 multiplications are:
    # s1 = (A00+A11)(B00+B11)
    # s2 = (A10+A11)(B00)
    # s3 = (A00)(B01-B11)
    # s4 = (A11)(B10-B00)
    # s5 = (A00+A01)(B11)
    # s6 = (A10-A00)(B00+B01)
    # s7 = (A01-A11)(B10+B11)
    #
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 - s4 + s6
    # C11 = s1 - s2 + s3 - s6
    #
    # Each s_m = a_m * b_m where a_m is a linear form in A and b_m in B.
    # In the tensor decomposition, s_m contributes to C[i,k] with coefficient
    # gamma^m_{ik}. The rank-1 terms are:
    # For each (i,j) with a^m_{ij} != 0 and each (j',k) with b^m_{j'k} != 0:
    #   But we need j = j' for the contraction!
    #
    # Actually, the correct way: s_m = (sum_{i,j} a^m_{ij} A_{ij})(sum_{j,k} b^m_{jk} B_{jk})
    # This equals sum_{i,j,j',k} a^m_{ij} b^m_{j'k} A_{ij} B_{j'k}
    # But for matrix multiplication, we need j = j', so:
    # = sum_{i,j,k} a^m_{ij} b^m_{jk} A_{ij} B_{jk}
    #
    # This contributes to C[i,k] with coefficient a^m_{ij} b^m_{jk} gamma^m_{ik}
    # for each (i,j,k). But the total contribution to T[i,j,k] from term m is:
    # a^m_{ij} * b^m_{jk} * gamma^m_{ik}
    #
    # So for the tensor decomposition:
    # U_m[i*m+j] = a^m_{ij}
    # V_m[j*p+k] = b^m_{jk}
    # W_m[k*n+i] = gamma^m_{ik}
    #
    # And T[i,j,k] = sum_m a^m_{ij} * b^m_{jk} * gamma^m_{ik}

    # Great! So each Strassen multiplication s_m gives ONE rank-1 term in the
    # tensor decomposition, where:
    # U_m[i,j] = coefficient of A[i,j] in a_m
    # V_m[j,k] = coefficient of B[j,k] in b_m
    # W_m[k,i] = coefficient of C[k,i]... wait, W is indexed by (k,i) where
    #           k is the col of C and i is the row of C.
    #           gamma^m_{ik} is the coefficient of C[i,k].
    #           So W_m[k*n+i] = gamma^m_{ik}

    # Now let me construct the full 4x4x4 Strassen decomposition.
    #
    # The 4x4x4 Strassen has 7 multiplications, each of which is a 2x2x2
    # Strassen with 7 sub-multiplications. Total: 49 rank-1 terms.
    #
    # For each of the 7 outer multiplications, I need to:
    # 1. Determine the 2x2 linear forms in A11 and B11
    # 2. Apply the 2x2x2 Strassen to get 7 sub-terms
    # 3. Map each sub-term to the full 4x4x4 indices

    # Let me define the 2x2x2 Strassen terms as a list of (a_coeffs, b_coeffs, c_coeffs)
    # where a_coeffs is a 2x2 matrix of coefficients for the A sub-block,
    # b_coeffs is a 2x2 for B sub-block, c_coeffs is a 2x2 for C sub-block.

    strassen_2x2 = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # s1 = (A00+A11)(B00+B11), C00 += 1, C11 += 1
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # s2 = (A10+A11)(B00), C10 += 1, C11 += -1
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
        # s3 = (A00)(B01-B11), C01 += 1, C11 += 1
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),
        # s4 = (A11)(B10-B00), C10 += -1, C00 += 1
        ([[0,0],[0,1]], [[0,0],[1,-1]], [[1,0],[-1,0]]),
        # s5 = (A00+A01)(B11), C01 += 1, C00 += -1
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
        # s6 = (A10-A00)(B00+B01), C10 += 1, C01 += 1
        ([[1,0],[1,0]], [[1,1],[0,0]], [[0,1],[1,1]]),
        # s7 = (A01-A11)(B10+B11), C00 += 1, C11 += 1
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,1]]),
    ]

    # Verify: for each (i,j,k) in {0,1}^3, sum_m a_m[i,j]*b_m[j,k]*c_m[k,i] should be 1.
    # Let me verify (0,0,0):
    # m1: 1*1*1 = 1
    # m2: 0*1*0 = 0
    # m3: 1*0*0 = 0
    # m4: 0*0*1 = 0
    # m5: 1*0*(-1) = 0
    # m6: 1*1*0 = 0
    # m7: 0*0*1 = 0
    # Sum = 1 ✓

    # Verify (0,0,1):
    # m1: 1*0*0 = 0
    # m2: 0*0*0 = 0
    # m3: 1*1*1 = 1
    # m4: 0*0*0 = 0
    # m5: 1*0*0 = 0
    # m6: 1*1*1 = 1
    # m7: 0*0*0 = 0
    # Sum = 2? That's wrong! Should be 1.

    # Let me recheck. T[0,0,1] = 1 (A[0,0]*B[0,1] in C[0,1]).
    # m3: a[0,0]=1, b[0,1]=1, c[1,0]=0 (c is indexed by [k,i], so c[1,0] means k=1,i=0)
    # Wait, I need to be more careful with the indexing.
    #
    # c_coeffs is a 2x2 matrix where c_coeffs[i][k] is the coefficient for C[i,k].
    # But W is indexed by (k,i), so W[k*n+i] = c_coeffs[i][k].
    #
    # For T[i,j,k] = sum_m a_m[i,j] * b_m[j,k] * c_m[i,k]
    #
    # Let me re-verify (0,0,1):
    # m1: a[0,0]=1, b[0,1]=0, c[0,1]=0 -> 0
    # m2: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # m3: a[0,0]=1, b[0,1]=1, c[0,1]=1 -> 1
    # m4: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # m5: a[0,0]=1, b[0,1]=0, c[0,1]=1 -> 0
    # m6: a[0,0]=1, b[0,1]=1, c[0,1]=1 -> 1
    # m7: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # Sum = 2. Still wrong!

    # I think my c_coeffs are wrong. Let me re-derive.
    #
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 - s4 + s6
    # C11 = s1 - s2 + s3 - s6
    #
    # So for each s_m, the c_coeffs should be:
    # s1: C00 += 1, C11 += 1 -> c = [[1,0],[0,1]]
    # s2: C10 += 1, C11 += -1 -> c = [[0,0],[1,-1]]
    # s3: C01 += 1, C11 += 1 -> c = [[0,1],[0,1]]
    # s4: C10 += -1, C00 += 1 -> c = [[1,0],[-1,0]]
    # s5: C01 += 1, C00 += -1 -> c = [[-1,1],[0,0]]
    # s6: C10 += 1, C01 += 1 -> c = [[0,1],[1,0]]
    # s7: C00 += 1, C11 += 1 -> c = [[1,0],[0,1]]

    # Let me re-verify (0,0,1) with corrected c:
    # m1: a[0,0]=1, b[0,1]=0, c[0,1]=0 -> 0
    # m2: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # m3: a[0,0]=1, b[0,1]=1, c[0,1]=1 -> 1
    # m4: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # m5: a[0,0]=1, b[0,1]=0, c[0,1]=1 -> 0
    # m6: a[0,0]=1, b[0,1]=1, c[0,1]=1 -> 1
    # m7: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # Sum = 2. Still 2!

    # Hmm, let me check the Strassen formulas more carefully.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    #
    # Using 1-indexed notation where A11 is top-left, etc.
    # In 0-indexed: A[0,0], A[0,1], A[1,0], A[1,1]
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[0,1]-B[1,1])
    # M4 = A[1,1](B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 - M4 + M6
    # C[1,1] = M1 - M2 + M3 - M6
    #
    # Let me verify C[0,1] = M3 + M5:
    # M3 = A[0,0]*(B[0,1]-B[1,1]) = A[0,0]*B[0,1] - A[0,0]*B[1,1]
    # M5 = (A[0,0]+A[0,1])*B[1,1] = A[0,0]*B[1,1] + A[0,1]*B[1,1]
    # M3 + M5 = A[0,0]*B[0,1] + A[0,1]*B[1,1] = C[0,1] ✓
    #
    # Now for the tensor: T[i,j,k] should be 1 for all i,j,k.
    # Let me check T[0,0,1] (i=0, j=0, k=1):
    # This corresponds to A[0,0]*B[0,1] which appears in C[0,1].
    #
    # From M3: a=[1,0;0,0], b=[0,1;0,-1], c for C[0,1] is +1, for C[1,1] is +1
    #   Contribution to T[0,0,1]: a[0,0]*b[0,1]*c[0,1] = 1*1*1 = 1
    # From M5: a=[1,1;0,0], b=[0,0;0,1], c for C[0,1] is +1, for C[0,0] is -1
    #   Contribution to T[0,0,1]: a[0,0]*b[0,1]*c[0,1] = 1*0*1 = 0
    # From M6: a=[1,0;1,0], b=[1,1;0,0], c for C[1,0] is +1, for C[0,1] is +1
    #   Contribution to T[0,0,1]: a[0,0]*b[0,1]*c[0,1] = 1*1*1 = 1
    #
    # Total: 1 + 0 + 1 = 2. But should be 1!
    #
    # I think the issue is that M6's c coefficients are wrong.
    # C[1,0] = M2 - M4 + M6, so M6 contributes +1 to C[1,0].
    # C[0,1] = M3 + M5, so M6 contributes 0 to C[0,1].
    # So c for M6 should be [[0,0],[1,0]] not [[0,1],[1,0]].

    # Let me redo all c coefficients:
    # M1: C[0,0]+=1, C[1,1]+=1 -> c = [[1,0],[0,1]]
    # M2: C[1,0]+=1, C[1,1]+=-1 -> c = [[0,0],[1,-1]]
    # M3: C[0,1]+=1, C[1,1]+=1 -> c = [[0,1],[0,1]]
    # M4: C[0,0]+=1, C[1,0]+=-1 -> c = [[1,0],[-1,0]]
    # M5: C[0,0]+=-1, C[0,1]+=1 -> c = [[-1,1],[0,0]]
    # M6: C[1,0]+=1 -> c = [[0,0],[1,0]]
    # M7: C[0,0]+=1, C[1,1]+=1 -> c = [[1,0],[0,1]]

    # Re-verify T[0,0,1]:
    # M1: a[0,0]=1, b[0,1]=0, c[0,1]=0 -> 0
    # M2: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # M3: a[0,0]=1, b[0,1]=1, c[0,1]=1 -> 1
    # M4: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # M5: a[0,0]=1, b[0,1]=0, c[0,1]=1 -> 0
    # M6: a[0,0]=1, b[0,1]=1, c[0,1]=0 -> 0
    # M7: a[0,0]=0, b[0,1]=0, c[0,1]=0 -> 0
    # Sum = 1 ✓

    # Let me verify T[0,1,0] (i=0, j=1, k=0):
    # A[0,1]*B[1,0] in C[0,0].
    # M1: a[0,1]=0, b[1,0]=0, c[0,0]=1 -> 0
    # M2: a[0,1]=0, b[1,0]=0, c[0,0]=0 -> 0
    # M3: a[0,1]=0, b[1,0]=0, c[0,0]=0 -> 0
    # M4: a[0,1]=0, b[1,0]=1, c[0,0]=1 -> 0
    # M5: a[0,1]=1, b[1,0]=0, c[0,0]=-1 -> 0
    # M6: a[0,1]=0, b[1,0]=0, c[0,0]=0 -> 0
    # M7: a[0,1]=1, b[1,0]=1, c[0,0]=1 -> 1
    # Sum = 1 ✓

    # Let me verify T[1,0,0] (i=1, j=0, k=0):
    # A[1,0]*B[0,0] in C[1,0].
    # M1: a[1,0]=0, b[0,0]=1, c[1,0]=0 -> 0
    # M2: a[1,0]=1, b[0,0]=1, c[1,0]=1 -> 1
    # M3: a[1,0]=0, b[0,0]=0, c[1,0]=0 -> 0
    # M4: a[1,0]=0, b[0,0]=-1, c[1,0]=-1 -> 0
    # M5: a[1,0]=0, b[0,0]=0, c[1,0]=0 -> 0
    # M6: a[1,0]=1, b[0,0]=1, c[1,0]=1 -> 1
    # M7: a[1,0]=0, b[0,0]=0, c[1,0]=0 -> 0
    # Sum = 2? Should be 1!

    # Hmm. Let me recheck M2.
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # C[1,0] = M2 - M4 + M6
    # C[1,1] = M1 - M2 + M3 - M6
    # So M2 contributes +1 to C[1,0] and -1 to C[1,1].
    # c for M2 = [[0,0],[1,-1]] ✓
    #
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # C[1,0] = M2 - M4 + M6, so M6 contributes +1 to C[1,0].
    # c for M6 = [[0,0],[1,0]] ✓
    #
    # T[1,0,0]:
    # M2: a[1,0]=1, b[0,0]=1, c[1,0]=1 -> 1
    # M6: a[1,0]=1, b[0,0]=1, c[1,0]=1 -> 1
    # Sum = 2.
    #
    # But A[1,0]*B[0,0] should appear exactly once in C[1,0].
    # The issue is that both M2 and M6 contain the term A[1,0]*B[0,0],
    # and both contribute to C[1,0]. But in the actual computation:
    # M2 = (A[1,0]+A[1,1])*B[0,0] = A[1,0]*B[0,0] + A[1,1]*B[0,0]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) = A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    #
    # C[1,0] = M2 - M4 + M6
    # = (A[1,0]*B[0,0] + A[1,1]*B[0,0]) - (A[1,1]*B[1,0] - A[1,1]*B[0,0]) + (A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1])
    # = A[1,0]*B[0,0] + A[1,1]*B[0,0] - A[1,1]*B[1,0] + A[1,1]*B[0,0] + A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    # = 2*A[1,0]*B[0,0] + 2*A[1,1]*B[0,0] - A[1,1]*B[1,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    #
    # But C[1,0] should be A[1,0]*B[0,0] + A[1,1]*B[1,0].
    # This doesn't match! I must have the Strassen formulas wrong.

    # Let me look up the correct Strassen formulas.
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    #
    # Let me verify C21 = M2 - M4 + M6:
    # M2 = (A21+A22)B11 = A21*B11 + A22*B11
    # M4 = A22(B21-B11) = A22*B21 - A22*B11
    # M6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M2 - M4 + M6 = A21*B11 + A22*B11 - A22*B21 + A22*B11 + A21*B11 + A21*B12 - A11*B11 - A11*B12
    # = 2*A21*B11 + 2*A22*B11 - A22*B21 + A21*B12 - A11*B11 - A11*B12
    #
    # C21 should be A21*B11 + A22*B21. This doesn't match!

    # I think I have the formula wrong. Let me look it up properly.
    #
    # Correct Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    #
    # Hmm, let me try a different source. Wikipedia says:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    #
    # Let me verify with a specific example.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19,22],[43,50]]
    #
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 7*5 = 35
    # M3 = 1*(6-8) = -2
    # M4 = 4*(7-5) = 8
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    #
    # C11 = 65 + 8 - 24 + (-30) = 19 ✓
    # C12 = -2 + 24 = 22 ✓
    # C21 = 35 - 8 + 22 = 49 ✗ (should be 43)
    # C22 = 65 - 35 + (-2) - 22 = 6 ✗ (should be 50)

    # The formulas are wrong! Let me find the correct ones.
    #
    # After more research, the correct Strassen formulas are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    #
    # Wait, I just verified this and it doesn't work. Let me try yet another variant.
    #
    # Actually, I think the issue is with the indexing. Let me use:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    #
    # C11 = ae+bg:
    # M1+M4-M5+M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh
    # = ae + bg ✓
    #
    # C12 = af+bh:
    # M3+M5 = (af-ah) + (ah+bh) = af+bh ✓
    #
    # C21 = ce+dg:
    # M2-M4+M6 = (ce+de) - (dg-de) + (ce+cf-ae-af)
    # = ce+de-dg+de+ce+cf-ae-af
    # = 2ce+2de-dg+cf-ae-af
    # This is NOT ce+dg!

    # I'm clearly getting the formula wrong. Let me just use a known correct
    # Strassen implementation and derive the tensor from it.
    #
    # Actually, let me just give up on the analytical approach and use a
    # simpler strategy: implement schoolbook for the 4x4x4 part (rank 64)
    # plus schoolbook for the remaining parts (rank 20), total rank 84.
    # This is still better than 100.
    #
    # Or better: let me use a verified Strassen decomposition.
    #
    # After careful derivation, the correct Strassen tensor decomposition
    # for 2x2x2 is:
    #
    # The 7 rank-1 terms (U, V, W) where U is in R^4 (A indices (i,j)),
    # V is in R^4 (B indices (j,k)), W is in R^4 (C indices (k,i)):
    #
    # I'll just use the schoolbook approach for safety. Rank 84 total.

    # Schoolbook for 4x4x4: rank 64
    # Schoolbook for 4x4x1: rank 16
    # Schoolbook for 4x1x1: rank 4
    # Total: 84

    rank_total = 84
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    col = 0

    # Part 1: Schoolbook 4x4x4 (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j, col] = 1
                V[j * 5 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1

    # Part 2: Schoolbook 4x4x1 (A[0:4,0:4] * B[0:4,4] -> C[0:4,4])
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    # Part 3: Schoolbook 4x1x1 (A[0:4,4] * B[4,4] -> C[0:4,4])
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1

    assert col == rank_total, f"col={col}, rank_total={rank_total}"

    # Verify
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
