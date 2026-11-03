"""Matrix multiplication tensor decomposition for 3x4x8."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x8 matrix.

    Strategy: Split the 3 rows into two groups:
      - Group 1: rows 0, 1 (2 rows)
      - Group 2: row 2 (1 row)

    For Group 1 (2x4x8): Use Strassen-like recursive decomposition.
      Split the 4 columns of A (and 4 rows of B) into two groups of 2.
      Split the 8 columns of B (and 8 columns of C) into four groups of 2.
      This gives 2 * 4 = 8 sub-problems of type 2x2x2.
      Each 2x2x2 uses Strassen's algorithm with rank 7.
      Total rank for Group 1: 8 * 7 = 56.

    For Group 2 (1x4x8): This is a rank-1 tensor in the i dimension,
      so we need rank = 4 * 8 = 32... wait, no.
      Actually for 1x4x8, the tensor is T[k*n+i, j*p+k, i*m+j] where i=2 only.
      W index: k*n + i = 8k + 2.
      U index: i*m + j = 8 + j for j=0..3.
      V index: j*p + k = 8j + k.
      The tensor is sum over j,k of U[8+j, c] * V[8j+k, c] * W[8k+2, c] = A[2,j]*B[j,k].
      This is just the outer product structure, rank = 4*8 = 32? No, it's just a standard
      matrix-vector product. The rank is 4*8 = 32 for schoolbook. But we can do better.
      Actually, this is just a linear map from (4*8)-dim space to... no.
      The tensor for 1x4x8 is: T_{j,k,c} where c indexes the "rank" dimension.
      We need sum_c U[j,c] * V[j,k,c]... wait let me re-read.

      U has shape (n*m, rank) = (12, rank)
      V has shape (m*p, rank) = (32, rank)
      W has shape (p*n, rank) = (24, rank)

      For i=2 (the third row of A):
      U[2*m+j, c] = U[8+j, c]
      V[j*p+k, c] = V[8j+k, c]
      W[k*n+2, c] = W[8k+2, c]

      The product is sum_c U[8+j,c] * V[8j+k,c] * W[8k+2,c] = C[2,k] = sum_j A[2,j]*B[j,k]

      So we need to represent the bilinear map (A[2,0..3], B[0..3,0..7]) -> C[2,0..7].
      This is equivalent to multiplying a 1x4 vector by a 4x8 matrix.
      The rank of this tensor is 4*8 = 32 for schoolbook. But we can use the same Strassen
      idea? No, because one dimension is 1.

      Actually, let me reconsider. Can we combine the groups?

      Alternative: Treat the whole 3x4x8 problem more cleverly.

      Let me think about known results. The rank of 2x2x2 is 7 (Strassen).
      The rank of 2x2xn is at most 7*ceil(n/2) for even n? Not exactly.

      For 2x4x8: split into 2x2x2 blocks. 4/2 * 8/2 = 2 * 4 = 8 blocks. Rank = 8*7 = 56.
      For 1x4x8: rank = 32.
      Total: 56 + 32 = 88.

      But we can do better for 1x4x8? The 1x4x8 tensor:
      T_{j,k} = A[2,j] * B[j,k]
      This is a bilinear form. The rank is the smallest r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
      where u_c is 4-dim, v_c is 32-dim, w_c is 8-dim.
      Actually u_c[j] = U[8+j,c], v_c[8j+k] = V[8j+k,c], w_c[k] = W[8k+2,c].
      We need sum_c u_c[j] * v_c[8j+k] * w_c[k] = A[2,j]*B[j,k].

      This is equivalent to: for each j, sum_c u_c[j]*w_c[k] * v_c[8j+k]... no, v_c depends on both j and k.

      Let me re-index. V has index (j,k) for j=0..3, k=0..7. So V[j,k,c] = V[8j+k,c].
      We need sum_c U[8+j,c] * V[8j+k,c] * W[8k+2,c] = A[2,j] * B[j,k].

      This is a 3-way tensor of size 4 x 32 x 8. The rank is at most 4*32*8 but we can do better.
      Actually, this is the same as the standard matrix multiplication rank problem for 1x4x8.
      The rank of 1xn x nxm is nm for schoolbook, but can it be lower?
      For 1x2x2: rank is 4 (schoolbook) or can it be 3? 
      1x2x2: T_{j,k} = A[j]*B[j,k]. j=0,1; k=0,1.
      This is a 2x2 tensor (in the bilinear sense: u is 2-dim, v is 4-dim, w is 2-dim).
      Rank 4 schoolbook. Can we do rank 3?
      sum_c u_c[j] * v_c[j,k] * w_c[k] = A[j]*B[j,k].
      This is equivalent to: for each (j,k), the coefficient of A[j]*B[j,k] must be 1, and all other cross terms 0.
      This is 4 independent monomials, so rank >= 4. So rank is exactly 4 for 1x2x2.
      Similarly for 1x4x8, rank = 4*8 = 32.

      So total rank = 56 + 32 = 88.

      Can we do better by not separating the groups? Maybe there's a clever decomposition
      that mixes the two row groups.

      Let's try a different approach: split the 8 columns into four groups of 2.
      For each group of 2 columns (k=2b, 2b+1 for b=0,1,2,3):
      We need to compute C[i, 2b:2b+2] = A[:, :] * B[:, 2b:2b+2] for i=0,1,2.
      This is a 3x4x2 sub-problem.
      For 3x4x2: split rows into (0,1) and (2).
      - Rows 0,1: 2x4x2. Split 4 columns into two groups of 2: 2x2x2 blocks. 2 blocks of 2x2x2, each rank 7. Total 14.
      - Row 2: 1x4x2. Rank = 4*2 = 8.
      Total per block: 14 + 8 = 22.
      Total for 4 blocks: 4 * 22 = 88. Same.

      Alternatively, for each block of 2 columns, can we share structure between the row groups?
      Probably not easily.

      Let me try yet another approach. What if we use a 2x2x8 decomposition with Strassen
      recursively on the 8 dimension?

      2x2x8: Split the 8 into four 2's. Each 2x2x2 has rank 7. Total 4*7 = 28.
      Then for the full 3x4x8:
      - Rows 0,1: 2x4x8. Split 4 into two 2's: two 2x2x8 blocks. Each 2x2x8 has rank 28. Total 56.
      - Row 2: 1x4x8. Rank 32.
      Total: 88.

      I think 88 is a good target. Let me implement it.

      Actually, wait. Can we reduce the 1x4x8 rank below 32? I argued no above. Let me double-check.
      The tensor is T_{j,k} = A_j * B_{j,k} for j=0..3, k=0..7.
      We want to write T = sum_c u_c ⊗ v_c ⊗ w_c where u_c ∈ R^4, v_c ∈ R^{4*8}, w_c ∈ R^8.
      The (j, 8j+k, k) component... this is getting confusing. Let me use the standard rank definition.
      The rank of the 1x4x8 matrix multiplication tensor is known to be 32. (It's the same as the
      rank of the 4x8 matrix multiplication when one side is a vector, which is 4*8.)

      So 88 is the target. Let me code it up.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88

    # Strassen 2x2x2 decomposition
    # For a 2x2x2 block with A indices (i1,i2), B indices (j1,j2), C indices (k1,k2):
    # M1 = A[i1,j1]*(B[j1,k1]+B[j2,k2])
    # M2 = A[i1,j2]*(B[j1,k1]+B[j1,k2])
    # M3 = A[i2,j1]*(B[j2,k1]+B[j2,k2])
    # M4 = A[i2,j2]*(B[j2,k1]+B[j2,k2])  # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For our case, the "A" is 2x2, "B" is 2x2, "C" is 2x2.
    # A indices: (i, j) where i is row in {0,1} (local), j is col in {0,1} (local)
    # B indices: (j, k) where j is row in {0,1} (local), k is col in {0,1} (local)
    # C indices: (i, k) where i is row in {0,1} (local), k is col in {0,1} (local)

    def add_strassen_block(i_base, j_base, k_base, r_offset):
        """
        Add a Strassen 2x2x2 block.
        i_base: base row index in A (and C)
        j_base: base col index in A (and row index in B)
        k_base: base col index in B (and C)
        r_offset: starting rank index
        """
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i_base + 0, r_offset + 0] += 1
        U[i_base + 1, r_offset + 0] += 1
        V[j_base + 0, r_offset + 0] += 1
        V[j_base + 1, r_offset + 0] += 1
        W[k_base + 0, r_offset + 0] += 1
        W[k_base + 1, r_offset + 0] += 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i_base + 1, r_offset + 1] += 1  # A[i2,j1]
        U[i_base + 1, r_offset + 1] += 1  # A[i2,j2] -> wait, this is wrong.
        # Let me re-do this carefully.
        # A has indices (i,j). i in {0,1} local, j in {0,1} local.
        # A[i1,j1] = A[i_base, j_base]
        # A[i1,j2] = A[i_base, j_base+1]
        # A[i2,j1] = A[i_base+1, j_base]
        # A[i2,j2] = A[i_base+1, j_base+1]
        # B has indices (j,k). j in {0,1} local, k in {0,1} local.
        # B[j1,k1] = B[j_base, k_base]
        # B[j1,k2] = B[j_base, k_base+1]
        # B[j2,k1] = B[j_base+1, k_base]
        # B[j2,k2] = B[j_base+1, k_base+1]
        # C has indices (i,k). i in {0,1} local, k in {0,1} local.
        # C[i1,k1] = C[i_base, k_base]
        # C[i1,k2] = C[i_base, k_base+1]
        # C[i2,k1] = C[i_base+1, k_base]
        # C[i2,k2] = C[i_base+1, k_base+1]

        # U index: i*m + j. So for A[i_base + di, j_base + dj], U index = (i_base+di)*m + (j_base+dj)
        # V index: j*p + k. So for B[j_base + dj, k_base + dk], V index = (j_base+dj)*p + (k_base+dk)
        # W index: k*n + i. So for C[k_base + dk, i_base + di], W index = (k_base+dk)*n + (i_base+di)

        # Let me redo the whole function.
        # First, reset what I did above (this is a draft, I'll rewrite cleanly).
        pass

    # Let me rewrite add_strassen_block cleanly.
    def add_strassen_block(i_base, j_base, k_base, r_offset):
        """
        Add a Strassen 2x2x2 block.
        i_base: base row index in A (and C), 0-indexed
        j_base: base col index in A (and row index in B), 0-indexed
        k_base: base col index in B (and C), 0-indexed
        r_offset: starting rank index
        """
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # A[i1,j1] -> U[(i_base)*m + j_base]
        # A[i2,j2] -> U[(i_base+1)*m + (j_base+1)]
        # B[j1,k1] -> V[(j_base)*p + k_base]
        # B[j2,k2] -> V[(j_base+1)*p + (k_base+1)]
        # C[i1,k1] -> W[(k_base)*n + i_base]
        # C[i2,k2] -> W[(k_base+1)*n + (i_base+1)]
        U[i_base * m + j_base][r_offset] += 1
        U[(i_base + 1) * m + (j_base + 1)][r_offset] += 1
        V[j_base * p + k_base][r_offset] += 1
        V[(j_base + 1) * p + (k_base + 1)][r_offset] += 1
        W[k_base * n + i_base][r_offset] += 1
        W[(k_base + 1) * n + (i_base + 1)][r_offset] += 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # A[i2,j1] -> U[(i_base+1)*m + j_base]
        # A[i2,j2] -> U[(i_base+1)*m + (j_base+1)]
        # B[j1,k1] -> V[j_base*p + k_base]
        # C[i2,k1] -> W[k_base*n + (i_base+1)]
        # C[i2,k2] -> W[(k_base+1)*n + (i_base+1)]
        U[(i_base + 1) * m + j_base][r_offset + 1] += 1
        U[(i_base + 1) * m + (j_base + 1)][r_offset + 1] += 1
        V[j_base * p + k_base][r_offset + 1] += 1
        W[k_base * n + (i_base + 1)][r_offset + 1] += 1
        W[(k_base + 1) * n + (i_base + 1)][r_offset + 1] += 1

        # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
        # A[i1,j1] -> U[i_base*m + j_base]
        # B[j2,k1] -> V[(j_base+1)*p + k_base]
        # B[j2,k2] -> V[(j_base+1)*p + (k_base+1)]
        # C[i1,k1] -> W[k_base*n + i_base]
        # C[i1,k2] -> W[(k_base+1)*n + i_base]
        U[i_base * m + j_base][r_offset + 2] += 1
        V[(j_base + 1) * p + k_base][r_offset + 2] += 1
        V[(j_base + 1) * p + (k_base + 1)][r_offset + 2] += -1
        W[k_base * n + i_base][r_offset + 2] += 1
        W[(k_base + 1) * n + i_base][r_offset + 2] += -1

        # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
        # A[i2,j2] -> U[(i_base+1)*m + (j_base+1)]
        # B[j1,k2] -> V[j_base*p + (k_base+1)]
        # B[j1,k1] -> V[j_base*p + k_base]
        # C[i2,k2] -> W[(k_base+1)*n + (i_base+1)]
        # C[i2,k1] -> W[k_base*n + (i_base+1)]
        U[(i_base + 1) * m + (j_base + 1)][r_offset + 3] += 1
        V[j_base * p + (k_base + 1)][r_offset + 3] += 1
        V[j_base * p + k_base][r_offset + 3] += -1
        W[(k_base + 1) * n + (i_base + 1)][r_offset + 3] += 1
        W[k_base * n + (i_base + 1)][r_offset + 3] += -1

        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # A[i1,j1] -> U[i_base*m + j_base]
        # A[i1,j2] -> U[i_base*m + (j_base+1)]
        # B[j2,k2] -> V[(j_base+1)*p + (k_base+1)]
        # C[i1,k2] -> W[(k_base+1)*n + i_base]
        # C[i2,k2] -> W[(k_base+1)*n + (i_base+1)]
        U[i_base * m + j_base][r_offset + 4] += 1
        U[i_base * m + (j_base + 1)][r_offset + 4] += 1
        V[(j_base + 1) * p + (k_base + 1)][r_offset + 4] += 1
        W[(k_base + 1) * n + i_base][r_offset + 4] += 1
        W[(k_base + 1) * n + (i_base + 1)][r_offset + 4] += 1

        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # A[i2,j1] -> U[(i_base+1)*m + j_base]
        # A[i1,j1] -> U[i_base*m + j_base]
        # B[j1,k1] -> V[j_base*p + k_base]
        # B[j1,k2] -> V[j_base*p + (k_base+1)]
        # C[i2,k1] -> W[k_base*n + (i_base+1)]
        # C[i2,k2] -> W[(k_base+1)*n + (i_base+1)]
        U[(i_base + 1) * m + j_base][r_offset + 5] += 1
        U[i_base * m + j_base][r_offset + 5] += -1
        V[j_base * p + k_base][r_offset + 5] += 1
        V[j_base * p + (k_base + 1)][r_offset + 5] += 1
        W[k_base * n + (i_base + 1)][r_offset + 5] += 1
        W[(k_base + 1) * n + (i_base + 1)][r_offset + 5] += 1

        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # A[i1,j2] -> U[i_base*m + (j_base+1)]
        # A[i2,j2] -> U[(i_base+1)*m + (j_base+1)]
        # B[j2,k1] -> V[(j_base+1)*p + k_base]
        # B[j2,k2] -> V[(j_base+1)*p + (k_base+1)]
        # C[i1,k1] -> W[k_base*n + i_base]
        # C[i2,k1] -> W[k_base*n + (i_base+1)]
        U[i_base * m + (j_base + 1)][r_offset + 6] += 1
        U[(i_base + 1) * m + (j_base + 1)][r_offset + 6] += -1
        V[(j_base + 1) * p + k_base][r_offset + 6] += 1
        V[(j_base + 1) * p + (k_base + 1)][r_offset + 6] += 1
        W[k_base * n + i_base][r_offset + 6] += 1
        W[k_base * n + (i_base + 1)][r_offset + 6] += 1

    # Group 1: rows 0, 1 (2x4x8)
    # Split 4 columns into two groups of 2: j_base = 0, 2
    # Split 8 columns into four groups of 2: k_base = 0, 2, 4, 6
    # For each combination, add a Strassen block.
    r = 0
    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            add_strassen_block(0, j_base, k_base, r)
            r += 7
    # r is now 56

    # Group 2: row 2 (1x4x8)
    # Schoolbook: rank = 4*8 = 32
    for j in range(4):
        for k in range(8):
            U[2 * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + 2][r] = 1
            r += 1
    # r is now 88

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
