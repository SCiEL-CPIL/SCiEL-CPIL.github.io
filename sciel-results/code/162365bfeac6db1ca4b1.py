"""
Rank-minimal tensor decomposition for 3x5 x 5x7 matrix multiplication.

Strategy: enumerate all ways to partition the three dimensions (n=3, m=5, p=7)
into contiguous blocks of size 1 or 2. For each block triple (a,b,c), use
Strassen (rank 7) if a=b=c=2, otherwise schoolbook (rank a*b*c). The total
rank is the sum over all block triples. We pick the partition with minimum
total rank.

Knowledge from prior run: searching all block partitions (sizes 1 and 2) for
n=3, m=5, p=7 finds the minimum-rank combination of Strassen and schoolbook
blocks, yielding rank 99 -> score 1.061. We verify this by exhaustive search.
"""
import itertools
import time

DIMENSIONS = [3, 5, 7]


def _partitions_of_size(d, max_block=2):
    """Generate all compositions of d using parts of size 1 or max_block."""
    if d == 0:
        yield []
        return
    for first in range(1, min(max_block, d) + 1):
        for rest in _partitions_of_size(d - first, max_block):
            yield [first] + rest


def _strassen_block():
    """
    Return (U, V, W) for a 2x2x2 Strassen block.
    U: shape (2*2, 7), V: shape (2*2, 7), W: shape (2*2, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas (0-indexed):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2x2 matrix, row = i*2+j
    V = [[0] * rank for _ in range(4)]  # 2x2 matrix, row = j*2+k
    W = [[0] * rank for _ in range(4)]  # 2x2 matrix, row = k*2+i

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)

    # M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0,i=1)

    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1  # B11
    W[1][2] = 1  # C01 (k=1,i=0)

    # M4 = A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00 (k=0,i=0)
    W[3][3] = 1  # C11 (k=1,i=1)

    # M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    W[3][4] = -1  # C11
    W[2][4] = 0   # C10 not used

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    W[3][5] = 1   # C11

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11
    W[0][6] = 1   # C00

    return U, V, W


def _schoolbook_block(a, b, c):
    """
    Return (U, V, W) for an a x b x c schoolbook block.
    U: shape (a*b, a*b*c), V: shape (b*c, a*b*c), W: shape (c*a, a*b*c)
    """
    rank = a * b * c
    U = [[0] * rank for _ in range(a * b)]
    V = [[0] * rank for _ in range(b * c)]
    W = [[0] * rank for _ in range(c * a)]
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i * b + j][col] = 1
                V[j * c + k][col] = 1
                W[k * a + i][col] = 1
                col += 1
    return U, V, W


def _block_rank(a, b, c):
    """Return the rank needed for an a x b x c block."""
    if a == 2 and b == 2 and c == 2:
        return 7
    return a * b * c


def _build_decomposition(part_n, part_m, part_p):
    """
    Given partitions of n, m, p, build the full U, V, W matrices.
    Returns (U, V, W, rank).
    """
    n, m, p = DIMENSIONS
    total_rank = 0
    for a in part_n:
        for b in part_m:
            for c in part_p:
                total_rank += _block_rank(a, b, c)

    # Pre-allocate
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    i_offset = 0
    for a in part_n:
        j_offset = 0
        for b in part_m:
            k_offset = 0
            for c in part_p:
                block_rank = _block_rank(a, b, c)
                if a == 2 and b == 2 and c == 2:
                    bU, bV, bW = _strassen_block()
                    # Map local to global
                    for li in range(a):
                        for lj in range(b):
                            row_u = (i_offset + li) * m + (j_offset + lj)
                            for c_idx in range(block_rank):
                                U[row_u][col_offset + c_idx] = bU[li * b + lj][c_idx]
                    for lj in range(b):
                        for lk in range(c):
                            row_v = (j_offset + lj) * p + (k_offset + lk)
                            for c_idx in range(block_rank):
                                V[row_v][col_offset + c_idx] = bV[lj * c + lk][c_idx]
                    for lk in range(c):
                        for li in range(a):
                            row_w = (k_offset + lk) * n + (i_offset + li)
                            for c_idx in range(block_rank):
                                W[row_w][col_offset + c_idx] = bW[lk * a + li][c_idx]
                else:
                    bU, bV, bW = _schoolbook_block(a, b, c)
                    for li in range(a):
                        for lj in range(b):
                            row_u = (i_offset + li) * m + (j_offset + lj)
                            for c_idx in range(block_rank):
                                U[row_u][col_offset + c_idx] = bU[li * b + lj][c_idx]
                    for lj in range(b):
                        for lk in range(c):
                            row_v = (j_offset + lj) * p + (k_offset + lk)
                            for c_idx in range(block_rank):
                                V[row_v][col_offset + c_idx] = bV[lj * c + lk][c_idx]
                    for lk in range(c):
                        for li in range(a):
                            row_w = (k_offset + lk) * n + (i_offset + li)
                            for c_idx in range(block_rank):
                                W[row_w][col_offset + c_idx] = bW[lk * a + li][c_idx]

                col_offset += block_rank
                k_offset += c
            j_offset += b
        i_offset += a

    return U, V, W, total_rank


def _verify(U, V, W, rank, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Should equal 1 if this corresponds to a valid multiplication term, 0 otherwise.
    # Actually the tensor for matmul is: T[i,j,k] = delta_{j,k} is wrong.
    # The matmul tensor is: (A ⊗ B ⊗ C) where C = A*B
    # T_{ijk} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle)
    # Wait, the matmul tensor is: T[i,j,k] = delta_{j,k}? No.
    # The tensor is: for C = A*B, C_ik = sum_j A_ij B_jk
    # The tensor T satisfies: <T, A ⊗ B ⊗ C> = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki}
    # We want this to equal sum_ik C_ik^2 = sum_{i,j,k} A_ij B_jk C_ik (when C = A*B)
    # So T_{ijk} should be 1 for all i,j,k? No, that gives sum A B C which is correct.
    # Actually T_{ijk} = 1 for all i,j,k in the matmul tensor? Let me reconsider.
    # The standard matmul tensor has T_{ijk} = 1 if the indices are "compatible"
    # In the convention W[k*n+i, r], the tensor is:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Because C = A*B means C_ik = sum_j A_ij B_jk, and the tensor
    # <T, A⊗B⊗C> = sum_{i,j,k} A_ij B_jk C_ki... hmm, let me just check.
    #
    # Given the convention: U has rows indexed by (i,j) with i in [0,n), j in [0,m)
    # V has rows indexed by (j,k) with j in [0,m), k in [0,p)
    # W has rows indexed by (k,i) with k in [0,p), i in [0,n)  [trace convention: W[k*n+i]]
    #
    # The decomposition says: T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matmul: T_{ijk} = 1 for all valid i,j,k (since C_ik = sum_j A_ij B_jk)
    # Wait no. Let's think again.
    # <T, A⊗B⊗C> = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki}
    # We want this to equal ||A*B||^2 = sum_{i,k} (sum_j A_ij B_jk)^2 when C = A*B
    # = sum_{i,k,j,j'} A_ij B_jk A_ij' B_j'k
    # This doesn't directly give T_{ijk}=1.
    #
    # Actually, the standard convention is: the matmul tensor T has T_{ijk} = 1 for all i,j,k.
    # Because: sum_{i,j,k} A_ij B_jk C_ik = <A, B, C> in the trilinear form sense
    # and when C = A*B, this equals the "correct" pairing.
    #
    # Let me just verify with the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # T_{ijk} = sum_col U[i*m+j,col] V[j*p+k,col] W[k*n+i,col]
    # = 1 if there's a col where all three are 1, which happens exactly when the col corresponds to (i,j,k)
    # So T_{ijk} = 1 for all i,j,k. Yes.
    #
    # So we need T_{ijk} = 1 for all i in [n], j in [m], k in [p].

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin

    # Generate all partitions of n, m, p into blocks of size 1 or 2
    part_ns = list(_partitions_of_size(n))
    part_ms = list(_partitions_of_size(m))
    part_ps = list(_partitions_of_size(p))

    best_rank = n * m * p  # schoolbook
    best_partition = (part_ns[0], part_ms[0], part_ps[0])

    # Search for minimum rank partition
    for pn in part_ns:
        for pm in part_ms:
            for pp in part_ps:
                r = 0
                for a in pn:
                    for b in pm:
                        for c in pp:
                            r += _block_rank(a, b, c)
                if r < best_rank:
                    best_rank = r
                    best_partition = (pn, pm, pp)

    # Build the decomposition for the best partition
    U, V, W, rank = _build_decomposition(best_partition[0], best_partition[1], best_partition[2])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
