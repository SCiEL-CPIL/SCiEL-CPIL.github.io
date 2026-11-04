"""Program for rank-minimal exact tensor decomposition of 5x5 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Start with schoolbook rank 125 as guaranteed valid fallback
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

    # Try to reduce rank using Strassen-based block decomposition
    # For 5x5, we can use a 4x4 Strassen for the top-left block
    # and handle the remaining rows/columns with schoolbook

    # Strategy: Use Strassen for 4x4 blocks where possible
    # The 5x5 matrix can be decomposed as:
    # A = [A11 A12; A21 A22] where A11 is 4x4, A12 is 4x1, A21 is 1x4, A22 is 1x1
    # B = [B11 B12; B21 B22] where B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1
    # C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22;
    #             A21*B11 + A22*B21, A21*B12 + A22*B22]

    # This gives us:
    # C11 (4x4) = A11*B11 + A12*B21 -> 4x4 Strassen (7) + 4*1*1 schoolbook (4) = 11
    # C12 (4x1) = A11*B12 + A12*B22 -> 4*4*1 schoolbook (16) + 4*1*1 (4) = 20
    # C21 (1x4) = A21*B11 + A22*B21 -> 1*4*4 schoolbook (16) + 1*1*4 (4) = 20
    # C22 (1x1) = A21*B12 + A22*B22 -> 1*4*1 schoolbook (4) + 1*1*1 (1) = 5
    # Total: 11 + 20 + 20 + 5 = 56... but this is not how tensor decomposition works directly

    # Actually, for tensor decomposition, we need to express each entry of C as a sum
    # of products of linear combinations of entries of A and B.

    # Let's try a different approach: use the known result that 5x5 matrix multiplication
    # can be done with rank < 125. The best known upper bound is around 94-100.

    # For now, let's implement a greedy reduction from schoolbook:
    # Try to merge pairs of rank-1 terms if they can be combined.

    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=object)
    V_np = np.array(V, dtype=object)
    W_np = np.array(W, dtype=object)

    # Greedy rank reduction: try to find pairs of terms that can be merged
    # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if:
    # u1*u2 == 0 elementwise (or v1*v2 == 0, or w1*w2 == 0)
    # This is a sufficient but not necessary condition.

    # A better approach: use the fact that if two terms have disjoint support
    # in U, V, or W, they can potentially be combined.

    # For simplicity, let's try a random local search approach:
    # 1. Start with schoolbook
    # 2. Randomly perturb the matrices
    # 3. Check if the decomposition is still valid
    # 4. If valid and rank is reduced, keep it

    # But this is complex. Let's instead try to use known efficient decompositions.

    # The best known decomposition for 5x5 matrix multiplication has rank 94.
    # Let's try to construct it using a block approach.

    # Alternative: use the fact that we can compute 5x5 = 4x4 + cross terms
    # A = [A44, a; a^T, s] where A44 is 4x4, a is 4x1, s is scalar
    # B = [B44, b; b^T, t] where B44 is 4x4, b is 4x1, t is scalar
    # C = [A44*B44 + a*b^T, A44*b + a*t;
    #       a^T*B44 + s*b^T, a^T*b + s*t]

    # C11 (4x4): A44*B44 (rank 16 schoolbook or 7 Strassen) + a*b^T (rank 4)
    #   Using Strassen for A44*B44: rank 7
    #   a*b^T: each entry (i,j) = a_i * b_j, rank 4 total
    #   Total for C11: 7 + 4 = 11
    #
    # C12 (4x1): A44*b (rank 16 schoolbook) + a*t (rank 4)
    #   A44*b: (i) = sum_j A44_ij * b_j, rank 16
    #   a*t: (i) = a_i * t, rank 4
    #   Total for C12: 16 + 4 = 20
    #
    # C21 (1x4): a^T*B44 (rank 16) + s*b^T (rank 4)
    #   Total for C21: 16 + 4 = 20
    #
    # C22 (1x1): a^T*b (rank 4) + s*t (rank 1)
    #   Total for C22: 4 + 1 = 5
    #
    # Grand total: 11 + 20 + 20 + 5 = 56

    # This is much better than 125! Let's implement this.

    # But wait, I need to be careful about the tensor decomposition structure.
    # The decomposition must satisfy:
    # C_ki = sum_r U_ir * V_rk * W_r (or similar indexing)

    # Let me re-read the interface:
    # U: (n*m, rank) - so U[i*m+j, r] is the coefficient of A_ij in the r-th term
    # V: (m*p, rank) - so V[j*p+k, r] is the coefficient of B_jk in the r-th term
    # W: (p*n, rank) - so W[k*n+i, r] is the coefficient for C_ki in the r-th term
    #
    # The tensor equation is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid matrix multiplication term
    # i.e., T[i,j,k] = 1 for all i,j,k (since C_ik = sum_j A_ij * B_jk)

    # So we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Let's build the 56-rank decomposition:

    # Initialize with rank 56
    rank_new = 56
    U_new = [[0] * rank_new for _ in range(n * m)]
    V_new = [[0] * rank_new for _ in range(m * p)]
    W_new = [[0] * rank_new for _ in range(p * n)]

    col = 0

    # C11 (4x4) = A44*B44 + a*b^T
    # A44*B44 using Strassen 4x4 (rank 7)
    # Strassen 4x4: divide 4x4 into 2x2 blocks
    # A44 = [A11 A12; A21 A22] where each is 2x2
    # B44 = [B11 B12; B21 B22] where each is 2x2

    # Strassen 2x2 has rank 7. For 4x4, we use 7 Strassen 2x2 computations,
    # each of rank 7, so total rank 49 for the 4x4 Strassen.

    # Wait, that's not right. Let me reconsider.

    # For a 4x4 matrix multiplication, we can use:
    # - Schoolbook: rank 16
    # - Strassen 2x2 applied to 2x2 blocks: 7 * 7 = 49 rank? No...

    # Actually, the standard Strassen for 4x4 divides into four 2x2 blocks and
    # uses 7 recursive 2x2 multiplications. Each 2x2 multiplication has rank 4
    # (schoolbook) or rank 7 (Strassen). If we use schoolbook for the 2x2, total
    # rank is 7*4 = 28. If we use Strassen for the 2x2, total rank is 7*7 = 49.

    # So schoolbook 4x4 is rank 16, which is better than Strassen 4x4 (rank 28 or 49).

    # Let's use schoolbook for the 4x4 blocks.

    # C11 (4x4) = A44*B44 + a*b^T
    # A44*B44: schoolbook, rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # C_ik += A_ij * B_jk
                U_new[i * m + j][col] = 1
                V_new[j * p + k][col] = 1
                W_new[k * n + i][col] = 1
                col += 1

    # a*b^T: rank 4
    for i in range(4):
        for k in range(4):
            # C_ik += a_i * b_k
            # a_i = A_{4,i} (0-indexed: A[4][i])
            # b_k = B_{k,4} (0-indexed: B[k][4])
            U_new[4 * m + i][col] = 1  # A[4,i]
            V_new[i * p + 4][col] = 1  # B[i,4]
            W_new[k * n + i][col] = 1  # C[k,i]
            col += 1

    # C12 (4x1) = A44*b + a*t
    # A44*b: rank 16
    for i in range(4):
        for j in range(4):
            # C_{i,4} += A_ij * b_j
            # b_j = B_{j,4}
            U_new[i * m + j][col] = 1
            V_new[j * p + 4][col] = 1
            W_new[4 * n + i][col] = 1  # C[4,i] -> index 4*n+i
            col += 1

    # a*t: rank 4
    for i in range(4):
        # C_{i,4} += a_i * t
        # a_i = A_{4,i}, t = B_{4,4}
        U_new[4 * m + i][col] = 1
        V_new[4 * p + 4][col] = 1
        W_new[4 * n + i][col] = 1
        col += 1

    # C21 (1x4) = a^T*B44 + s*b^T
    # a^T*B44: rank 16
    for j in range(4):
        for k in range(4):
            # C_{4,k} += a_j * B_jk
            # a_j = A_{4,j}
            U_new[4 * m + j][col] = 1
            V_new[j * p + k][col] = 1
            W_new[k * n + 4][col] = 1  # C[k,4] -> index k*n+4
            col += 1

    # s*b^T: rank 4
    for k in range(4):
        # C_{4,k} += s * b_k
        # s = A_{4,4}, b_k = B_{k,4}
        U_new[4 * m + 4][col] = 1
        V_new[k * p + 4][col] = 1
        W_new[k * n + 4][col] = 1
        col += 1

    # C22 (1x1) = a^T*b + s*t
    # a^T*b: rank 4
    for j in range(4):
        # C_{4,4} += a_j * b_j
        # a_j = A_{4,j}, b_j = B_{j,4}
        U_new[4 * m + j][col] = 1
        V_new[j * p + 4][col] = 1
        W_new[4 * n + 4][col] = 1
        col += 1

    # s*t: rank 1
    # C_{4,4} += s * t
    # s = A_{4,4}, t = B_{4,4}
    U_new[4 * m + 4][col] = 1
    V_new[4 * p + 4][col] = 1
    W_new[4 * n + 4][col] = 1
    col += 1

    # Verify col == rank_new
    assert col == rank_new, f"Expected {rank_new} columns, got {col}"

    # Verify the decomposition is correct
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank_new):
                    total += U_new[i * m + j][r] * V_new[j * p + k][r] * W_new[k * n + i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
