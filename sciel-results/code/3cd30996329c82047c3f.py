import numpy as np
import time
from fractions import Fraction

def construct():
    _t0 = time.time()
    _deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 4, 7  # A is 2x4, B is 4x7, C is 2x7
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i  # W[k*n+i]

    # Build the multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    # Actually, the tensor for matrix multiplication A(n,m) * B(m,p) = C(n,p) is:
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Wait, the decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # But this sums over r, and the free indices are i, j, k in a specific way.
    # Let's use the standard tensor notation.
    # The multiplication tensor T has indices (i, j, k) for A[i,j], B[j,k], C[i,k].
    # A rank-1 term is u_{i,j} * v_{j,k} * w_{i,k}.
    # The full tensor is T_{i,j,k} = 1 (for the "multiply" part, but actually it's the identity on the j-index contraction).
    # More precisely, the tensor M in the space A^* \otimes B^* \otimes C is:
    # M = sum_{i,j,k} e_i \otimes e_j \otimes e_k  (where e_i is basis for C, but we need to be careful with duals)
    # Actually, let's just use the coefficient condition:
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [m), k in [p).
    # Note that U is indexed by (i,j), V by (j,k), W by (i,k) [via k*n+i].
    # So the condition is: for each triple (i,j,k), the sum over r of the product is 1.

    # Let's formulate this as a linear system.
    # Variables: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for r=0..R-1.
    # This is a system of quadratic equations (products of variables), not linear.
    # So we can't just use linear algebra directly on the variables.

    # However, we can use a greedy approach:
    # 1. Start with R = some lower bound (e.g., 50).
    # 2. Try to find a set of rank-1 terms (with integer/half-integer coefficients) that sum to T.
    # 3. A rank-1 term is defined by non-zero entries in U_r, V_r, W_r.
    # 4. The contribution of term r to T[i,j,k] is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # 5. We need sum_r (U_r[i,j] * V_r[j,k] * W_r[i,k]) = 1 for all (i,j,k).

    # This is still hard. Let's stick to the block decomposition but try to optimize the rank.
    # The current best is 50. Can we do better?
    # Theoretical lower bound: The tensor rank of 2x4x7 multiplication.
    # For n x m x p, the rank is at least max(n*m, m*p, p*n) / something? No.
    # Actually, for n=m=p, Strassen gives 7n^log2(7). For 2x4x7, it's irregular.

    # Let's try to construct a rank-49 or lower solution.
    # The previous solution used:
    # 3 blocks of 2x2x2 (Strassen, rank 7 each) = 21
    # 1 block of 2x2x1 (schoolbook, rank 4) = 4
    # Wait, the previous solution had:
    # Block 1 (cols 0,1): 2 Strassen blocks (A_L*B1_L + A_R*B1_R) = 14
    # Block 2 (cols 2,3): 2 Strassen blocks = 14
    # Block 3 (cols 4,5): 2 Strassen blocks = 14
    # Block 4 (col 6): 2 schoolbook 2x2x1 blocks = 8
    # Total = 14+14+14+8 = 50.

    # Can we combine the last block (col 6) more efficiently?
    # Instead of 2 schoolbook blocks (rank 4 each), can we do it in rank 3 or less?
    # A 2x4 matrix times a 4x1 matrix.
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6] + A[0,2]B[2,6] + A[0,3]B[3,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6] + A[1,2]B[2,6] + A[1,3]B[3,6]
    # This is a 2x4x1 tensor. The rank of a 2xn x 1 tensor is n (schoolbook) unless there's structure.
    # Actually, the tensor rank of 2x4x1 is 4. So we can't do better than 4 for that block alone.
    # But maybe we can share terms between the last column and the previous columns?
    # That would require a non-block-diagonal decomposition.

    # Let's try a different approach: use the "greedy sparse support search" as suggested.
    # But that's computationally expensive. Given the 140s deadline, we might not have time for a full search.

    # Alternative: Try to find a rank-49 solution by modifying the block decomposition.
    # For example, can we merge one of the Strassen blocks with the schoolbook block?
    # This is unlikely to work easily.

    # Let's just return the rank-50 solution, but with a cleaner implementation.
    # Actually, let's try to see if we can get rank 49 by using a different split.
    # What if we split B into 4x3 and 4x4?
    # 4x3: split into 4x2 + 4x1.
    # 4x2: 2 Strassen blocks = 14.
    # 4x1: 2 schoolbook blocks = 8.
    # 4x4: 4 Strassen blocks = 28.
    # Total = 14+8+28 = 50. Same.

    # What if we use a 3x3 Strassen-like algorithm for some part?
    # Probably not worth the complexity.

    # Let's just implement the rank-50 solution correctly and efficiently.
    rank = 50
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)

    col = 0

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (a10+a11)b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        # M3 = a00(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4 = a11(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5 = (a00+a01)b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        # M6 = (a10-a00)(b00+b01)
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        # M7 = (a01-a11)(b10+b11)
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(a00, a01, a10, a11, b00, b10, c00, c10):
        nonlocal col
        U[a00, col] += 1; V[b00, col] += 1; W[c00, col] += 1
        col += 1
        U[a01, col] += 1; V[b10, col] += 1; W[c00, col] += 1
        col += 1
        U[a10, col] += 1; V[b00, col] += 1; W[c10, col] += 1
        col += 1
        U[a11, col] += 1; V[b10, col] += 1; W[c10, col] += 1
        col += 1

    # Block 1: Cols 0,1
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
                 w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1))
    add_strassen(u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
                 v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
                 w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1))

    # Block 2: Cols 2,3
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
                 w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3))
    add_strassen(u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
                 v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3),
                 w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3))

    # Block 3: Cols 4,5
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,4), v_idx(0,5), v_idx(1,4), v_idx(1,5),
                 w_idx(0,4), w_idx(0,5), w_idx(1,4), w_idx(1,5))
    add_strassen(u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
                 v_idx(2,4), v_idx(2,5), v_idx(3,4), v_idx(3,5),
                 w_idx(0,4), w_idx(0,5), w_idx(1,4), w_idx(1,5))

    # Block 4: Col 6
    add_schoolbook_221(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                       v_idx(0,6), v_idx(1,6),
                       w_idx(0,6), w_idx(1,6))
    add_schoolbook_221(u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
                       v_idx(2,6), v_idx(3,6),
                       w_idx(0,6), w_idx(1,6))

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(col):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    # Fallback: return schoolbook rank 56
                    U = np.zeros((nm, 56), dtype=int)
                    V = np.zeros((mp, 56), dtype=int)
                    W = np.zeros((pn, 56), dtype=int)
                    col2 = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m+j2, col2] = 1
                                V[j2*p+k2, col2] = 1
                                W[k2*n+i2, col2] = 1
                                col2 += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
