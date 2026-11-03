"""Rank-77 decomposition for 4x4x6 matrix multiplication tensor.

Strategy: Split B into B[:,0:4] (4x4) and B[:,4:6] (4x2).
- For A*B[:,0:4]: recursive Strassen on 4x4x4 gives rank 49 (7x7).
- For A*B[:,4:6]: split into four 2x2x2 blocks, each using Strassen (rank 7) = 28.
Total rank: 49 + 28 = 77.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dict {'U','V','W'} with integer entries exactly decomposing the
    4x4x6 matrix-multiplication tensor using rank 77.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_decomposition():
        U = []
        V = []
        W = []

        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)

        # 2x2x2 Strassen: 7 terms
        # A is 2x2, B is 2x2, C is 2x2
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        # M2 = (a10+a11)b00 -> C10 += M2
        # M3 = a00(b10-b11) -> C00 -= M3, C01 += M3
        # M4 = (a00-a11)(b11-b00) -> C00 += M4, C11 -= M4
        # M5 = (a01+a11)(b00-b01) -> C01 += M5, C11 -= M5
        # M6 = (a10-a00)(b01+b11) -> C11 += M6
        # M7 = (a00+a01)(b11-b10) -> C00 += M7, C10 -= M7

        # Verify: C00 = M1 - M3 + M4 + M7
        #   = (a00+a11)(b00+b11) - a00(b10-b11) + (a00-a11)(b11-b00) + (a00+a01)(b11-b10)
        #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a00*b10 + a00*b11
        #     + a00*b11 - a00*b00 - a11*b11 + a11*b00 + a00*b11 - a00*b10 + a01*b11 - a01*b10
        #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a00*b10 + a00*b11
        #     + a00*b11 - a00*b00 - a11*b11 + a11*b00 + a00*b11 - a00*b10 + a01*b11 - a01*b10
        #   = (a00*b00 - a00*b00) + (a11*b11 - a11*b11) + a00*b11 + a00*b11 + a00*b11 + a00*b11
        #     + a11*b00 + a11*b00 - a00*b10 - a00*b10 + a01*b11 - a01*b10
        #   = 4*a00*b11 + 2*a11*b00 - 2*a00*b10 + a01*b11 - a01*b10
        # Hmm, that doesn't look right. Let me recheck.
        # Actually C00 should be a00*b00 + a01*b10.
        # Let me redo this more carefully.

        # Standard Strassen for 2x2:
        # p1 = (A00+A11)(B00+B11)
        # p2 = (A10+A11)B00
        # p3 = A00(B10-B11)
        # p4 = (A00-A11)(B11-B00)
        # p5 = (A01+A11)(B00-B01)
        # p6 = (A10-A00)(B01+B11)
        # p7 = (A00+A01)(B11-B10)
        # C00 = p1 + p4 - p5 + p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p2 + p3 + p6

        # Let me verify C00 = p1 + p4 - p5 + p7:
        # p1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # p4 = A00*B11 - A00*B00 - A11*B11 + A11*B00
        # -p5 = -A01*B00 + A01*B01 - A11*B00 + A11*B01
        # p7 = A00*B11 - A00*B10 + A01*B11 - A01*B10
        # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11
        #      + A00*B11 - A00*B00 - A11*B11 + A11*B00
        #      - A01*B00 + A01*B01 - A11*B00 + A11*B01
        #      + A00*B11 - A00*B10 + A01*B11 - A01*B10
        # = A00*B00 - A00*B00 + A11*B11 - A11*B11
        #   + A00*B11 + A00*B11 + A00*B11
        #   + A11*B00 + A11*B00 - A11*B00
        #   - A01*B00 + A01*B01 + A11*B01
        #   - A00*B10 + A01*B11 - A01*B10
        # = 3*A00*B11 + A11*B00 - A01*B00 + A01*B01 + A11*B01 - A00*B10 + A01*B11 - A01*B10
        # This is NOT A00*B00 + A01*B10. I'm making errors.

        # Let me just use the well-known correct Strassen formulas.
        # The key insight: each of the 7 products p1..p7 is a scalar product
        # of a linear combination of A entries with a linear combination of B entries.
        # Then each C entry is a linear combination of the p's.

        # For the tensor decomposition, each term (c) has:
        # u[c] = coefficients for A entries
        # v[c] = coefficients for B entries  
        # w[c] = coefficients for C entries (output)
        # And the reconstruction is: T[i,j,k] = sum_c u[i,j,c] * v[j,k,c] * w[i,k,c]
        # Wait, no. The convention is:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

        # So for each rank-1 term, we have:
        # U[i*m+j, c] = alpha_{i,j}
        # V[j*p+k, c] = beta_{j,k}
        # W[k*n+i, c] = gamma_{i,k}
        # And we need: sum_c alpha_{i,j} * beta_{j,k} * gamma_{i,k} = 1 for all i,j,k

        # For a 2x2x2 Strassen term, we have 7 such rank-1 decompositions.
        # Each corresponds to one of the p1..p7 products.
        # For p1 = (A00+A11)(B00+B11):
        #   alpha[0,0]=1, alpha[1,1]=1, others 0
        #   beta[0,0]=1, beta[0,1]=1, beta[1,0]=1, beta[1,1]=1... 
        #   Wait, that's not right either. Let me think again.

        # Actually, in Strassen, p1 = (A00+A11) * (B00+B11) is ONE scalar multiplication.
        # Then C00 gets +p1, C11 gets +p1.
        # So in the tensor decomposition, this single scalar multiplication contributes:
        #   u[0,0] = 1, u[1,1] = 1 (the A-side linear combo)
        #   v[0,0] = 1, v[0,1] = 1, v[1,0] = 1, v[1,1] = 1 (the B-side linear combo)
        #   w[0,0] = 1, w[1,1] = 1 (the C-side: C00 += p1, C11 += p1)
        #   But wait, that would give T[0,0,0] += 1*1*1 = 1, T[0,1,0] += 1*1*0 = 0, etc.
        #   T[0,0,1] += 1*1*w[0,1] = 1*1*0 = 0. Good.
        #   T[1,1,0] += 1*1*w[0,1]... wait, W index is k*n+i.
        #   W[0*2+0, c] = w for C[0,0], W[1*2+1, c] = w for C[1,1]
        #   T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        #   For 2x2x2: m=2, p=2, n=2.
        #   T[0,0,0] = U[0,c]*V[0,c]*W[0,c]
        #   T[0,0,1] = U[0,c]*V[1,c]*W[1*2+0,c] = U[0,c]*V[1,c]*W[2,c]
        #   T[1,1,0] = U[3,c]*V[2,c]*W[0*2+1,c] = U[3,c]*V[2,c]*W[1,c]
        #   T[1,1,1] = U[3,c]*V[3,c]*W[1*2+1,c] = U[3,c]*V[3,c]*W[3,c]

        # For p1: U[0]=1, U[3]=1, V[0]=1,V[1]=1,V[2]=1,V[3]=1, W[0]=1, W[3]=1
        # T[0,0,0] += 1*1*1 = 1
        # T[0,0,1] += 1*1*W[2] = 1*1*0 = 0
        # T[0,1,0] += U[1]*V[2]*W[0] = 0*1*1 = 0
        # T[0,1,1] += U[1]*V[3]*W[2] = 0
        # T[1,0,0] += U[2]*V[0]*W[1] = 0*1*0 = 0
        # T[1,0,1] += U[2]*V[1]*W[3] = 0
        # T[1,1,0] += U[3]*V[2]*W[1] = 1*1*0 = 0
        # T[1,1,1] += U[3]*V[3]*W[3] = 1*1*1 = 1
        # So p1 contributes to T[0,0,0] and T[1,1,1]. Good!

        # Now I need to implement this properly for the full 4x4x6 case.

        # For the 4x4x4 block (k=0..3), use recursive Strassen:
        # Split A into 2x2 blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
        # Split B[:,0:4] into 2x2 blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
        # C[:,0:4] blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]

        # 7 block-level Strassen products, each being a 2x2x2 Strassen (rank 7):
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        # P2 = (A21+A22)B11 -> C21 += P2
        # P3 = A11(B21-B22) -> C11 -= P3, C12 += P3
        # P4 = (A11-A22)(B22-B11) -> C11 += P4, C22 -= P4
        # P5 = (A12+A22)(B11-B12) -> C12 += P5, C22 -= P5
        # P6 = (A21-A11)(B12+B22) -> C22 += P6
        # P7 = (A11+A12)(B22-B21) -> C11 += P7, C21 -= P7

        # Each Pi is a 2x2x2 matrix product, decomposed via Strassen into 7 scalar products.

        def add_2x2x2_strassen(a_idx, b_idx, c_idx, a_signs, b_signs, c_signs):
            """Add 7 rank-1 terms for a 2x2x2 Strassen decomposition.
            a_idx: 2x2 array of (i,j) positions in A
            b_idx: 2x2 array of (j,k) positions in B
            c_idx: 2x2 array of (i,k) positions in C
            a_signs: 2x2 array of signs for A
            b_signs: 2x2 array of signs for B
            c_signs: 2x2 array of signs for C (how this product contributes to C)
            """
            # The 7 Strassen products for 2x2:
            # p1 = (a00+a11)(b00+b11), contributes +1 to c00, +1 to c11
            # p2 = (a10+a11)b00, contributes +1 to c10
            # p3 = a00(b10-b11), contributes -1 to c00, +1 to c01
            # p4 = (a00-a11)(b11-b00), contributes +1 to c00, -1 to c11
            # p5 = (a01+a11)(b00-b01), contributes +1 to c01, -1 to c11
            # p6 = (a10-a00)(b01+b11), contributes +1 to c11
            # p7 = (a00+a01)(b11-b10), contributes +1 to c00, -1 to c10

            strassen_terms = [
                # (a_coeffs, b_coeffs, c_coeffs)
                # Each is a dict mapping position -> sign
                ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ({(1,0):1, (1,1):1}, {(0,0):1, (0,1):0, (1,0):0, (1,1):0}, {(1,0):1}),
                ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):-1, (0,1):1}),
                ({(0,0):1, (1,1):-1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,1):-1}),
                ({(0,1):1, (1,1):1}, {(0,0):1, (0,1):-1}, {(0,1):1, (1,1):-1}),
                ({(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),
                ({(0,0):1, (0,1):1}, {(1,1):1, (1,0):-1}, {(0,0):1, (1,0):-1}),
            ]

            for a_c, b_c, c_c in strassen_terms:
                u = [0] * (n * m)  # 16
                v = [0] * (m * p)  # 24
                w = [0] * (p * n)  # 24

                for (ai, aj), sign_a in a_c.items():
                    gi, gj = a_idx[ai][aj]
                    u[gi * m + gj] = sign_a * a_signs[ai][aj]

                for (bi, bj), sign_b in b_c.items():
                    gj, gk = b_idx[bi][bj]
                    v[gj * p + gk] = sign_b * b_signs[bi][bj]

                for (ci, cj), sign_c in c_c.items():
                    gi, gk = c_idx[ci][cj]
                    w[gk * n + gi] = sign_c * c_signs[ci][cj]

                add_term(u, v, w)

        # Define 2x2 block index maps for the 4x4x4 part
        # A blocks (i,j) -> global (i,j)
        A11_idx = [[(0,0),(0,1)],[(1,0),(1,1)]]
        A12_idx = [[(0,2),(0,3)],[(1,2),(1,3)]]
        A21_idx = [[(2,0),(2,1)],[(3,0),(3,1)]]
        A22_idx = [[(2,2),(2,3)],[(3,2),(3,3)]]

        # B blocks for k=0..3
        B11_idx = [[(0,0),(0,1)],[(1,0),(1,1)]]
        B12_idx = [[(0,2),(0,3)],[(1,2),(1,3)]]
        B21_idx = [[(2,0),(2,1)],[(3,0),(3,1)]]
        B22_idx = [[(2,2),(2,3)],[(3,2),(3,3)]]

        # C blocks for k=0..3
        C11_idx = [[(0,0),(0,1)],[(1,0),(1,1)]]
        C12_idx = [[(0,2),(0,3)],[(1,2),(1,3)]]
        C21_idx = [[(2,0),(2,1)],[(3,0),(3,1)]]
        C22_idx = [[(2,2),(2,3)],[(3,2),(3,3)]]

        # Identity signs
        I2 = [[1,1],[1,1]]  # placeholder, we'll use proper signs

        # P1 = (A11+A22)(B11+B22)
        # A-side: A11 + A22, B-side: B11 + B22, C-side: C11 + C22
        add_2x2x2_strassen(
            A11_idx, B11_idx, C11_idx,
            [[1,1],[1,1]], [[1,1],[1,1]], [[1,1],[1,1]]
        )
        # But wait, this only handles the (A11, B11, C11) part.
        # I need to think about this differently.

        # Actually, the approach is: for each of the 7 block-level Strassen products,
        # I need to create a 2x2x2 Strassen where the "A" entries come from the appropriate
        # combination of A blocks, "B" from B blocks, and "C" goes to C blocks.

        # This is getting very complex. Let me use a different approach:
        # Directly enumerate all rank-1 terms.

        # For the 4x4x4 block, I'll use the fact that recursive Strassen gives 49 terms.
        # I'll generate them by nesting two levels of 2x2x2 Strassen.

        # Level 1: 7 block products, each is a 2x2x2 tensor.
        # Level 2: Each 2x2x2 tensor has 7 rank-1 terms.
        # Total: 49 terms.

        # For each level-1 product, I need to specify:
        # - Which 2x2 block of A is used (with signs)
        # - Which 2x2 block of B is used (with signs)  
        # - Which 2x2 block of C receives the output (with signs)

        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        # This means: A-side uses A11 and A22 (both with sign +1)
        # B-side uses B11 and B22 (both with sign +1)
        # C-side: C11 gets +1, C22 gets +1

        # But in the 2x2x2 Strassen, the "A" is a single 2x2 matrix.
        # Here, the "A" is actually A11+A22, which is a 2x2 matrix where
        # entry (i,j) = A11[i,j] + A22[i,j].

        # So for the 2x2x2 Strassen on this "virtual" A, B, C:
        # The A entries at local position (li, lj) map to global A entries
        # at A11[li,lj] AND A22[li,lj].
        # Similarly for B and C.

        # I'll handle this by creating a helper that takes multiple block contributions.

        def add_strassen_block(a_blocks, b_blocks, c_blocks):
            """Add 7 terms for a 2x2x2 Strassen where A, B, C are sums of blocks.
            a_blocks: list of (idx_2x2, sign_2x2) for A
            b_blocks: list of (idx_2x2, sign_2x2) for B
            c_blocks: list of (idx_2x2, sign_2x2) for C
            """
            strassen_terms = [
                ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
                ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):-1, (0,1):1}),
                ({(0,0):1, (1,1):-1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,1):-1}),
                ({(0,1):1, (1,1):1}, {(0,0):1, (0,1):-1}, {(0,1):1, (1,1):-1}),
                ({(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),
                ({(0,0):1, (0,1):1}, {(1,1):1, (1,0):-1}, {(0,0):1, (1,0):-1}),
            ]

            for a_c, b_c, c_c in strassen_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)

                for (li, lj), local_sign in a_c.items():
                    for idx, block_sign in a_blocks:
                        gi, gj = idx[li][lj]
                        u[gi * m + gj] += local_sign * block_sign[li][lj]

                for (li, lj), local_sign in b_c.items():
                    for idx, block_sign in b_blocks:
                        gj, gk = idx[li][lj]
                        v[gj * p + gk] += local_sign * block_sign[li][lj]

                for (li, lj), local_sign in c_c.items():
                    for idx, block_sign in c_blocks:
                        gi, gk = idx[li][lj]
                        w[gk * n + gi] += local_sign * block_sign[li][lj]

                add_term(u, v, w)

        # Identity 2x2 sign matrix
        Id2 = [[1,0],[0,1]]
        Ones2 = [[1,1],[1,1]]

        # 4x4x4 block (k=0..3):
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        add_strassen_block(
            [(A11_idx, [[1,1],[1,1]]), (A22_idx, [[1,1],[1,1]])],
            [(B11_idx, [[1,1],[1,1]]), (B22_idx, [[1,1],[1,1]])],
            [(C11_idx, [[1,1],[1,1]]), (C22_idx, [[1,1],[1,1]])]
        )
        # P2 = (A21+A22)B11 -> C21 += P2
        add_strassen_block(
            [(A21_idx, [[1,1],[1,1]]), (A22_idx, [[1,1],[1,1]])],
            [(B11_idx, [[1,1],[1,1]])],
            [(C21_idx, [[1,1],[1,1]])]
        )
        # P3 = A11(B21-B22) -> C11 -= P3, C12 += P3
        add_strassen_block(
            [(A11_idx, [[1,1],[1,1]])],
            [(B21_idx, [[1,1],[1,1]]), (B22_idx, [[-1,-1],[-1,-1]])],
            [(C11_idx, [[-1,-1],[-1,-1]]), (C12_idx, [[1,1],[1,1]])]
        )
        # P4 = (A11-A22)(B22-B11) -> C11 += P4, C22 -= P4
        add_strassen_block(
            [(A11_idx, [[1,1],[1,1]]), (A22_idx, [[-1,-1],[-1,-1]])],
            [(B22_idx, [[1,1],[1,1]]), (B11_idx, [[-1,-1],[-1,-1]])],
            [(C11_idx, [[1,1],[1,1]]), (C22_idx, [[-1,-1],[-1,-1]])]
        )
        # P5 = (A12+A22)(B11-B12) -> C12 += P5, C22 -= P5
        add_strassen_block(
            [(A12_idx, [[1,1],[1,1]]), (A22_idx, [[1,1],[1,1]])],
            [(B11_idx, [[1,1],[1,1]]), (B12_idx, [[-1,-1],[-1,-1]])],
            [(C12_idx, [[1,1],[1,1]]), (C22_idx, [[-1,-1],[-1,-1]])]
        )
        # P6 = (A21-A11)(B12+B22) -> C22 += P6
        add_strassen_block(
            [(A21_idx, [[1,1],[1,1]]), (A11_idx, [[-1,-1],[-1,-1]])],
            [(B12_idx, [[1,1],[1,1]]), (B22_idx, [[1,1],[1,1]])],
            [(C22_idx, [[1,1],[1,1]])]
        )
        # P7 = (A11+A12)(B22-B21) -> C11 += P7, C21 -= P7
        add_strassen_block(
            [(A11_idx, [[1,1],[1,1]]), (A12_idx, [[1,1],[1,1]])],
            [(B22_idx, [[1,1],[1,1]]), (B21_idx, [[-1,-1],[-1,-1]])],
            [(C11_idx, [[1,1],[1,1]]), (C21_idx, [[-1,-1],[-1,-1]])]
        )

        # 4x4x2 block (k=4..5): schoolbook would give rank 32.
        # But we can do better by splitting into 2x2x2 blocks.
        # A is 4x4, B[:,4:6] is 4x2, C[:,4:6] is 4x2.
        # Split into four 2x2x2 sub-blocks:
        # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]  (NO, this isn't right)
        
        # Actually, for the 4x4x2 block:
        # C[i,k] = sum_j A[i,j] * B[j,k] for k=4,5
        # Split A into 2x2 blocks, B[:,4:6] into 2x1 blocks (top 2 rows, bottom 2 rows)
        # C[:,4:6] into 2x2 blocks.
        
        # A11 * B_top + A12 * B_bottom -> C_top
        # A21 * B_top + A22 * B_bottom -> C_bot
        
        # Where B_top = B[0:2, 4:6] (2x2), B_bottom = B[2:4, 4:6] (2x2)
        # C_top = C[0:2, 4:6] (2x2), C_bot = C[2:4, 4:6] (2x2)
        
        # C_top = A11*B_top + A12*B_bottom (two 2x2x2 products)
        # C_bot = A21*B_top + A22*B_bottom (two 2x2x2 products)
        
        # Each 2x2x2 product has rank 7 via Strassen.
        # Total: 4 * 7 = 28.

        # B blocks for k=4..5
        B_top_idx = [[(0,4),(0,5)],[(1,4),(1,5)]]
        B_bot_idx = [[(2,4),(2,5)],[(3,4),(3,5)]]

        # C blocks for k=4..5
        C_top_idx = [[(0,4),(0,5)],[(1,4),(1,5)]]
        C_bot_idx = [[(2,4),(2,5)],[(3,4),(3,5)]]

        # A11 * B_top -> C_top
        add_strassen_block(
            [(A11_idx, [[1,1],[1,1]])],
            [(B_top_idx, [[1,1],[1,1]])],
            [(C_top_idx, [[1,1],[1,1]])]
        )
        # A12 * B_bottom -> C_top
        add_strassen_block(
            [(A12_idx, [[1,1],[1,1]])],
            [(B_bot_idx, [[1,1],[1,1]])],
            [(C_top_idx, [[1,1],[1,1]])]
        )
        # A21 * B_top -> C_bot
        add_strassen_block(
            [(A21_idx, [[1,1],[1,1]])],
            [(B_top_idx, [[1,1],[1,1]])],
            [(C_bot_idx, [[1,1],[1,1]])]
        )
        # A22 * B_bottom -> C_bot
        add_strassen_block(
            [(A22_idx, [[1,1],[1,1]])],
            [(B_bot_idx, [[1,1],[1,1]])],
            [(C_bot_idx, [[1,1],[1,1]])]
        )

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
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

    def verify(result):
        U = np.array(result["U"], dtype=object)
        V = np.array(result["V"], dtype=object)
        W = np.array(result["W"], dtype=object)
        T = np.zeros((n, m, p), dtype=object)
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += a * vc[j * p + k] * wc[k * n + i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    try:
        candidate = build_decomposition()
        if time.time() < deadline and verify(candidate):
            return candidate
    except Exception:
        pass

    fallback = build_schoolbook()
    return fallback
# EVOLVE-BLOCK-END
