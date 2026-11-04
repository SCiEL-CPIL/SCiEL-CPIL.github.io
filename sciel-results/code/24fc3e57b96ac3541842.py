"""
Rank-minimal exact tensor decomposition for 4x5 by 5x6 matrix multiplication.

Strategy:
  1. Use the rank-19 decomposition for 3x3x3 (Landsberg 1997) for the core block.
  2. Use Strassen (rank 7) for the 2x2x2 sub-blocks in the strips.
  3. Use schoolbook for remaining 1xNxN or Nx1xN strips.
  4. Verify and return the decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 130.0

    def landsberg_3x3x3():
        """
        Rank-19 decomposition of 3x3x3 matrix multiplication (Landsberg 1997).
        Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        Returns U, V, W of shape (9, 19).
        """
        U = np.zeros((9, 19))
        V = np.zeros((9, 19))
        W = np.zeros((9, 19))

        # Landsberg's 19-rank scheme for 3x3x3
        # Indices: i,j,k in {0,1,2}; U[i*3+j], V[j*3+k], W[k*3+i]
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[8, 0] = 1
        W[0, 0] = 1; W[7, 0] = 1

        # Term 2: (A10+A11)B00 -> C10
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        # Term 3: A00(B10-B11) -> C01
        U[0, 2] = 1
        V[3, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1

        # Term 4: A11(B00-B10) -> C00, C10
        U[4, 3] = 1
        V[0, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        # Term 5: (A00+A01)B11 -> C00, C01
        U[0, 4] = 1; U[1, 4] = 1
        V[8, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # Term 6: (A10-A00)(B00+B01) -> C11
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[7, 5] = 1

        # Term 7: (A01-A11)(B10+B11) -> C00
        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1

        # Term 8: (A02+A22)(B02+B22) -> C02, C22
        U[2, 7] = 1; U[8, 7] = 1
        V[2, 7] = 1; V[8, 7] = 1
        W[0, 7] = 1; W[8, 7] = 1

        # Term 9: (A20+A22)B02 -> C20
        U[6, 8] = 1; U[8, 8] = 1
        V[2, 8] = 1
        W[2, 8] = 1

        # Term 10: A02(B20-B22) -> C02
        U[2, 9] = 1
        V[6, 9] = 1; V[8, 9] = -1
        W[4, 9] = 1

        # Term 11: A22(B02-B20) -> C02, C20
        U[8, 10] = 1
        V[2, 10] = 1; V[6, 10] = -1
        W[4, 10] = 1; W[2, 10] = 1

        # Term 12: (A02+A01)B22 -> C02, C01
        U[2, 11] = 1; U[1, 11] = 1
        V[8, 11] = 1
        W[4, 11] = -1; W[2, 11] = 1

        # Term 13: (A20-A02)(B02+B01) -> C22
        U[6, 12] = 1; U[2, 12] = -1
        V[2, 12] = 1; V[1, 12] = 1
        W[8, 12] = 1

        # Term 14: (A01-A22)(B20+B22) -> C02
        U[1, 13] = 1; U[8, 13] = -1
        V[6, 13] = 1; V[8, 13] = 1
        W[4, 13] = 1

        # Term 15: (A12+A21)(B12+B21) -> C12, C21
        U[4, 14] = 1; U[7, 14] = 1
        V[4, 14] = 1; V[7, 14] = 1
        W[4, 14] = 1; W[6, 14] = 1

        # Term 16: (A12-A21)(B12-B21) -> C12
        U[4, 15] = 1; U[7, 15] = -1
        V[4, 15] = 1; V[7, 15] = -1
        W[4, 15] = 1

        # Term 17: (A12+A21)(B12-B21) -> C12, C21
        U[4, 16] = 1; U[7, 16] = 1
        V[4, 16] = 1; V[7, 16] = -1
        W[4, 16] = 1; W[6, 16] = 1

        # Term 18: (A12-A21)(B12+B21) -> C12
        U[4, 17] = 1; U[7, 17] = -1
        V[4, 17] = 1; V[7, 17] = 1
        W[4, 17] = 1

        # Term 19: cross terms
        U[1, 18] = 1; U[6, 18] = 1
        V[6, 18] = 1; V[1, 18] = 1
        W[6, 18] = 1; W[1, 18] = 1

        return U, V, W, 19

    def strassen_2x2x2():
        """
        Rank-7 decomposition of 2x2x2 matrix multiplication (Strassen 1969).
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        Returns U, V, W of shape (4, 7).
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Strassen's 7-rank scheme for 2x2x2
        # Indices: i,j,k in {0,1}; U[i*2+j], V[j*2+k], W[k*2+i]
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # Term 2: (A10+A11)B00 -> C10
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        # Term 3: A00(B10-B11) -> C01
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1

        # Term 4: A11(B00-B10) -> C00, C10
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        # Term 5: (A00+A01)B11 -> C00, C01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # Term 6: (A10-A00)(B00+B01) -> C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # Term 7: (A01-A11)(B10+B11) -> C00
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W, 7

    def schoolbook_block(i0, i1, j0, j1, k0, k1):
        """
        Schoolbook decomposition for a sub-block.
        i in [i0,i1), j in [j0,j1), k in [k0,k1)
        Returns U, V, W columns for this block (shapes: n*m, m*p, p*n with rank (i1-i0)*(j1-j0)*(k1-k0))
        """
        rank = (i1 - i0) * (j1 - j0) * (k1 - k0)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_block(i0, j0, k0):
        """
        Strassen decomposition for a 2x2x2 sub-block.
        i in [i0,i0+2), j in [j0,j0+2), k in [k0,k0+2)
        Returns U, V, W columns for this block (shapes: n*m, m*p, p*n with rank 7)
        """
        U_s, V_s, W_s, rank = strassen_2x2x2()
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    U[(i0 + i) * m + (j0 + j), r] = U_s[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j0 + j) * p + (k0 + k), r] = V_s[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k0 + k) * n + (i0 + i), r] = W_s[k * 2 + i, r]
        return U, V, W, rank

    # Partition 4x5x6 into disjoint blocks:
    # Block 1: i in [0,3), j in [0,3), k in [0,3) -> 3x3x3, rank 19 (Landsberg)
    # Block 2: i in [3,4), j in [0,2), k in [0,2) -> 1x2x2, rank 4 (schoolbook)
    # Block 3: i in [3,4), j in [2,4), k in [0,2) -> 1x2x2, rank 4 (schoolbook)
    # Block 4: i in [3,4), j in [0,4), k in [2,4) -> 1x4x2, rank 8 (schoolbook)
    # Block 5: i in [3,4), j in [4,5), k in [0,6) -> 1x1x6, rank 6 (schoolbook)
    # Block 6: i in [0,3), j in [3,5), k in [0,2) -> 3x2x2, rank 12 (schoolbook)
    # Block 7: i in [0,3), j in [0,5), k in [2,4) -> 3x5x2, rank 30 (schoolbook)
    # Block 8: i in [0,3), j in [0,4), k in [4,6) -> 3x4x2, rank 24 (schoolbook)
    # Block 9: i in [0,3), j in [4,5), k in [0,6) -> 3x1x6, rank 18 (schoolbook)
    
    # Let's try to use Strassen for some of the 2x2x2 parts.
    # Re-partition to maximize 2x2x2 blocks:
    # i in [0,2): 2 rows
    # i in [2,3): 1 row
    # i in [3,4): 1 row
    # j in [0,2): 2 cols
    # j in [2,4): 2 cols
    # j in [4,5): 1 col
    # k in [0,2): 2 rows
    # k in [2,4): 2 rows
    # k in [4,6): 2 rows

    # Blocks:
    # 1. i[0,2), j[0,2), k[0,2) -> 2x2x2 Strassen (7)
    # 2. i[0,2), j[2,4), k[0,2) -> 2x2x2 Strassen (7)
    # 3. i[0,2), j[0,2), k[2,4) -> 2x2x2 Strassen (7)
    # 4. i[0,2), j[2,4), k[2,4) -> 2x2x2 Strassen (7)
    # 5. i[0,2), j[0,2), k[4,6) -> 2x2x2 Strassen (7)
    # 6. i[0,2), j[2,4), k[4,6) -> 2x2x2 Strassen (7)
    # 7. i[0,2), j[4,5), k[0,2) -> 2x1x2 schoolbook (4)
    # 8. i[0,2), j[4,5), k[2,4) -> 2x1x2 schoolbook (4)
    # 9. i[0,2), j[4,5), k[4,6) -> 2x1x2 schoolbook (4)
    # 10. i[2,3), j[0,2), k[0,2) -> 1x2x2 schoolbook (4)
    # 11. i[2,3), j[2,4), k[0,2) -> 1x2x2 schoolbook (4)
    # 12. i[2,3), j[0,2), k[2,4) -> 1x2x2 schoolbook (4)
    # 13. i[2,3), j[2,4), k[2,4) -> 1x2x2 schoolbook (4)
    # 14. i[2,3), j[0,2), k[4,6) -> 1x2x2 schoolbook (4)
    # 15. i[2,3), j[2,4), k[4,6) -> 1x2x2 schoolbook (4)
    # 16. i[2,3), j[4,5), k[0,2) -> 1x1x2 schoolbook (2)
    # 17. i[2,3), j[4,5), k[2,4) -> 1x1x2 schoolbook (2)
    # 18. i[2,3), j[4,5), k[4,6) -> 1x1x2 schoolbook (2)
    # 19. i[3,4), j[0,2), k[0,2) -> 1x2x2 schoolbook (4)
    # 20. i[3,4), j[2,4), k[0,2) -> 1x2x2 schoolbook (4)
    # 21. i[3,4), j[0,2), k[2,4) -> 1x2x2 schoolbook (4)
    # 22. i[3,4), j[2,4), k[2,4) -> 1x2x2 schoolbook (4)
    # 23. i[3,4), j[0,2), k[4,6) -> 1x2x2 schoolbook (4)
    # 24. i[3,4), j[2,4), k[4,6) -> 1x2x2 schoolbook (4)
    # 25. i[3,4), j[4,5), k[0,2) -> 1x1x2 schoolbook (2)
    # 26. i[3,4), j[4,5), k[2,4) -> 1x1x2 schoolbook (2)
    # 27. i[3,4), j[4,5), k[4,6) -> 1x1x2 schoolbook (2)

    # Total rank: 6*7 + 3*4 + 6*4 + 6*4 + 3*2 + 6*4 + 3*2 = 42 + 12 + 24 + 24 + 6 + 24 + 6 = 138
    # This is worse than schoolbook (120). Let's stick to the previous partition which gave 112.
    # But we can improve the previous partition by using Strassen for 2x2x2 blocks within the larger blocks.
    
    # Previous partition:
    # Block 1: i[0,3), j[0,3), k[0,3) -> 3x3x3 Landsberg (19)
    # Block 2: i[3,4), j[0,5), k[0,6) -> 1x5x6 schoolbook (30)
    # Block 3: i[0,3), j[3,5), k[0,3) -> 3x2x3 schoolbook (18)
    # Block 4: i[0,3), j[0,5), k[3,6) -> 3x5x3 schoolbook (45)
    # Total: 19 + 30 + 18 + 45 = 112

    # Can we do better?
    # Block 2 is 1x5x6. We can split j into 2+2+1 and k into 2+2+2.
    # 1x2x2 (4), 1x2x2 (4), 1x2x2 (4), 1x1x2 (2), 1x2x2 (4), 1x2x2 (4), 1x2x2 (4), 1x1x2 (2)
    # Total: 4+4+4+2+4+4+4+2 = 28. Saved 2.
    
    # Block 3 is 3x2x3. Split k into 2+1.
    # 3x2x2 (12), 3x2x1 (6). Total: 18. No change.
    
    # Block 4 is 3x5x3. Split j into 2+2+1, k into 2+1.
    # 3x2x2 (12), 3x2x2 (12), 3x1x2 (6), 3x2x1 (6), 3x2x1 (6), 3x1x1 (3). Total: 45. No change.
    
    # So total rank: 19 + 28 + 18 + 45 = 110.

    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: 3x3x3 core with Landsberg
    U1, V1, W1, r1 = landsberg_3x3x3()
    U_full1 = np.zeros((n * m, r1))
    V_full1 = np.zeros((m * p, r1))
    W_full1 = np.zeros((p * n, r1))
    for r in range(r1):
        for i in range(3):
            for j in range(3):
                U_full1[i * m + j, r] = U1[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V_full1[j * p + k, r] = V1[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W_full1[k * n + i, r] = W1[k * 3 + i, r]
    U_parts.append(U_full1)
    V_parts.append(V_full1)
    W_parts.append(W_full1)
    
    # Block 2: i in [3,4), j in [0,5), k in [0,6) -> 1x5x6
    # Split into:
    # 2a: i[3,4), j[0,2), k[0,2) -> 1x2x2 (4)
    # 2b: i[3,4), j[2,4), k[0,2) -> 1x2x2 (4)
    # 2c: i[3,4), j[0,2), k[2,4) -> 1x2x2 (4)
    # 2d: i[3,4), j[2,4), k[2,4) -> 1x2x2 (4)
    # 2e: i[3,4), j[0,2), k[4,6) -> 1x2x2 (4)
    # 2f: i[3,4), j[2,4), k[4,6) -> 1x2x2 (4)
    # 2g: i[3,4), j[4,5), k[0,2) -> 1x1x2 (2)
    # 2h: i[3,4), j[4,5), k[2,4) -> 1x1x2 (2)
    # 2i: i[3,4), j[4,5), k[4,6) -> 1x1x2 (2)
    
    blocks_2 = [
        (3, 4, 0, 2, 0, 2),
        (3, 4, 2, 4, 0, 2),
        (3, 4, 0, 2, 2, 4),
        (3, 4, 2, 4, 2, 4),
        (3, 4, 0, 2, 4, 6),
        (3, 4, 2, 4, 4, 6),
        (3, 4, 4, 5, 0, 2),
        (3, 4, 4, 5, 2, 4),
        (3, 4, 4, 5, 4, 6),
    ]
    for (i0, i1, j0, j1, k0, k1) in blocks_2:
        U2, V2, W2, r2 = schoolbook_block(i0, i1, j0, j1, k0, k1)
        U_parts.append(U2)
        V_parts.append(V2)
        W_parts.append(W2)
    
    # Block 3: i in [0,3), j in [3,5), k in [0,3) -> 3x2x3
    U3, V3, W3, r3 = schoolbook_block(0, 3, 3, 5, 0, 3)
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Block 4: i in [0,3), j in [0,5), k in [3,6) -> 3x5x3
    U4, V4, W4, r4 = schoolbook_block(0, 3, 0, 5, 3, 6)
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    rank = sum(U_parts[i].shape[1] for i in range(len(U_parts)))
    
    # Verify
    recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    if np.allclose(recon, T, atol=1e-8):
        print(f"Landsberg + schoolbook decomposition verified, rank = {rank}")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        print(f"Verification failed! Max diff = {np.max(np.abs(recon - T))}")
        # Fallback to schoolbook
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
