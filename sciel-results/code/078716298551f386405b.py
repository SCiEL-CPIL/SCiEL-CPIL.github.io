import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # 42-rank Strassen-like decomposition:
    # 2x4 by 4x6 -> 2x6
    # Split A (2x4) into 2x2 blocks: A = [[A00, A01],[A10, A11]]
    # Split B (4x6) into 2x3 blocks: B = [[B00,B01,B02],[B10,B11,B12]]
    # C (2x6) = A*B, split into 2x3 blocks: C = [[C00,C01,C02],[C10,C11,C12]]
    # Each C_ij is 1x2
    # Use Strassen-like products for the 2x2 block structure
    # M1 = A00*(B00+B10)
    # M2 = (A00+A01)*B10
    # M3 = (A10+A11)*B00
    # M4 = A11*(B10+B11)
    # M5 = (A00+A11)*(B00+B11)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A00-A10)*(B00+B10)
    # But B has 2 rows and 3 cols of 2-col blocks, so B10 is 2x2, B11 is 2x2, etc.
    # Wait, B is 4x6. Split into 2x3 blocks of size 2x2.
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x2
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12

    # This is essentially 2x(2x3) matrix mult = 2*2*3*2*2 = 48 products (schoolbook)
    # Strassen on the 2x2 block structure (treating each 2x2 block as a scalar):
    # For 2x2 by 2x3:
    # R(2,2,3): we need to multiply 2x2 by 2x3.
    # Can use Strassen for the 2x2 part and then handle the 3 columns.
    # Actually, let me just use the known rank 42 decomposition.

    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Indexing:
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x6: B[j, k] where j in {0,1,2,3}, k in {0,...,5}
    # C is 2x6: C[i, k] where i in {0,1}, k in {0,...,5}
    # U has shape (n*m, rank) = (8, 42): U[i*m + j, r] = coefficient for A[i,j] in product r
    # V has shape (m*p, rank) = (24, 42): V[j*p + k, r] = coefficient for B[j,k] in product r
    # W has shape (p*n, rank) = (12, 42): W[k*n + i, r] = coefficient for product r in C[i,k]

    # Split into 2x2 blocks of A (each 1x2) and 2x3 blocks of B (each 2x2)
    # A00 = A[0,0:2], A01 = A[0,2:4]
    # A10 = A[1,0:2], A11 = A[1,2:4]
    # B00 = B[0:2,0:2], B01 = B[0:2,2:4], B02 = B[0:2,4:6]
    # B10 = B[2:4,0:2], B11 = B[2:4,2:4], B12 = B[2:4,4:6]

    # Products (each is 1x2 x 2x2 = 1x2):
    # M1 = A00 * (B00 + B10)  -> contributes to C00, C10
    # M2 = (A00 + A01) * B10  -> contributes to C00
    # M3 = (A10 + A11) * B00  -> contributes to C10
    # M4 = A11 * (B10 + B11)  -> contributes to C11
    # M5 = (A00 + A11) * (B00 + B11) -> contributes to C00, C01, C10, C11
    # M6 = (A01 - A11) * (B10 + B11) -> contributes to C01
    # M7 = (A00 - A10) * (B00 + B10) -> contributes to C00
    # M8 = A00 * B02  -> C02
    # M9 = A01 * B12  -> C02
    # M10 = A10 * B02 -> C12
    # M11 = A11 * B12 -> C12

    # Wait, this gives 11 products for the "main" part, but I need to handle all 6 output columns.
    # Let me reconsider. The 2x2 block Strassen applies to the row/column structure.
    # For each pair of column-blocks, I can use a 2x2 Strassen.
    # Actually, the standard approach: treat it as (2x2)*(2x3) where the 2x3 has three 2x2 blocks.
    # R(2,2,3) can be done in 2*R(2,2,2) + something? No.
    # Let me just use the explicit 42-rank construction from the current program, which works.

    # The current program's steps define 42 products. Let me just use them directly but cleaner.
    # Step 1: 6 products
    # Step 2: 6 products
    # Step 3: 6 products
    # Step 4: 6 products
    # Step 5: 6 products
    # Step 6: 6 products
    # Step 7: 6 products
    # Total: 42

    # Let me re-derive from scratch using a clean block approach.
    # A = [[a,b,c,d],[e,f,g,h]] (2x4)
    # B = [[p,q,r,s,t,u],[v,w,x,y,z,aa],[bb,cc,dd,ee,ff,gg],[hh,ii,jj,kk,ll,mm]] (4x6)
    # C = A*B (2x6)

    # C[0,0] = ap+bv+cbh+dh
    # C[0,1] = aq+bw+cci+di
    # C[0,2] = ar+bx+cdj+dk
    # C[0,3] = as+by+cdk+dl
    # C[0,4] = at+bz+cdl+dm
    # C[0,5] = au+bba+cdd+dn
    # C[1,0] = ep+fv+gbh+hh
    # C[1,1] = eq+fw+cci+hi
    # C[1,2] = er+fx+cdj+hk
    # C[1,3] = es+fy+cdk+hl
    # C[1,4] = et+fz+cdl+hm
    # C[1,5] = eu+fba+cdd+hn

    # This is getting complex. Let me just use the current program's working decomposition.
    # The current program achieves rank 42 and scores 1.1429.

    # I'll just copy the working decomposition from the current program.
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]

    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c

        y_coeffs = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            r_block = j % 2
            c_block = k % 3
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))

        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0),
            (x1_coeffs, y_coeffs[3], 0),
            (x0_coeffs, y_coeffs[1], 1),
            (x1_coeffs, y_coeffs[4], 1),
            (x0_coeffs, y_coeffs[2], 2),
            (x1_coeffs, y_coeffs[5], 2),
        ]

        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c

            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1:
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2:
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3:
                    i_C = 1
                    k_C = 3 + z_idx

                W[k_C * n + i_C, col] += sign

            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
