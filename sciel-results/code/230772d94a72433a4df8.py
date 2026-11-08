"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Deadline guard: return schoolbook if we run out of time
    start_time = time.time()
    DEADLINE = 130.0  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 96

    # Strassen's rank-7 decomposition for 2x2x2
    # For C = A*B where A is 2x2, B is 2x2, C is 2x2
    # Strassen uses 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For each 2x2x2 block, we need to express the 7 Strassen terms
    # as rank-1 contributions to U, V, W.
    #
    # The matrix multiplication tensor for 2x2x2: T[i,j,k] where
    # i indexes rows of A (0,1), j indexes cols of A / rows of B (0,1),
    # k indexes cols of B (0,1).
    #
    # In the global 4x4x6 problem:
    # A is 4x4, B is 4x6, C is 6x4 (C = B*A, with W[k*n+i, r] convention)
    #
    # Wait, let me re-read the convention. W[k*n+i, r] where:
    # - i is row index of A (0..n-1)
    # - j is col index of A / row index of B (0..m-1)  
    # - k is col index of B (0..p-1)
    # - The output is C[k, i] = sum_j A[i,j] * B[j,k]
    # - W is indexed by [k*n+i]
    #
    # So U has shape (n*m, rank) = (16, rank), indexed by [i*m+j]
    # V has shape (m*p, rank) = (24, rank), indexed by [j*p+k]
    # W has shape (p*n, rank) = (24, rank), indexed by [k*n+i]
    #
    # For a rank-1 term c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] contributes
    # to the product.
    #
    # Block decomposition: split A into 2x2 blocks of size 2x2, B into 2x3 blocks,
    # C into 3x2 blocks.
    #
    # A is 4x4: blocks A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # B is 4x6: blocks B[0:2, 0:3], B[0:2, 3:6], B[2:4, 0:3], B[2:4, 3:6]
    # C is 6x4: blocks C[0:3, 0:2], C[0:3, 2:4], C[3:6, 0:2], C[3:6, 2:4]
    #
    # C[bi, bj] = sum over intermediate blocks: A[bi, bk] * B[bk, bj]
    #
    # For C[0:3, 0:2] (top-left 3x2 block of C):
    #   = A[0:2, 0:2] * B[0:2, 0:3] + A[0:2, 2:4] * B[2:4, 0:3]
    #
    # Each 2x2 * 2x3 = 3x2 product can be decomposed.
    #
    # Actually, let me think about this differently. We can decompose the 4x4x6
    # into 12 sub-problems of 2x2x2 by splitting further.
    #
    # A: 4x4 -> 4 blocks of 2x2 (rows 0-1,2-3 x cols 0-1,2-3)
    # B: 4x6 -> 6 blocks of 2x2 (rows 0-1,2-3 x cols 0-1,2-3,4-5)
    # C: 6x4 -> 6 blocks of 2x2 (rows 0-1,2-3,4-5 x cols 0-1,2-3)
    #
    # C[br, bc] = sum over middle index: A[br, bm] * B[bm, bc]
    # where br in {0,1,2,3} (row block of C = col block of B), 
    # bm in {0,1} (middle index block), bc in {0,1,2} (col block of C = row block of A)
    #
    # Wait, I need to be more careful. Let's use:
    # A has row blocks ar in {0,1} (2 rows each) and col blocks ac in {0,1} (2 cols each)
    # B has row blocks br in {0,1} (2 rows each) and col blocks bc in {0,1,2} (2 cols each)
    # C has row blocks cr in {0,1,2} (2 rows each) and col blocks cc in {0,1} (2 cols each)
    #
    # C[cr, cc] = sum over middle: A[cc, bm] * B[bm, cr]  where bm in {0,1}
    #
    # So for each (cr, cc) pair, we have two 2x2x2 sub-problems:
    # C[cr, cc] = A[cc, 0] * B[0, cr] + A[cc, 1] * B[1, cr]
    #
    # Total sub-problems: 3 (cr) * 2 (cc) * 2 (bm) = 12 sub-problems of 2x2x2
    #
    # Each 2x2x2 can be done with rank 7 (Strassen), so total rank = 12 * 7 = 84
    #
    # Let's implement Strassen for each 2x2x2 sub-problem.
    #
    # For a 2x2x2 sub-problem with:
    # A_sub is 2x2 (rows from A[cc, bm], cols are the 2 rows of that block)
    # B_sub is 2x2 (rows from B[bm, cr], cols are the 2 cols of that block)
    # C_sub is 2x2 (rows from C[cr, cc])
    #
    # Strassen's 7 terms for computing C_sub = A_sub * B_sub:
    # Let A_sub = [[a00, a01], [a10, a11]]
    # Let B_sub = [[b00, b01], [b10, b11]]
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Each M is a product of a linear combination of A entries and a linear
    # combination of B entries. So each M contributes one rank-1 term to the
    # tensor decomposition.
    #
    # For term M1: (a00+a11)(b00+b11)
    #   U entries: a00 and a11 get coefficient 1
    #   V entries: b00 and b11 get coefficient 1
    #   W entries: C00 gets +1, C11 gets +1
    #
    # Let me map the local indices to global indices.
    #
    # For sub-problem (cc, bm, cr):
    # A_sub[i, j] corresponds to global A[cc*2+i, bm*2+j]
    #   In U: index = (cc*2+i)*m + (bm*2+j) = (cc*2+i)*4 + (bm*2+j)
    # B_sub[i, j] corresponds to global B[bm*2+i, cr*2+j]
    #   In V: index = (bm*2+i)*p + (cr*2+j) = (bm*2+i)*6 + (cr*2+j)
    # C_sub[i, j] corresponds to global C[cr*2+i, cc*2+j]
    #   In W: index = (cr*2+i)*n + (cc*2+j) = (cr*2+i)*4 + (cc*2+j)

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for cc in range(2):  # col block of A / col block of C
        for bm in range(2):  # middle index block
            for cr in range(3):  # row block of B / row block of C
                # Local indices:
                # A_sub[i,j]: global A[cc*2+i, bm*2+j], U index = (cc*2+i)*4 + (bm*2+j)
                # B_sub[i,j]: global B[bm*2+i, cr*2+j], V index = (bm*2+i)*6 + (cr*2+j)
                # C_sub[i,j]: global C[cr*2+i, cc*2+j], W index = (cr*2+i)*4 + (cc*2+j)

                # Define local-to-global index mappings
                def u_idx(i, j):
                    return (cc * 2 + i) * 4 + (bm * 2 + j)

                def v_idx(i, j):
                    return (bm * 2 + i) * 6 + (cr * 2 + j)

                def w_idx(i, j):
                    return (cr * 2 + i) * 4 + (cc * 2 + j)

                # Strassen's 7 terms
                # M1 = (a00+a11)(b00+b11)
                U[u_idx(0, 0), col] += 1
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 0), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] += 1  # C00
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M2 = (a10+a11)b00
                U[u_idx(1, 0), col] += 1
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 0), col] += 1
                W[w_idx(1, 0), col] += 1  # C10
                col += 1

                # M3 = a00(b10-b11)
                U[u_idx(0, 0), col] += 1
                V[v_idx(1, 0), col] += 1
                V[v_idx(1, 1), col] -= 1
                W[w_idx(0, 1), col] += 1  # C01
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M4 = a11(b01-b10)
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 1), col] += 1
                V[v_idx(1, 0), col] -= 1
                W[w_idx(0, 0), col] += 1  # C00
                W[w_idx(1, 0), col] += 1  # C10
                col += 1

                # M5 = (a00+a01)b11
                U[u_idx(0, 0), col] += 1
                U[u_idx(0, 1), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] -= 1  # C00
                W[w_idx(0, 1), col] += 1  # C01
                col += 1

                # M6 = (a10-a00)(b00+b10)
                U[u_idx(1, 0), col] += 1
                U[u_idx(0, 0), col] -= 1
                V[v_idx(0, 0), col] += 1
                V[v_idx(1, 0), col] += 1
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M7 = (a01-a11)(b01+b11)
                U[u_idx(0, 1), col] += 1
                U[u_idx(1, 1), col] -= 1
                V[v_idx(0, 1), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] += 1  # C00
                W[w_idx(0, 1), col] += 1  # C01  -- wait, let me recheck
                # C00 = M1+M4-M5+M7, so M7 contributes +1 to C00
                # C11 = M1-M2+M3+M6, M7 does NOT contribute to C11
                # Let me recheck: C01 = M3+M5, C10 = M2+M4
                # M7 only appears in C00.
                # Actually wait: let me re-derive.
                # C00 = M1+M4-M5+M7
                # C01 = M3+M5
                # C10 = M2+M4
                # C11 = M1-M2+M3+M6
                # So M7 only contributes to C00.
                # I made an error above. Let me fix.
                col += 1

    # Fix the M7 W entry: should only be C00, not C01
    # The last column (col-1) is M7 for the last sub-problem
    # Let me just redo this more carefully by resetting and rebuilding.

    # Actually, let me just verify and fix by rebuilding cleanly.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for cc in range(2):
        for bm in range(2):
            for cr in range(3):
                def u_idx(i, j):
                    return (cc * 2 + i) * 4 + (bm * 2 + j)

                def v_idx(i, j):
                    return (bm * 2 + i) * 6 + (cr * 2 + j)

                def w_idx(i, j):
                    return (cr * 2 + i) * 4 + (cc * 2 + j)

                # M1 = (a00+a11)(b00+b11)
                U[u_idx(0, 0), col] += 1
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 0), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] += 1  # C00
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M2 = (a10+a11)b00
                U[u_idx(1, 0), col] += 1
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 0), col] += 1
                W[w_idx(1, 0), col] += 1  # C10
                col += 1

                # M3 = a00(b10-b11)
                U[u_idx(0, 0), col] += 1
                V[v_idx(1, 0), col] += 1
                V[v_idx(1, 1), col] -= 1
                W[w_idx(0, 1), col] += 1  # C01
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M4 = a11(b01-b10)
                U[u_idx(1, 1), col] += 1
                V[v_idx(0, 1), col] += 1
                V[v_idx(1, 0), col] -= 1
                W[w_idx(0, 0), col] += 1  # C00
                W[w_idx(1, 0), col] += 1  # C10
                col += 1

                # M5 = (a00+a01)b11
                U[u_idx(0, 0), col] += 1
                U[u_idx(0, 1), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] -= 1  # C00
                W[w_idx(0, 1), col] += 1  # C01
                col += 1

                # M6 = (a10-a00)(b00+b10)
                U[u_idx(1, 0), col] += 1
                U[u_idx(0, 0), col] -= 1
                V[v_idx(0, 0), col] += 1
                V[v_idx(1, 0), col] += 1
                W[w_idx(1, 1), col] += 1  # C11
                col += 1

                # M7 = (a01-a11)(b01+b11)
                U[u_idx(0, 1), col] += 1
                U[u_idx(1, 1), col] -= 1
                V[v_idx(0, 1), col] += 1
                V[v_idx(1, 1), col] += 1
                W[w_idx(0, 0), col] += 1  # C00
                col += 1

    # Verify correctness
    if time.time() - start_time > DEADLINE:
        # Return schoolbook as fallback
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s), dtype=float)
        V_s = np.zeros((m * p, rank_s), dtype=float)
        W_s = np.zeros((p * n, rank_s), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c] = 1
                    V_s[j * p + k, c] = 1
                    W_s[k * n + i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Verify with random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = B_test @ A_test  # C[k, i] = sum_j B[k, j] * A[i, j]
    # Wait, the convention is W[k*n+i, r], and C = B * A? Let me check.
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This gives sum_{i,j,k,r} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For this to equal (B*A)[k,i] = sum_j B[k,j]*A[i,j], we need:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = B[k,j]*A[i,j] summed over j
    # So the product computes C[k,i] = sum_j A[i,j]*B[k,j]
    # That's C = B * A (where B is m x p and A is n x m, so B*A is p x n)
    # C_expected[k, i] = sum_j B[k, j] * A[i, j]
    
    # Compute via decomposition
    C_decomp = np.zeros((p, n))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_decomp[k, i] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check
    max_err = np.max(np.abs(C_decomp - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s), dtype=float)
        V_s = np.zeros((m * p, rank_s), dtype=float)
        W_s = np.zeros((p * n, rank_s), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c] = 1
                    V_s[j * p + k, c] = 1
                    W_s[k * n + i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
