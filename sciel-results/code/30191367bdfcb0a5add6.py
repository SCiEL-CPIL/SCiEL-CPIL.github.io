import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    deadline = time.time() + 120
    n, m, p = 2, 4, 6
    # Schoolbook as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school

    # Try Strassen-based decomposition
    # A is 2x4, B is 4x6, C is 2x6
    # Split columns of A and rows of B into two halves:
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x3 (top 2 rows, bottom 2 rows)
    # C = A1*B1 + A2*B2
    #
    # This is a 2x2 block matmul with "right side" being 2x3 blocks
    # Strassen for 2x2 blocks: 7 multiplications
    # But B blocks are 2x3, not 2x2
    #
    # Alternative: split into 2x2 and 2x2 and 2x2 for the 6 columns
    # A is 2x4, B is 4x6
    # Split B into three 2x2 blocks: B = [B1 B2 B3]
    # C = A1*B1 + A2*B2 + A3*B3 + A4*B4 ... wait, A is only 2x4
    #
    # Let me try a different approach: direct rank reduction
    # The matmul tensor (2,4,6) can potentially be decomposed with rank < 48
    #
    # Known result: rank of (2,4,6) matmul tensor
    # Using Strassen: view as (2,2) x (2,4) x (4,6) won't work directly
    #
    # Let's try: split A into top row and bottom row (1x4 each)
    # Split B into left 2 cols and right 4 cols... 
    #
    # Actually, the simplest improvement: use the fact that
    # rank(M_(2,4,6)) <= rank(M_(2,2,2)) * 2 * 3 = 7 * 6 = 42
    # by tiling 2x2 Strassen with 3 column groups
    #
    # Or: rank(M_(2,4,6)) <= rank(M_(2,2,6)) * 2
    # rank(M_(2,2,6)): split into three 2x2x2 blocks -> 7*3 = 21
    # So total <= 42
    #
    # Let me construct rank 42 via 2x2 Strassen tiled over 3 column-pairs
    #
    # A: 2x4 = [a11 a12 a13 a14; a21 a22 a23 a24]
    # B: 4x6 = rows r1..r4, cols c1..c6
    #
    # Group columns of B into 3 pairs: (c1,c2), (c3,c4), (c5,c6)
    # For each pair, we have a 4x2 submatrix of B
    # And A is 2x4
    #
    # This doesn't cleanly give 2x2x2 blocks.
    #
    # Better: group rows of B into pairs: (r1,r2), (r3,r4)
    # And columns into pairs: (c1,c2), (c3,c4), (c5,c6)
    # A is 2x4, split into A_left (2x2) and A_right (2x2)
    #
    # C = A_left * B_top + A_right * B_bottom
    # where B_top is 2x6 (rows 1-2 of B), B_bottom is 2x6 (rows 3-4)
    #
    # Now C is sum of two 2x6 products.
    # Each 2x6 product = 2x2 * 2x6, which can be split into three 2x2x2 blocks
    # Each 2x2x2 has rank 7 (Strassen)
    # So each 2x6 has rank <= 21, total <= 42
    #
    # Let me construct this explicitly.
    #
    # For A_left (2x2) * B_top (2x6):
    # B_top = [B1 B2 B3] where each Bj is 2x2
    # A_left * Bj uses Strassen (7 rank-1 terms)
    #
    # Similarly for A_right * B_bottom
    #
    # Total rank: 7*3 + 7*3 = 42

    def strassen_decomposition(A_rows, A_cols, B_rows, B_cols):
        """Return rank-1 terms for A (a x b) * B (b x c) using Strassen if a=b=c=2, else schoolbook."""
        a, b, c = A_rows, A_cols, B_rows, B_cols
        if a == 2 and b == 2 and c == 2:
            # Strassen: 7 rank-1 decompositions
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # M1 = (a+d)(e+h) -> u=[1,0,1,0] for A, v=[1,1,0,0] for B
            # Actually, let me use the standard form.
            # For A (2x2) with entries a11,a12,a21,a22 and B (2x2) with b11,b12,b21,b22:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b21-b22)  -- wait, standard Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-d)(e+f)
            # M7 = (b-d)(g+h)
            # where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            terms = []
            # M1: u = [1,0,1,0] (a+d), v = [1,1,0,0] (e+h), w contributes to C11(+1), C22(+1)
            terms.append(([1,0,1,0], [1,1,0,0], [1,0,0,1]))
            # M2: u = [0,0,1,1] (c+d), v = [1,0,0,0] (e), w to C21(+1), C22(-1)
            terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
            # M3: u = [1,0,0,0] (a), v = [0,1,1,-1] (f-h), w to C12(+1), C22(+1)
            terms.append(([1,0,0,0], [0,1,1,-1], [0,1,0,1]))
            # M4: u = [0,0,0,1] (d), v = [0,0,1,-1] (g-c), w to C11(+1), C21(+1)
            terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
            # M5: u = [1,1,0,0] (a+b), v = [0,0,0,1] (h), w to C11(-1), C12(+1)
            terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
            # M6: u = [0,0,1,-1] (c-d), v = [1,1,0,0] (e+f), w to C22(+1)
            terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
            # M7: u = [0,1,0,-1] (b-d), v = [0,0,1,1] (g+h), w to C11(+1)
            terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
            return terms
        else:
            # Schoolbook
            result = []
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        u = [0]*(a*b)
                        v = [0]*(b*c)
                        w = [0]*(c*a)
                        u[i*b+j] = 1
                        v[j*c+k] = 1
                        w[k*a+i] = 1
                        result.append((u, v, w))
            return result

    # Build rank-42 decomposition
    # A: 2x4, index (i, j) where i in {0,1}, j in {0,1,2,3}
    # B: 4x6, index (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # C: 2x6, index (i, k) where i in {0,1}, k in {0,1,2,3,4,5}
    #
    # Split: A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    #
    # For A_left * B_top: A_left is 2x2, B_top is 2x6
    #   Split B_top into three 2x2 blocks: B_top[0:2, 0:2], B_top[0:2, 2:4], B_top[0:2, 4:6]
    #   Each gives 7 Strassen terms
    # For A_right * B_bottom: A_right is 2x2, B_bottom is 2x6
    #   Similarly 3 blocks of 2x2, each 7 terms
    # Total: 42 terms

    U_terms = []
    V_terms = []
    W_terms = []

    n = 2  # rows of A / C
    m = 4  # cols of A / rows of B
    p = 6  # cols of B / C

    # A_left * B_top: rows of A are 0,1; A cols 0,1; B rows 0,1; B cols 0..5
    # For each 2x2 block of B (cols 0-1, 2-3, 4-5):
    for block in range(3):
        bcol_offset = block * 2
        # Strassen on A_left (2x2) * B_sub (2x2)
        # A indices: (i, j) -> i*m + j = i*4 + j
        # B indices: (j, k) -> j*p + k = j*6 + k
        # C indices: (k, i) -> k*n + i = k*2 + i (per task convention W[k*n+i, r])
        
        terms = strassen_decomposition(2, 2, 2, 2)
        for (u_local, v_local, w_local) in terms:
            u = [0] * (n * m)  # 8
            v = [0] * (m * p)  # 24
            w = [0] * (p * n)  # 12
            
            # u_local is for A_left which is A[i,j] with i in 0,1, j in 0,1
            # u_local index: [a11, a12, a21, a22] = [A[0,0], A[0,1], A[1,0], A[1,1]]
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                u[i * m + j] = u_local[idx]
            
            # v_local is for B_sub which is B[j, k] with j in 0,1, k in bcol_offset, bcol_offset+1
            # v_local index: [b11, b12, b21, b22] = [B[0,bc], B[0,bc+1], B[1,bc], B[1,bc+1]]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                v[j * p + (bcol_offset + k)] = v_local[idx]
            
            # w_local is for C contribution: w_local = [C11, C12, C21, C22]
            # But our C is 2x6, and this block contributes to C[i, bcol_offset + k]
            # w_local index: [C[0,0], C[0,1], C[1,0], C[1,1]] for the 2x2 block
            # In global C: C[i, bcol_offset + k]
            # Task convention: W[k*n + i, r]
            for idx in range(4):
                i = idx // 2
                k = idx % 2
                w[(bcol_offset + k) * n + i] = w_local[idx]
            
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)

    # A_right * B_bottom: A cols 2,3; B rows 2,3; B cols 0..5
    for block in range(3):
        bcol_offset = block * 2
        arow_offset = 2  # A_right uses A[0,2], A[0,3], A[1,2], A[1,3]
        brow_offset = 2  # B_bottom uses B[2,0..5], B[3,0..5]
        
        terms = strassen_decomposition(2, 2, 2, 2)
        for (u_local, v_local, w_local) in terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # u_local: A_right[i,j] where i in 0,1, j in 0,1 corresponds to A[i, 2+j]
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                u[i * m + (arow_offset + j)] = u_local[idx]
            
            # v_local: B_bottom[j,k] where j in 0,1, k in 0,1 corresponds to B[2+j, bcol_offset+k]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                v[(brow_offset + j) * p + (bcol_offset + k)] = v_local[idx]
            
            # w_local: C[i, bcol_offset+k]
            for idx in range(4):
                i = idx // 2
                k = idx % 2
                w[(bcol_offset + k) * n + i] = w_local[idx]
            
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)

    rank = len(U_terms)
    if rank < best_rank:
        best = {"U": U_terms, "V": V_terms, "W": W_terms}
        best_rank = rank

    if time.time() > deadline - 5:
        return best

    return best
