import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110

    # Strategy: Block decomposition with Strassen for 2x2x2 blocks.
    # Partition dimensions:
    # A (4x5): Rows 4=2+2, Cols 5=2+2+1
    #   Blocks: A11(2x2), A12(2x2), A13(2x1)
    #           A21(2x2), A22(2x2), A23(2x1)
    # B (5x5): Rows 5=2+2+1, Cols 5=2+2+1
    #   Blocks: B11(2x2), B12(2x2), B13(2x1)
    #           B21(2x2), B22(2x2), B23(2x1)
    #           B31(1x2), B32(1x2), B33(1x1)
    # C (4x5): Rows 4=2+2, Cols 5=2+2+1
    #   Blocks: C11(2x2), C12(2x2), C13(2x1)
    #           C21(2x2), C22(2x2), C23(2x1)
    
    # Terms in C = A * B:
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33

    # Rank contributions:
    # 2x2 * 2x2 -> 2x2: Use Strassen (Rank 7) instead of Schoolbook (Rank 8).
    #   There are 6 such terms: (11,11), (12,21), (21,11), (21,12), (22,22), (11,12)... wait.
    #   Let's list the 2x2 * 2x2 terms:
    #   1. A11(2x2) * B11(2x2) -> C11(2x2)
    #   2. A12(2x2) * B21(2x2) -> C11(2x2)
    #   3. A11(2x2) * B12(2x2) -> C12(2x2)
    #   4. A12(2x2) * B22(2x2) -> C12(2x2)
    #   5. A21(2x2) * B11(2x2) -> C21(2x2)
    #   6. A22(2x2) * B21(2x2) -> C21(2x2)
    #   7. A21(2x2) * B12(2x2) -> C22(2x2)
    #   8. A22(2x2) * B22(2x2) -> C22(2x2)
    #   Total 8 terms of type 2x2*2x2.
    #   Schoolbook rank for these: 8 * 8 = 64.
    #   Strassen rank for these: 8 * 7 = 56.
    #   Savings: 8.

    # Other terms (Schoolbook):
    # 2x2 * 2x1 -> 2x1:
    #   1. A11(2x2) * B13(2x1) -> C13(2x1) : Rank 2*2*1 = 4
    #   2. A12(2x2) * B23(2x1) -> C13(2x1) : Rank 4
    #   3. A21(2x2) * B13(2x2? No B13 is 2x1) -> C23(2x1) : Rank 4
    #   4. A22(2x2) * B23(2x1) -> C23(2x1) : Rank 4
    #   Total: 16

    # 2x1 * 1x2 -> 2x2:
    #   1. A13(2x1) * B31(1x2) -> C11(2x2) : Rank 2*1*2 = 4
    #   2. A13(2x1) * B32(1x2) -> C12(2x2) : Rank 4
    #   3. A23(2x1) * B31(1x2) -> C21(2x2) : Rank 4
    #   4. A23(2x1) * B32(1x2) -> C22(2x2) : Rank 4
    #   Total: 16

    # 2x1 * 1x1 -> 2x1:
    #   1. A13(2x1) * B33(1x1) -> C13(2x1) : Rank 2
    #   2. A23(2x1) * B33(1x1) -> C23(2x1) : Rank 2
    #   Total: 4

    # Total Rank = 56 (Strassen parts) + 16 + 16 + 4 = 92.
    # This is an improvement over 100.

    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Helper to add Strassen 2x2x2 block
    # A indices: rows ar..ar+1, cols ac..ac+1
    # B indices: rows br..br+1, cols bc..bc+1
    # C indices: rows cr..cr+1, cols cc..cc+1
    def add_strassen(ar, ac, br, bc, cr, cc):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[ar*m+ac][col] = 1; U[(ar+1)*m+(ac+1)][col] = 1
        V[br*p+bc][col] = 1; V[(br+1)*p+(bc+1)][col] = 1
        W[cc*n+cr][col] = 1; W[(cc+1)*n+(cr+1)][col] = 1
        col += 1
        # M2 = (A21+A22)B11
        U[(ar+1)*m+ac][col] = 1; U[(ar+1)*m+(ac+1)][col] = 1
        V[br*p+bc][col] = 1
        W[cc*n+(cr+1)][col] = 1; W[(cc+1)*n+(cr+1)][col] = -1
        col += 1
        # M3 = A11(B12-B22)
        U[ar*m+ac][col] = 1
        V[br*p+(bc+1)][col] = 1; V[(br+1)*p+(bc+1)][col] = -1
        W[(cc+1)*n+cr][col] = 1; W[(cc+1)*n+(cr+1)][col] = 1
        col += 1
        # M4 = A22(B21-B11)
        U[(ar+1)*m+(ac+1)][col] = 1
        V[(br+1)*p+bc][col] = 1; V[br*p+bc][col] = -1
        W[cc*n+cr][col] = -1; W[cc*n+(cr+1)][col] = 1
        col += 1
        # M5 = (A11+A12)B22
        U[ar*m+ac][col] = 1; U[ar*m+(ac+1)][col] = 1
        V[(br+1)*p+(bc+1)][col] = 1
        W[cc*n+cr][col] = 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[(ar+1)*m+ac][col] = 1; U[ar*m+ac][col] = -1
        V[br*p+bc][col] = 1; V[br*p+(bc+1)][col] = 1
        W[(cc+1)*n+(cr+1)][col] = 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[ar*m+(ac+1)][col] = 1; U[(ar+1)*m+(ac+1)][col] = -1
        V[(br+1)*p+bc][col] = 1; V[(br+1)*p+(bc+1)][col] = 1
        W[cc*n+cr][col] = -1; W[(cc+1)*n+cr][col] = 1
        col += 1

    # Helper to add schoolbook terms for small blocks
    # ar, ac: A top-left, size ar_size x ac_size
    # br, bc: B top-left, size br_size x bc_size
    # cr, cc: C top-left, size cr_size x cc_size
    def add_schoolbook(ar, ac, br, bc, cr, cc, ar_size, ac_size, br_size, bc_size):
        nonlocal col
        for i in range(ar_size):
            for j in range(ac_size):
                for k in range(bc_size):
                    add_term(ar*m+ac+i+j, br*p+bc+j+k, cc*n+cr+k+i)

    # --- Part 1: Strassen 2x2x2 terms ---
    # Indices:
    # A11: ar=0, ac=0, size 2x2
    # A12: ar=0, ac=2, size 2x2
    # A21: ar=2, ac=0, size 2x2
    # A22: ar=2, ac=2, size 2x2
    # B11: br=0, bc=0, size 2x2
    # B12: br=0, bc=2, size 2x2
    # B21: br=2, bc=0, size 2x2
    # B22: br=2, bc=2, size 2x2
    # C11: cr=0, cc=0, size 2x2
    # C12: cr=0, cc=2, size 2x2
    # C21: cr=2, cc=0, size 2x2
    # C22: cr=2, cc=2, size 2x2

    # Term 1: A11 * B11 -> C11
    add_strassen(0, 0, 0, 0, 0, 0)
    # Term 2: A12 * B21 -> C11
    add_strassen(0, 2, 2, 0, 0, 0)
    # Term 3: A11 * B12 -> C12
    add_strassen(0, 0, 0, 2, 0, 2)
    # Term 4: A12 * B22 -> C12
    add_strassen(0, 2, 2, 2, 0, 2)
    # Term 5: A21 * B11 -> C21
    add_strassen(2, 0, 0, 0, 2, 0)
    # Term 6: A22 * B21 -> C21
    add_strassen(2, 2, 2, 0, 2, 0)
    # Term 7: A21 * B12 -> C22
    add_strassen(2, 0, 0, 2, 2, 2)
    # Term 8: A22 * B22 -> C22
    add_strassen(2, 2, 2, 2, 2, 2)
    
    # --- Part 2: Schoolbook 2x2 * 2x1 -> 2x1 ---
    # B13: br=0, bc=4, size 2x1
    # B23: br=2, bc=4, size 2x1
    # C13: cr=0, cc=4, size 2x1
    # C23: cr=2, cc=4, size 2x1

    # A11 * B13 -> C13
    add_schoolbook(0, 0, 0, 4, 0, 4, 2, 2, 2, 1)
    # A12 * B23 -> C13
    add_schoolbook(0, 2, 2, 4, 0, 4, 2, 2, 2, 1)
    # A21 * B13 -> C23
    add_schoolbook(2, 0, 0, 4, 2, 4, 2, 2, 2, 1)
    # A22 * B23 -> C23
    add_schoolbook(2, 2, 2, 4, 2, 4, 2, 2, 2, 1)

    # --- Part 3: Schoolbook 2x1 * 1x2 -> 2x2 ---
    # A13: ar=0, ac=4, size 2x1
    # A23: ar=2, ac=4, size 2x1
    # B31: br=4, bc=0, size 1x2
    # B32: br=4, bc=2, size 1x2
    # C11: cr=0, cc=0, size 2x2
    # C12: cr=0, cc=2, size 2x2
    # C21: cr=2, cc=0, size 2x2
    # C22: cr=2, cc=2, size 2x2

    # A13 * B31 -> C11
    add_schoolbook(0, 4, 4, 0, 0, 0, 2, 1, 1, 2)
    # A13 * B32 -> C12
    add_schoolbook(0, 4, 4, 2, 0, 2, 2, 1, 1, 2)
    # A23 * B31 -> C21
    add_schoolbook(2, 4, 4, 0, 2, 0, 2, 1, 1, 2)
    # A23 * B32 -> C22
    add_schoolbook(2, 4, 4, 2, 2, 2, 2, 1, 1, 2)

    # --- Part 4: Schoolbook 2x1 * 1x1 -> 2x1 ---
    # B33: br=4, bc=4, size 1x1
    # C13: cr=0, cc=4, size 2x1
    # C23: cr=2, cc=4, size 2x1

    # A13 * B33 -> C13
    add_schoolbook(0, 4, 4, 4, 0, 4, 2, 1, 1, 1)
    # A23 * B33 -> C23
    add_schoolbook(2, 4, 4, 4, 2, 4, 2, 1, 1, 1)

    # Verify rank
    if col != rank:
        # If something went wrong, fall back to schoolbook to ensure validity
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
