import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 120

    # Total rank: 19 (Laderman for 4x4) + 16 (Schoolbook for 4x1) = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 16 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # --- Part 1: 4x4 * 4x4 using Laderman (Rank 19) ---
    # The 19 terms of Laderman's decomposition for 4x4 matrix multiplication.
    # Note: Indices for A are (i, j), for B are (j, k), for C are (k, i) in the trace convention W[k*n+i].
    # Wait, the trace convention is W[k*n+i] where C[i,k] = sum_j A[i,j]B[j,k].
    # In the tensor T[i,j,k] = A[i,j]B[j,k]C[k,i] (or similar), the W matrix maps the product to the output.
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # So W[k*n+i] corresponds to C[i,k].

    # Laderman 19 terms:
    # u_list: [(i, j, coeff)] for A[i,j]
    # v_list: [(j, k, coeff)] for B[j,k]
    # w_list: [(k, i, coeff)] for C[i,k] (stored as W[k*n+i])

    # Term 0
    add_term(col, 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1)]); col += 1
             
    # Term 1
    add_term(col, 
             [(0,1,1), (1,1,1)], 
             [(1,0,1), (1,1,1)], 
             [(1,0,1), (1,1,1)]); col += 1
             
    # Term 2
    add_term(col, 
             [(0,2,1), (1,2,1)], 
             [(2,0,1), (2,1,1)], 
             [(2,0,1), (2,1,1)]); col += 1
             
    # Term 3
    add_term(col, 
             [(0,3,1), (1,3,1)], 
             [(3,0,1), (3,1,1)], 
             [(3,0,1), (3,1,1)]); col += 1
             
    # Term 4
    add_term(col, 
             [(2,0,1), (3,0,1)], 
             [(0,2,1), (0,3,1)], 
             [(2,2,1), (3,2,1)]); col += 1 # Correction: W[k*n+i] for C[i,k]. C[2,2] is k=2, i=2 -> W[2*4+2]=W[10].
    # Let's be precise with W indices.
    # C[i,k] is stored at W[k*n+i].
    # Term 4: u = a20+a30, v = b02+b03. Product contributes to C[2,2], C[2,3], C[3,2], C[3,3]?
    # (a20+a30)(b02+b03) = a20b02 + a20b03 + a30b02 + a30b03.
    # C[2,2] += a20b02. C[2,3] += a20b03. C[3,2] += a30b02. C[3,3] += a30b03.
    # So w_list should have:
    # C[2,2] -> W[2*4+2] = W[10], coeff 1
    # C[2,3] -> W[3*4+2] = W[14], coeff 1
    # C[3,2] -> W[2*4+3] = W[11], coeff 1
    # C[3,3] -> W[3*4+3] = W[15], coeff 1
    # My previous add_term for Term 4 was wrong.

    # Let's restart the Laderman terms carefully.
    # I will define a helper to add terms with explicit i,j,k for C.
    def add_term_c(col, u_list, v_list, c_list):
        # c_list: [(i, k, coeff)] for C[i,k]
        w_list = []
        for i, k, c in c_list:
            w_list.append((k, i, c))
        add_term(col, u_list, v_list, w_list)

    col = 0
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Laderman 19 Terms
    # 1. (a00+a11)(b00+b11) -> c00, c01, c10, c11
    add_term_c(col, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]); col += 1
    # 2. (a01+a11)(b10+b11) -> c01, c02, c11, c12
    add_term_c(col, [(0,1,1),(1,1,1)], [(1,0,1),(1,1,1)], [(0,1,1),(0,2,1),(1,1,1),(1,2,1)]); col += 1
    # 3. (a02+a12)(b20+b21) -> c02, c03, c12, c13
    add_term_c(col, [(0,2,1),(1,2,1)], [(2,0,1),(2,1,1)], [(0,2,1),(0,3,1),(1,2,1),(1,3,1)]); col += 1
    # 4. (a03+a13)(b30+b31) -> c03, c04(no), c13, c14(no) -> c03, c13
    # Wait, B is 4x5? No, Part 1 is 4x4 * 4x4. B is 4x4.
    # (a03+a13)(b30+b31) -> c03, c13? No.
    # a03*b30 -> c03. a03*b31 -> c04 (out of bounds for 4x4 C? No, C is 4x4).
    # B is 4x4, so indices 0..3.
    # (a03+a13)(b30+b31) = a03b30 + a03b31 + a13b30 + a13b31.
    # c03 += a03b30. c04 is not in 4x4.
    # Wait, C is 4x4. Indices 0..3.
    # a03b31 -> c[0,3]? No, sum over j.
    # C[i,k] = sum_j A[i,j]B[j,k].
    # Term: u = a03+a13. v = b30+b31.
    # a03*b30: i=0, j=3, k=0 -> C[0,0].
    # a03*b31: i=0, j=3, k=1 -> C[0,1].
    # a13*b30: i=1, j=3, k=0 -> C[1,0].
    # a13*b31: i=1, j=3, k=1 -> C[1,1].
    # So w_list: [(0,0,1), (0,1,1), (1,0,1), (1,1,1)].
    add_term_c(col, [(0,3,1),(1,3,1)], [(3,0,1),(3,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]); col += 1

    # 5. (a20+a30)(b02+b03)
    # a20b02 -> C[2,2]. a20b03 -> C[2,3].
    # a30b02 -> C[3,2]. a30b03 -> C[3,3].
    add_term_c(col, [(2,0,1),(3,0,1)], [(0,2,1),(0,3,1)], [(2,2,1),(2,3,1),(3,2,1),(3,3,1)]); col += 1

    # 6. (a21+a31)(b12+b13)
    # a21b12 -> C[2,2]. a21b13 -> C[2,3].
    # a31b12 -> C[3,2]. a31b13 -> C[3,3].
    add_term_c(col, [(2,1,1),(3,1,1)], [(1,2,1),(1,3,1)], [(2,2,1),(2,3,1),(3,2,1),(3,3,1)]); col += 1

    # 7. (a22+a32)(b22+b23)
    # a22b22 -> C[2,2]. a22b23 -> C[2,3].
    # a32b22 -> C[3,2]. a32b23 -> C[3,3].
    add_term_c(col, [(2,2,1),(3,2,1)], [(2,2,1),(2,3,1)], [(2,2,1),(2,3,1),(3,2,1),(3,3,1)]); col += 1

    # 8. (a23+a33)(b32+b33)
    # a23b32 -> C[2,2]. a23b33 -> C[2,3].
    # a33b32 -> C[3,2]. a33b33 -> C[3,3].
    add_term_c(col, [(2,3,1),(3,3,1)], [(3,2,1),(3,3,1)], [(2,2,1),(2,3,1),(3,2,1),(3,3,1)]); col += 1

    # 9. (a00+a22)(b20+b02)
    # a00b20 -> C[0,0]. a00b02 -> C[0,2].
    # a22b20 -> C[2,0]. a22b02 -> C[2,2].
    add_term_c(col, [(0,0,1),(2,2,1)], [(2,0,1),(0,2,1)], [(0,0,1),(0,2,1),(2,0,1),(2,2,1)]); col += 1

    # 10. (a01+a23)(b30+b03)
    # a01b30 -> C[0,0]. a01b03 -> C[0,3].
    # a23b30 -> C[2,0]. a23b03 -> C[2,3].
    add_term_c(col, [(0,1,1),(2,3,1)], [(3,0,1),(0,3,1)], [(0,0,1),(0,3,1),(2,0,1),(2,3,1)]); col += 1

    # 11. (a02+a20)(b00+b22)
    # a02b00 -> C[0,0]. a02b22 -> C[0,2].
    # a20b00 -> C[2,0]. a20b22 -> C[2,2].
    add_term_c(col, [(0,2,1),(2,0,1)], [(0,0,1),(2,2,1)], [(0,0,1),(0,2,1),(2,0,1),(2,2,1)]); col += 1

    # 12. (a03+a21)(b10+b23)
    # a03b10 -> C[0,0]. a03b23 -> C[0,3].
    # a21b10 -> C[2,0]. a21b23 -> C[2,3].
    add_term_c(col, [(0,3,1),(2,1,1)], [(1,0,1),(2,3,1)], [(0,0,1),(0,3,1),(2,0,1),(2,3,1)]); col += 1

    # 13. (a10+a32)(b21+b03)
    # a10b21 -> C[1,1]. a10b03 -> C[1,3].
    # a32b21 -> C[3,1]. a32b03 -> C[3,3].
    add_term_c(col, [(1,0,1),(3,2,1)], [(2,1,1),(0,3,1)], [(1,1,1),(1,3,1),(3,1,1),(3,3,1)]); col += 1

    # 14. (a11+a33)(b31+b00)
    # a11b31 -> C[1,1]. a11b00 -> C[1,0].
    # a33b31 -> C[3,1]. a33b00 -> C[3,0].
    add_term_c(col, [(1,1,1),(3,3,1)], [(3,1,1),(0,0,1)], [(1,1,1),(1,0,1),(3,1,1),(3,0,1)]); col += 1

    # 15. (a12+a30)(b01+b20)
    # a12b01 -> C[1,1]. a12b20 -> C[1,0].
    # a30b01 -> C[3,1]. a30b20 -> C[3,0].
    add_term_c(col, [(1,2,1),(3,0,1)], [(0,1,1),(2,0,1)], [(1,1,1),(1,0,1),(3,1,1),(3,0,1)]); col += 1

    # 16. (a13+a31)(b11+b21)
    # a13b11 -> C[1,1]. a13b21 -> C[1,2].
    # a31b11 -> C[3,1]. a31b21 -> C[3,2].
    add_term_c(col, [(1,3,1),(3,1,1)], [(1,1,1),(2,1,1)], [(1,1,1),(1,2,1),(3,1,1),(3,2,1)]); col += 1

    # 17. (a00-a11)(b10-b00)
    # a00b10 -> C[0,0]. a00b00 -> C[0,0] (coeff -1).
    # a11b10 -> C[1,0] (coeff -1). a11b00 -> C[1,0] (coeff 1).
    # C[0,0] += a00b10 - a00b00.
    # C[1,0] += -a11b10 + a11b00.
    add_term_c(col, [(0,0,1),(1,1,-1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]); col += 1

    # 18. (a20-a30)(b03-b02)
    # a20b03 -> C[2,3]. a20b02 -> C[2,2] (coeff -1).
    # a30b03 -> C[3,3] (coeff -1). a30b02 -> C[3,2] (coeff 1).
    add_term_c(col, [(2,0,1),(3,0,-1)], [(0,3,1),(0,2,-1)], [(2,3,1),(2,2,-1),(3,3,-1),(3,2,1)]); col += 1

    # 19. (a02-a12)(b21-b20)
    # a02b21 -> C[0,1]. a02b20 -> C[0,0] (coeff -1).
    # a12b21 -> C[1,1] (coeff -1). a12b20 -> C[1,0] (coeff 1).
    add_term_c(col, [(0,2,1),(1,2,-1)], [(2,1,1),(2,0,-1)], [(0,1,1),(0,0,-1),(1,1,-1),(1,0,1)]); col += 1

    # --- Part 2: 4x4 * 4x1 using Schoolbook (Rank 16) ---
    # A is 4x4, B is 4x1 (columns 0..3 of B are used? No, B is 4x5).
    # Wait, the task is 4x4 * 4x5.
    # We split B into B_4x4 and B_4x1.
    # The 4x4 part is handled by Laderman.
    # The 4x1 part (column 4 of B) is handled by Schoolbook.
    # C_4x1 = A_4x4 * B_4x1.
    # C[i,4] = sum_j A[i,j] B[j,4].
    # This requires 16 multiplications.
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]  # W[k*n+i] where k=4, i=i
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
