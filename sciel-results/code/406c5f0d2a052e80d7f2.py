"""Program for matrix multiplication 2x4x6 with a rank-reduced exact tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.monotonic() + 120  # safety margin before 150s kill
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    # We will use rank 42 decomposition (3 Strassen 2x4x2 blocks)
    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    # Strassen decomposition for 2x2x2 matrix multiplication
    # A (2x2) * B (2x2) = C (2x2)
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = a11 * (b21 - b22)
    # M2 = (a11 + a12) * b22
    # M3 = (a21 + a22) * b11
    # M4 = a22 * (b12 - b11)
    # M5 = (a11 + a22) * (b11 + b22)
    # M6 = (a12 - a22) * (b21 + b22)
    # M7 = (a21 - a11) * (b11 + b12)
    #
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M7
    # C22 = M5 - M1 + M3 - M4

    # For our 2x4x6 problem:
    # A is 2x4, B is 4x6, C is 2x6
    # We split B into 3 blocks of 4x2: B = [B1, B2, B3] where each Bj is 4x2
    # C = A * B = A * B1 + A * B2 + A * B3
    # Each A*Bj is 2x2 multiplication

    # For a 2x4 * 4x2 multiplication:
    # A is 2x4: [[a11, a12, a13, a14], [a21, a22, a23, a24]]
    # B is 4x2: [[b11, b12], [b21, b22], [b31, b32], [b41, b42]]
    # C = A*B is 2x2:
    # C11 = a11*b11 + a12*b21 + a13*b31 + a14*b41
    # C12 = a11*b12 + a12*b22 + a13*b32 + a14*b42
    # C21 = a21*b11 + a22*b21 + a23*b31 + a24*b41
    # C22 = a21*b12 + a22*b22 + a23*b32 + a24*b42

    # We can group the 4x4 part using Strassen by treating:
    # A' = [[a11, a12], [a21, a22]] (left 2x2 of A)
    # B' = [[b11, b12], [b21, b22]] (top 2x2 of B)
    # A'' = [[a13, a14], [a23, a24]] (right 2x2 of A)
    # B'' = [[b31, b32], [b41, b42]] (bottom 2x2 of B)
    # C = A'*B' + A''*B''
    # This requires 2 Strassen decompositions = 14 rank terms

    # Strassen coefficients for 2x2x2:
    # For each Strassen block, we have 7 products
    # The U, V, W vectors for each product tell us how A, B, C are involved

    # Strassen 2x2x2 factorization:
    # Product 1: M1 = a11 * (b21 - b22)
    #   U: a11 -> row 0, col 0
    #   V: b21 - b22 -> row 1, col 0 = 1, row 1, col 1 = -1 (but V is for B which is 2x2)
    #   W: M1 contributes to C12 (row 0, col 1) with +1
    #
    # Let me be more careful. For 2x2x2:
    # A indices: (i,j) where i=0,1 is row of A, j=0,1 is col of A
    # B indices: (j,k) where j=0,1 is row of B, k=0,1 is col of B
    # C indices: (i,k) where i=0,1 is row of C, k=0,1 is col of C
    #
    # In our flattened format:
    # U is (n*m, rank) = (4, rank) for the 2x4 part... wait
    #
    # Let me re-examine. For the full 2x4x6 problem:
    # U is (n*m, rank) = (8, rank)
    # V is (m*p, rank) = (24, rank)
    # W is (p*n, rank) = (12, rank)
    #
    # U[i*m+j, c] where i in 0..1, j in 0..3
    # V[j*p+k, c] where j in 0..3, k in 0..5
    # W[k*n+i, c] where k in 0..5, i in 0..1

    # For the 2x4x2 block (A is 2x4, B_block is 4x2, C_block is 2x2):
    # We split into two 2x2x2 Strassen blocks:
    # Block 1: A_left (2x2) * B_top (2x2) -> contributes to C
    # Block 2: A_right (2x2) * B_bottom (2x2) -> contributes to C
    #
    # For Block 1:
    # A_left[i, j] for i=0,1; j=0,1 -> U[i*4 + j, c]
    # B_top[j, k] for j=0,1; k=0,1 -> V[j*6 + k, c]
    # C[i, k] for i=0,1; k=0,1 -> W[k*2 + i, c]
    #
    # For Block 2:
    # A_right[i, j] for i=0,1; j=2,3 -> U[i*4 + j, c]
    # B_bottom[j, k] for j=2,3; k=0,1 -> V[j*6 + k, c]
    # C[i, k] for i=0,1; k=0,1 -> W[k*2 + i, c]

    # Strassen 2x2x2 factors (7 products):
    # Let's define them as (U_coef, V_coef, W_coef) for each of the 7 products
    # For a 2x2x2 block with A indices (i,j), B indices (j,k), C indices (i,k):
    
    # Product 1: M1 = a00 * (b10 - b11)
    # U: a00 = U[0*2+0] = U[0]
    # V: b10 - b11 = V[1*2+0] - V[1*2+1] = V[2] - V[3]
    # W: M1 contributes to C[0,1] = W[1*2+0] = W[2] with +1
    
    # Product 2: M2 = (a00 + a01) * b11
    # U: a00 + a01 = U[0] + U[1]
    # V: b11 = V[3]
    # W: M2 contributes to C[0,0] = W[0] with -1
    
    # Product 3: M3 = (a10 + a11) * b00
    # U: a10 + a11 = U[2] + U[3]
    # V: b00 = V[0]
    # W: M3 contributes to C[1,0] = W[1] with +1
    
    # Product 4: M4 = a11 * (b01 - b00)
    # U: a11 = U[3]
    # V: b01 - b00 = V[1] - V[0]
    # W: M4 contributes to C[0,0] = W[0] with +1 and C[1,1] = W[3] with -1
    
    # Product 5: M5 = (a00 + a11) * (b00 + b11)
    # U: a00 + a11 = U[0] + U[3]
    # V: b00 + b11 = V[0] + V[3]
    # W: M5 contributes to C[0,0] = W[0] with +1, C[0,1] = W[2] with +1, C[1,0] = W[1] with +1, C[1,1] = W[3] with +1
    
    # Product 6: M6 = (a01 - a11) * (b10 + b11)
    # U: a01 - a11 = U[1] - U[3]
    # V: b10 + b11 = V[2] + V[3]
    # W: M6 contributes to C[0,0] = W[0] with +1
    
    # Product 7: M7 = (a10 - a00) * (b00 + b01)
    # U: a10 - a00 = U[2] - U[0]
    # V: b00 + b01 = V[0] + V[1]
    # W: M7 contributes to C[1,0] = W[1] with +1

    # Now let's define the Strassen factors for a generic 2x2x2 block.
    # For each factor, we specify:
    # - U_indices: list of (local_u_index, coefficient)
    # - V_indices: list of (local_v_index, coefficient)
    # - W_indices: list of (local_w_index, coefficient)
    # where local indices are for the 2x2 sub-block

    strassen_factors = [
        # Factor 1: M1 = a00 * (b10 - b11)
        {
            'U': [(0, 1)],
            'V': [(2, 1), (3, -1)],
            'W': [(2, 1)]  # C[0,1]
        },
        # Factor 2: M2 = (a00 + a01) * b11
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1)]  # C[0,0]
        },
        # Factor 3: M3 = (a10 + a11) * b00
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(1, 1)]  # C[1,0]
        },
        # Factor 4: M4 = a11 * (b01 - b00)
        {
            'U': [(3, 1)],
            'V': [(1, 1), (0, -1)],
            'W': [(0, 1), (3, -1)]  # C[0,0] +1, C[1,1] -1
        },
        # Factor 5: M5 = (a00 + a11) * (b00 + b11)
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (2, 1), (1, 1), (3, 1)]  # C[0,0]+1, C[0,1]+1, C[1,0]+1, C[1,1]+1
        },
        # Factor 6: M6 = (a01 - a11) * (b10 + b11)
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]  # C[0,0]
        },
        # Factor 7: M7 = (a10 - a00) * (b00 + b01)
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(1, 1)]  # C[1,0]
        },
    ]

    # For the full 2x4x6 problem, we have 3 blocks of 2x4x2.
    # Block b (b=0,1,2) corresponds to columns 2*b and 2*b+1 of B, and C.
    # Within each block, we split into 2 Strassen 2x2x2 blocks:
    #   Sub-block 0: A_left (cols 0,1) * B_top (rows 0,1)
    #   Sub-block 1: A_right (cols 2,3) * B_bottom (rows 2,3)

    col = 0
    for b in range(3):  # 3 blocks of 2x4x2
        for sub in range(2):  # 2 Strassen 2x2x2 sub-blocks per 2x4x2 block
            for factor in strassen_factors:
                # Determine global indices
                # A_local[i, j_local] where j_local in 0,1
                # If sub=0: A_left, so j_global = j_local
                # If sub=1: A_right, so j_global = 2 + j_local
                j_offset = 2 * sub
                # B_local[j_local, k_local] where j_local in 0,1, k_local in 0,1
                # If sub=0: B_top, so j_global = j_local, k_global = 2*b + k_local
                # If sub=1: B_bottom, so j_global = 2 + j_local, k_global = 2*b + k_local
                # C_local[i, k_local] where i in 0,1, k_local in 0,1
                # C global: k_global = 2*b + k_local, i global = i
                # W index: k_global * n + i = (2*b + k_local) * 2 + i

                for (u_local, u_coef) in factor['U']:
                    i, j_local = u_local // 2, u_local % 2
                    u_global = i * m + (j_local + j_offset)
                    U[u_global][col] += u_coef

                for (v_local, v_coef) in factor['V']:
                    j_local, k_local = v_local // 2, v_local % 2
                    v_global = (j_local + j_offset) * p + (2 * b + k_local)
                    V[v_global][col] += v_coef

                for (w_local, w_coef) in factor['W']:
                    i, k_local = w_local // 2, w_local % 2
                    w_global = (2 * b + k_local) * n + i
                    W[w_global][col] += w_coef

                col += 1
                if time.monotonic() > deadline - 10:
                    break
            if time.monotonic() > deadline - 10:
                break
        if time.monotonic() > deadline - 10:
            break

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
