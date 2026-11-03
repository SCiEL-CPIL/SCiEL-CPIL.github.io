"""Rewritten program using a different construction approach."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2x2 Strassen block
    def add_strassen(a_i, a_js, b_js, b_ks, c_i, c_ks, start_col):
        # a_i: row index in A (0 or 1)
        # a_js: list of 2 col indices in A
        # b_js: list of 2 row indices in B
        # b_ks: list of 2 col indices in B
        # c_i: row index in C (0 or 1)
        # c_ks: list of 2 col indices in C
        
        # M1 = (A[a_i, a_js[0]] + A[a_i+1, a_js[1]]) * (B[b_js[0], b_ks[0]] + B[b_js[1], b_ks[1]])
        # Wait, Strassen uses A00, A11 etc.
        # A00 is A[0,0], A11 is A[1,1].
        # So M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        
        # Term 1
        U[a_i * m + a_js[0]][start_col] += 1
        U[(a_i + 1) * m + a_js[1]][start_col] += 1
        V[b_js[0] * p + b_ks[0]][start_col] += 1
        V[b_js[1] * p + b_ks[1]][start_col] += 1
        
        # W for M1:
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        W[c_ks[0] * n + c_i][start_col] += 1
        W[c_ks[1] * n + c_i][start_col] += 1
        W[c_ks[0] * n + c_i + 1][start_col] += 1
        W[c_ks[1] * n + c_i + 1][start_col] += 1
        
        # Term 2: M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[(a_i + 1) * m + a_js[0]][start_col + 1] += 1
        U[(a_i + 1) * m + a_js[1]][start_col + 1] += 1
        V[b_js[0] * p + b_ks[0]][start_col + 1] += 1
        
        # W for M2:
        # C10 += M2, C11 -= M2
        W[c_ks[0] * n + c_i + 1][start_col + 1] += 1
        W[c_ks[1] * n + c_i + 1][start_col + 1] -= 1
        
        # Term 3: M3 = A[0,0] * B[1,0]
        U[a_i * m + a_js[0]][start_col + 2] += 1
        V[b_js[1] * p + b_ks[0]][start_col + 2] += 1
        
        # W for M3:
        # C01 += M3, C11 += M3
        W[c_ks[1] * n + c_i][start_col + 2] += 1
        W[c_ks[1] * n + c_i + 1][start_col + 2] += 1
        
        # Term 4: M4 = A[1,1] * B[1,1]
        U[(a_i + 1) * m + a_js[1]][start_col + 3] += 1
        V[b_js[1] * p + b_ks[1]][start_col + 3] += 1
        
        # W for M4:
        # C00 += M4, C10 += M4
        W[c_ks[0] * n + c_i][start_col + 3] += 1
        W[c_ks[0] * n + c_i + 1][start_col + 3] += 1
        
        # Term 5: M5 = (A[0,0] + A[0,1]) * (B[1,1] - B[1,0])
        U[a_i * m + a_js[0]][start_col + 4] += 1
        U[a_i * m + a_js[1]][start_col + 4] += 1
        V[b_js[1] * p + b_ks[1]][start_col + 4] += 1
        V[b_js[1] * p + b_ks[0]][start_col + 4] -= 1
        
        # W for M5:
        # C00 -= M5, C01 += M5
        W[c_ks[0] * n + c_i][start_col + 4] -= 1
        W[c_ks[1] * n + c_i][start_col + 4] += 1
        
        # Term 6: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[(a_i + 1) * m + a_js[0]][start_col + 5] += 1
        U[a_i * m + a_js[0]][start_col + 5] -= 1
        V[b_js[0] * p + b_ks[0]][start_col + 5] += 1
        V[b_js[0] * p + b_ks[1]][start_col + 5] += 1
        
        # W for M6:
        # C11 += M6
        W[c_ks[1] * n + c_i + 1][start_col + 5] += 1
        
        # Term 7: M7 = (A[0,1] - A[1,1]) * (B[0,1] + B[1,1])
        U[a_i * m + a_js[1]][start_col + 6] += 1
        U[(a_i + 1) * m + a_js[1]][start_col + 6] -= 1
        V[b_js[0] * p + b_ks[1]][start_col + 6] += 1
        V[b_js[1] * p + b_ks[1]][start_col + 6] += 1
        
        # W for M7:
        # C00 += M7
        W[c_ks[0] * n + c_i][start_col + 6] += 1
        
        return start_col + 7

    # Function to add a 2x2x1 Linear block
    def add_linear_2x2x1(a_i, a_js, b_js, b_k, c_i, c_k, start_col):
        # C[c_i, c_k] = sum_j A[a_i, a_js[j]] * B[b_js[j], b_k]
        # This is 2 terms.
        # Term 1: A[a_i, a_js[0]] * B[b_js[0], b_k]
        U[a_i * m + a_js[0]][start_col] += 1
        V[b_js[0] * p + b_k][start_col] += 1
        W[c_k * n + c_i][start_col] += 1
        
        # Term 2: A[a_i+1, a_js[1]] * B[b_js[1], b_k]
        # Wait, C has 2 rows.
        # C[0, k] = A[0, 0]B[0, k] + A[0, 1]B[1, k]
        # C[1, k] = A[1, 0]B[0, k] + A[1, 1]B[1, k]
        # So for each i in {0,1}, we need 2 terms.
        # Total 4 terms.
        
        # I already did i=0, j=0.
        # Let's do i=0, j=1
        U[a_i * m + a_js[1]][start_col + 1] += 1
        V[b_js[1] * p + b_k][start_col + 1] += 1
        W[c_k * n + c_i][start_col + 1] += 1
        
        # i=1, j=0
        U[(a_i + 1) * m + a_js[0]][start_col + 2] += 1
        V[b_js[0] * p + b_k][start_col + 2] += 1
        W[c_k * n + c_i + 1][start_col + 2] += 1
        
        # i=1, j=1
        U[(a_i + 1) * m + a_js[1]][start_col + 3] += 1
        V[b_js[1] * p + b_k][start_col + 3] += 1
        W[c_k * n + c_i + 1][start_col + 3] += 1
        
        return start_col + 4

    # A_L: i=0,1. j=0,1.
    # B_T: j=0,1. k=0..6.
    # C_L: i=0,1. k=0..6.
    
    # A_R: i=0,1. j=2,3.
    # B_B: j=2,3. k=0..6.
    # C_R: i=0,1. k=0..6.
    
    # Process A_L * B_T
    # k=0,1: Strassen
    col = add_strassen(0, [0, 1], [0, 1], [0, 1], 0, [0, 1], col)
    # k=2,3: Strassen
    col = add_strassen(0, [0, 1], [0, 1], [2, 3], 0, [2, 3], col)
    # k=4,5: Strassen
    col = add_strassen(0, [0, 1], [0, 1], [4, 5], 0, [4, 5], col)
    # k=6: Linear
    col = add_linear_2x2x1(0, [0, 1], [0, 1], 6, 0, 6, col)
    
    # Process A_R * B_B
    # k=0,1: Strassen
    col = add_strassen(0, [2, 3], [2, 3], [0, 1], 0, [0, 1], col)
    # k=2,3: Strassen
    col = add_strassen(0, [2, 3], [2, 3], [2, 3], 0, [2, 3], col)
    # k=4,5: Strassen
    col = add_strassen(0, [2, 3], [2, 3], [4, 5], 0, [4, 5], col)
    # k=6: Linear
    col = add_linear_2x2x1(0, [2, 3], [2, 3], 6, 0, 6, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
