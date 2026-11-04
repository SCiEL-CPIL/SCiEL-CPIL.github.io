DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x3x6 multiplication into two 3x3x3 multiplications.
    # Block 1: Columns 0-2 of B
    # Block 2: Columns 3-5 of B
    # Each 3x3x3 multiplication has a schoolbook rank of 9.
    # Total rank = 9 + 9 = 18?
    # Wait. Schoolbook for 3x3x3 is rank 9?
    # C_ik = sum_j A_ij B_jk.
    # There are 9 terms (i,k pairs) summing over j.
    # No, the tensor rank is the number of multiplications.
    # For 3x3x3, there are 9 output entries, each sum of 3 products.
    # Total products = 27.
    # So rank is 27.
    # My previous calculation of 54 was 27 + 27.
    # If I use rank 27 for each block, total is 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in 0..2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                k_global = k + 3
                U[i * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
