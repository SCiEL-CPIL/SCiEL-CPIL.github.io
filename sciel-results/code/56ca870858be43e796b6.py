"""Strassen-based decomposition for 2x2 * 2x15 matrix multiplication."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Use Strassen's algorithm on pairs of columns
    # 7 pairs of columns (14 columns) use 7 multiplications each
    # 1 remaining column uses 4 schoolbook multiplications
    # Total rank = 7*7 + 4 = 53
    
    num_pairs = p // 2  # 7 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4  # 49 + 4 = 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    
    # Strassen's 7 products for a 2x2 block
    for pair_idx in range(num_pairs):
        k_a = pair_idx * 2
        k_b = pair_idx * 2 + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        r = col
        U[0][r] = 1
        U[3][r] = 1
        V[k_a][r] = 1
        V[15 + k_b][r] = 1
        W[k_a * 2][r] = 1
        W[k_b * 2 + 1][r] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        r = col
        U[2][r] = 1
        U[3][r] = 1
        V[k_a][r] = 1
        W[k_a * 2 + 1][r] = 1
        W[k_b * 2 + 1][r] = -1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        r = col
        U[0][r] = 1
        V[k_b][r] = 1
        V[15 + k_b][r] = -1
        W[k_b * 2][r] = 1
        W[k_b * 2 + 1][r] = 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        r = col
        U[3][r] = 1
        V[15 + k_a][r] = 1
        V[k_a][r] = -1
        W[k_a * 2][r] = 1
        W[k_a * 2 + 1][r] = 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        r = col
        U[0][r] = 1
        U[1][r] = 1
        V[15 + k_b][r] = 1
        W[k_a * 2][r] = -1
        W[k_b * 2][r] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        r = col
        U[2][r] = 1
        U[0][r] = -1
        V[k_a][r] = 1
        V[k_b][r] = 1
        W[k_b * 2 + 1][r] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        r = col
        U[1][r] = 1
        U[3][r] = -1
        V[15 + k_a][r] = 1
        V[15 + k_b][r] = 1
        W[k_a * 2][r] = 1
        col += 1
    
    # Handle the remaining single column (k = 14)
    if remainder > 0:
        k = 2 * num_pairs  # k = 14
        
        # Product 1: a00 * B[0, k]
        r = col
        U[0][r] = 1
        V[k][r] = 1
        W[k * 2][r] = 1
        col += 1
        
        # Product 2: a01 * B[1, k]
        r = col
        U[1][r] = 1
        V[15 + k][r] = 1
        W[k * 2][r] = 1
        col += 1
        
        # Product 3: a10 * B[0, k]
        r = col
        U[2][r] = 1
        V[k][r] = 1
        W[k * 2 + 1][r] = 1
        col += 1
        
        # Product 4: a11 * B[1, k]
        r = col
        U[3][r] = 1
        V[15 + k][r] = 1
        W[k * 2 + 1][r] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
