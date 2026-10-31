"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-56 decomposition for 4x4x4 matrix multiplication.
    
    Uses 8 independent rank-7 decompositions for 2x2 block products.
    A 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices.
    """
    n, m, p = 4, 4, 4
    rank = 56  # 8 block-products * 7 rank each
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 16 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # 2x2 rank-7 decomposition coefficients
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # t1 = a*e, t2 = b*h, t3 = (a+b)*g, t4 = (c+d)*e, t5 = (c-d)*(g+h), t6 = (a+c)*f, t7 = (b+d)*h
    # C00 = t1 + t3, C01 = t2 + t6, C10 = t4 + t7, C11 = t5 + t6 + t7
    
    # U factors for 2x2: rows are [a,b,c,d], columns are t1..t7
    # V factors for 2x2: rows are [e,f,g,h], columns are t1..t7
    # W factors for 2x2: rows are [C00,C01,C10,C11], columns are t1..t7
    
    U2 = [
        [1, 0, 1, 0, 1, 1, 0],  # a
        [0, 1, 1, 0, 0, 0, 1],  # b
        [0, 0, 0, 1, 1, 1, 0],  # c
        [0, 0, 0, 1, -1, 0, 1], # d
    ]
    V2 = [
        [1, 0, 0, 1, 0, 0, 0],  # e
        [0, 0, 0, 0, 0, 1, 0],  # f
        [0, 0, 1, 0, 1, 0, 0],  # g
        [0, 1, 0, 0, 1, 0, 1],  # h
    ]
    W2 = [
        [1, 0, 1, 0, 0, 0, 0],  # C00 = t1 + t3
        [0, 1, 0, 0, 0, 1, 0],  # C01 = t2 + t6
        [0, 0, 0, 1, 0, 0, 1],  # C10 = t4 + t7
        [0, 0, 0, 0, 1, 1, 1],  # C11 = t5 + t6 + t7
    ]
    
    # Block structure: 4x4 = 2x2 blocks of 2x2
    # Block (I,J) of A is at rows I*2..I*2+1, cols J*2..J*2+1
    # For output block (I,J): C[I][J] = sum_K A[I][K] * B[K][J]
    # Each term A[I][K]*B[K][J] is a 2x2 product
    
    # 8 block products: (I,K,J) for I in {0,1}, K in {0,1}, J in {0,1}
    # That's 8 products, each rank 7, total rank 56
    
    block_products = []
    for I in range(2):
        for K in range(2):
            for J in range(2):
                block_products.append((I, K, J))
    
    for bp_idx, (I, K, J) in enumerate(block_products):
        base_col = bp_idx * 7
        
        # A block (I,K): rows I*2, I*2+1; cols K*2, K*2+1
        # Within block, element (i,j) is at global row I*2+i, col K*2+j
        # A2 element mapping: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        a_rows = [(I*2 + 0, K*2 + 0), (I*2 + 0, K*2 + 1), (I*2 + 1, K*2 + 0), (I*2 + 1, K*2 + 1)]
        
        # B block (K,J): rows K*2, K*2+1; cols J*2, J*2+1
        b_rows = [(K*2 + 0, J*2 + 0), (K*2 + 0, J*2 + 1), (K*2 + 1, J*2 + 0), (K*2 + 1, J*2 + 1)]
        
        # C block (I,J): rows I*2, I*2+1; cols J*2, J*2+1
        c_rows = [(I*2 + 0, J*2 + 0), (I*2 + 0, J*2 + 1), (I*2 + 1, J*2 + 0), (I*2 + 1, J*2 + 1)]
        
        for t in range(7):
            col = base_col + t
            # U: A elements
            for i in range(4):
                r, c = a_rows[i]
                U[r * m + c][col] = U2[i][t]
            # V: B elements
            for i in range(4):
                r, c = b_rows[i]
                V[r * p + c][col] = V2[i][t]
            # W: C elements
            for i in range(4):
                r, c = c_rows[i]
                W[r * n + c][col] = W2[i][t]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
