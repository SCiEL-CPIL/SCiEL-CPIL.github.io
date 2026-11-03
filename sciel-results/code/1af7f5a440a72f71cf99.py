import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a 2x2 block structure with 3x3 sub-blocks.
    # Strassen for 2x2 blocks: Rank 7
    # For each 3x3 block multiplication, we use a rank-19 decomposition.
    # Total rank = 7 * 19 = 133
    
    rank = 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen combinations for 2x2 blocks
    # A_comb[t] is a list of (row, col, coeff) for A
    # B_comb[t] is a list of (row, col, coeff) for B
    # C_comb[t] is a list of (row, col, coeff) for C
    # Note: C_comb indices are (row, col) of the C block
    
    A_comb_0 = [(0,0,1), (1,1,1)]
    B_comb_0 = [(0,0,1), (1,1,1)]
    C_comb_0 = [(0,0,1), (1,1,1)]
    
    A_comb_1 = [(1,0,1), (1,1,1)]
    B_comb_1 = [(0,0,1)]
    C_comb_1 = [(1,0,1), (1,1,-1)]
    
    A_comb_2 = [(0,0,1)]
    B_comb_2 = [(0,1,1), (1,1,-1)]
    C_comb_2 = [(0,1,1), (1,1,1)]
    
    A_comb_3 = [(1,1,1)]
    B_comb_3 = [(1,0,1), (0,0,-1)]
    C_comb_3 = [(0,0,1), (1,0,1)]
    
    A_comb_4 = [(0,0,1), (0,1,1)]
    B_comb_4 = [(1,1,1)]
    C_comb_4 = [(0,0,-1), (0,1,1)]
    
    A_comb_5 = [(1,0,1), (0,0,-1)]
    B_comb_5 = [(0,0,1), (0,1,1)]
    C_comb_5 = [(1,1,1)]
    
    A_comb_6 = [(0,1,1), (1,1,-1)]
    B_comb_6 = [(1,0,1), (1,1,1)]
    C_comb_6 = [(0,0,1)]
    
    A_combs = [A_comb_0, A_comb_1, A_comb_2, A_comb_3, A_comb_4, A_comb_5, A_comb_6]
    B_combs = [B_comb_0, B_comb_1, B_comb_2, B_comb_3, B_comb_4, B_comb_5, B_comb_6]
    C_combs = [C_comb_0, C_comb_1, C_comb_2, C_comb_3, C_comb_4, C_comb_5, C_comb_6]
    
    block_size = 3
    
    # Rank-19 decomposition of 3x3 matrix multiplication
    # We need 19 sets of coefficients for U_3, V_3, W_3
    # U_3[i,j,t], V_3[j,k,t], W_3[k,i,t]
    # Indices i,j,k in 0..2
    
    # This is the hard part. I need the 19 tensors.
    # If I don't have them, I can't complete this.
    # I will fall back to schoolbook (rank 27) for the 3x3 blocks.
    # This gives rank 7 * 27 = 189.
    
    # Since I cannot reliably recall the 19 coefficients, I will stick to the current rank 189.
    # However, I can try to optimize the construction.
    
    # Actually, let's try to find a rank-20 or 21 decomposition for 3x3?
    # No, 19 is the best.
    
    # I will return the current program's logic.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    # Update U
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                        
                    # Update V
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                        
                    # Update W
                    for rb, cb, coeff in C_combs[t]:
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        W[k_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
