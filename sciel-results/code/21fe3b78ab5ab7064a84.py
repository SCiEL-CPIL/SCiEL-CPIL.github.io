"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We aim for a rank 56 decomposition using a Kronecker product strategy.
    # Strategy: 
    # 1. Split the inner dimension m=4 into two parts of size 2.
    # 2. This gives two sub-problems: 2x2 * 2x5 and 2x2 * 2x5.
    # 3. We will use a high-rank decomposition for each 2x2 * 2x5 sub-problem.
    # 4. To align with the "Kronecker" theme, we model each 2x2 * 2x5 
    #    as part of a larger 4x4 * 4x4 structure, but we need to handle the 5th column.
    #    
    # Let's use a simple valid decomposition of rank 56.
    # Trivial rank is 40. We can add 16 zero terms to reach 56?
    # Or use a structured decomposition.
    # 
    # Let's use the "Strassen for 2x2x2, Standard for 2x2x2" Kronecker product 
    # which gives rank 28 for 4x4x4.
    # We have two 2x2x2 blocks in the input A (left and right).
    # We have two 2x5 blocks in the input B (top and bottom).
    # C = A_L B_T + A_R B_B.
    # 
    # If we use rank 28 for A_L B_T and rank 28 for A_R B_B, total rank 56.
    # How to get rank 28 for 2x2 * 2x5?
    # We can pad 2x5 to 4x4? No.
    # We can use 28 terms for 2x2 * 2x5.
    # The trivial rank for 2x2 * 2x5 is 20.
    # So we can just use 20 terms and add 8 zero terms?
    # 
    # Let's construct a rank 56 decomposition explicitly.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    # Sub-problem 1: A_L (2x2) * B_T (2x5)
    # A_L indices: (0,0), (0,1), (1,0), (1,1)
    # B_T indices: (0,0)..(0,4), (1,0)..(1,4)
    # We will use 28 terms for this part.
    # First 20 terms: Trivial multiplication
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(k, i)][col] += 1
                col += 1
                
    # Add 8 zero terms to reach rank 28 for this sub-problem
    for _ in range(8):
        col += 1
        
    # Sub-problem 2: A_R (2x2) * B_B (2x5)
    # A_R indices: (0,2), (0,3), (1,2), (1,3)
    # B_B indices: (2,0)..(2,4), (3,0)..(3,4)
    # We will use 28 terms for this part.
    # First 20 terms: Trivial multiplication
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[u_idx(i, j+2)][col] = 1
                V[v_idx(j+2, k)][col] = 1
                W[w_idx(k, i)][col] += 1
                col += 1
                
    # Add 8 zero terms to reach rank 28 for this sub-problem
    for _ in range(8):
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
