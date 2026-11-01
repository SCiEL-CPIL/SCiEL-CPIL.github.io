import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=12, so 6 blocks of size 2
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Winograd 2x2x2 decomposition vectors
    # u: (i, j) -> 4 entries
    # v: (j, k) -> 2 entries per block
    # w: (k, i) -> 4 entries per block? No, w is (k, i)
    
    # The 7 terms for 2x2x2 all-ones tensor:
    # This is a known decomposition.
    # u_r, v_r, w_r
    # I will use the decomposition from the literature.
    # For 2x2x2 matrix mult, rank 7.
    
    # Vectors for U (size 4)
    # Vectors for V (size 2)
    # Vectors for W (size 4)
    
    # Term 1
    u1 = [1, 0, 0, 0] # i=0,j=0
    v1 = [1, 0]       # j=0, k=0,1
    w1 = [1, 0, 0, 0] # k=0,1; i=0,1 -> k*n+i
    # Wait, W index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Let's define the 7 terms properly.
    # This is the standard Winograd F2(2) x F2(2) decomposition.
    
    terms = [
        # u, v, w
        ([1, 0, 0, 0], [1, 0], [1, 0, 0, 0]), # 1
        ([0, 1, 0, 0], [0, 1], [0, 1, 0, 0]), # 2
        ([0, 0, 1, 0], [1, 0], [0, 0, 1, 0]), # 3
        ([0, 0, 0, 1], [0, 1], [0, 0, 0, 1]), # 4
        ([1, 1, 0, 0], [1, 1], [1, 1, 1, 1]), # 5
        ([1, 0, 1, 0], [1, 0], [1, 1, 1, 1]), # 6
        ([0, 1, 0, 1], [0, 1], [1, 1, 1, 1]), # 7
    ]
    
    # These are the standard Winograd terms for 2x2x2.
    # Let's verify they sum to the all-ones tensor.
    # For (i=0, j=0, k=0):
    # T = sum u[i,j] * v[j,k] * w[k,i]
    # r1: 1*1*1 = 1
    # r2: 0
    # r3: 0
    # r4: 0
    # r5: 1*1*1 = 1
    # r6: 1*1*1 = 1
    # r7: 0
    # Sum = 3.
    # We need 1.
    # So we need to scale.
    # But Winograd uses half-integers.
    # The reconstruction coefficients are not 1.
    # In the tensor decomposition T = sum u v w, the w vector contains the reconstruction coefficients.
    # So I need to solve for w.
    
    # Actually, I can just use the schoolbook for each block (rank 8) -> 48.
    # But I want 42.
    # Let's use the rank 7 decomposition with correct coefficients.
    # I will use the known rank 7 decomposition for 2x2x2 matrix mult.
    
    # Correct Winograd coefficients:
    # m1 = a*x
    # m2 = b*z
    # m3 = c*x
    # m4 = d*w
    # m5 = (a+b)*(x+z)
    # m6 = (a+c)*x
    # m7 = (b+d)*z
    # C00 = (m1 + m2 + m5 - m3 - m4 + ...)/2 ?
    
    # Let's just use the rank 8 (schoolbook) for now to be safe, or try to implement the 7 correctly.
    # Given the time constraint, I will implement the rank 42 using the 7-term Winograd structure.
    # I will hardcode the U, V, W matrices for one block and repeat.
    
    # Block U (4x7)
    # Block V (2x7)
    # Block W (4x7)
    
    # Using the terms above, but with correct W scaling.
    # For 2x2x2, the rank 7 decomposition is:
    # u1=(1,0,0,0), v1=(1,0), w1=(1,0,0,0)
    # u2=(0,1,0,0), v2=(0,1), w2=(0,1,0,0)
    # u3=(0,0,1,0), v3=(1,0), w3=(0,0,1,0)
    # u4=(0,0,0,1), v4=(0,1), w4=(0,0,0,1)
    # u5=(1,1,0,0), v5=(1,1), w5=(0.5, 0.5, 0.5, 0.5)
    # u6=(1,0,1,0), v6=(1,0), w6=(0.5, 0.5, 0.5, 0.5)
    # u7=(0,1,0,1), v7=(0,1), w7=(0.5, 0.5, 0.5, 0.5)
    
    # Let's check (0,0,0):
    # r1: 1*1*1 = 1
    # r5: 1*1*0.5 = 0.5
    # r6: 1*1*0.5 = 0.5
    # Sum = 2.
    # Still not 1.
    
    # I will use the schoolbook rank 48 to ensure correctness, as finding the exact rank 7 coefficients quickly is risky.
    # However, 42 is a significant improvement.
    # Let's use the rank 7 decomposition from a reliable source.
    # The tensor is the all-ones tensor.
    # Rank 7 decomposition of all-ones 2x2x2:
    # u1=(1,0,0,0), v1=(1,0), w1=(1,0,0,0)
    # u2=(0,1,0,0), v2=(0,1), w2=(0,1,0,0)
    # u3=(1,1,0,0), v3=(1,1), w3=(1,1,1,1)
    # u4=(1,1,1,1), v4=(1,1), w4=(1,1,1,1)
    # This is not it.
    
    # I will stick to the schoolbook 48 to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
