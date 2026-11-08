import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    # Use rank-10 decomposition for each 2x4x2 block, 3 blocks for 6 columns
    # Total rank = 3 * 10 = 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 8 x 30
    V = [[0] * rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    # Rank-10 decomposition for 2x4x2 matrix multiplication
    # A is 2x4, B is 4x2, C = A*B is 2x2
    # We use a known rank-10 scheme
    
    # For 2x4x2, we can think of it as two 2x2x2 multiplications with shared A columns
    # Actually, let's use a direct construction.
    
    # The key insight: 2x4x2 can be done in rank 10 using the following approach:
    # Split A = [A1, A2] where A1, A2 are 2x2
    # C = A1*B1 + A2*B2 where B1, B2 are 4x2... no that's not right.
    
    # Let me use a different approach. For 2x4x2:
    # We need to compute 4 outputs, each is a sum of 4 products = 16 scalar mults
    # Rank 10 means we need 10 rank-1 tensors.
    
    # Known result: r(2,4,2) = 10
    # Construction: use the fact that r(2,4,2) <= r(2,2,2)*2 + r(2,2,2) - ... 
    # Actually r(2,4,2) = 2*r(2,2,2) + r(1,2,2) - ... let me just hardcode.
    
    # Alternative: use Strassen's 7 for each 2x2x2, but that gives 14.
    # The rank-10 uses a more clever combination.
    
    # Let me use a verified rank-10 construction for 2x4x2.
    # I'll use the approach from the literature.
    
    # For safety, let me use rank 14 per block (7 for each 2x2x2) = 42 total, same as before.
    # But let me try to get rank 30 with a verified construction.
    
    # Actually, the simplest correct approach: for each pair of output columns,
    # we compute a 2x4 x 4x2 product. We can do each in rank 10.
    
    # Rank-10 for 2x4x2 (A: 2x4, B: 4x2, C: 2x2):
    # Let A = [a0 a1 a2 a3; a4 a5 a6 a7]
    # Let B = [b00 b01; b10 b11; b20 b21; b30 b31]
    # 
    # C00 = a0*b00 + a1*b10 + a2*b20 + a3*b30
    # C01 = a0*b01 + a1*b11 + a2*b21 + a3*b31
    # C10 = a4*b00 + a5*b10 + a6*b20 + a7*b30
    # C11 = a4*b01 + a5*b11 + a6*b21 + a7*b31
    
    # Use the following rank-10 scheme (based on known results):
    # We introduce 10 intermediate values that are products of linear combinations.
    
    # Let me use a simpler but still rank-30 approach:
    # For each 2-column block, use rank 10.
    
    # Here's a concrete rank-10 for 2x4x2:
    # s1 = (a0+a1) * (b00+b10)  -> contributes to C00
    # s2 = (a0+a1) * (b01+b11)  -> contributes to C01  
    # s3 = (a0-a1) * (b00-b10)  -> helps separate
    # s4 = (a0-a1) * (b01-b11)  -> helps separate
    # s5 = (a2+a3) * (b20+b30)  -> contributes to C00
    # s6 = (a2+a3) * (b21+b31)  -> contributes to C01
    # s7 = (a2-a3) * (b20-b30)  -> helps separate
    # s8 = (a2-a3) * (b21-b31)  -> helps separate
    # s9 = (a4+a5) * (b00+b10)  -> contributes to C10
    # s10 = (a4+a5) * (b01+b11) -> contributes to C11
    
    # But this only gives 10 products and I need to handle all 4 outputs with 4 terms each.
    # This doesn't work directly because I'm missing terms.
    
    # Let me just use the safe rank-42 solution but clean it up.
    # Or better: use rank 14 per 2-column block (Strassen on 2x2x2 twice) = 42.
    
    # I'll keep rank 42 as it's verified to work.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use the existing verified rank-42 construction
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                if time.time() > deadline:
                    break
                
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
