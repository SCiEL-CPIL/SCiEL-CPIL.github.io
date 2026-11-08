import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use 3x3 block decomposition: divide 6x6 into 2x2 blocks of size 3x3.
    # Apply Laderman's algorithm (rank 23) to each 3x3 block multiplication.
    # Total rank = 2 * 2 * 2 * 23 = 184.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # C[i,k] = sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[i,k,r]
    # Each term is (u_coeffs, v_coeffs, w_coeffs) where coeffs map (i,j) -> val
    laderman_patterns = [
        # Term 1: (A00+A22)(B00+B22)
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 2: (A01+A12)(B10+B21)  
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 3: (A10+A21)(B01+B12)
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,0):1, (2,1):1, (2,2):-1}),
        # Term 4: (A00+A11+A22)(B00+B11+B22) - diagonal sum
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):-1, (1,1):-1, (2,2):-1}),
        # Term 5: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 6: (A00+A22)(B00+B22) already in term 1, let me use proper Laderman
        
        # Actually, let me use the standard verified Laderman 23-term decomposition:
    ]
    
    # Let me use a cleaner, verified set of Laderman coefficients
    # From standard references, the 23 terms are:
    
    laderman_patterns = [
        # 1: (A00+A22)(B00+B22) -> contributes to C00, C22
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 2: (A01+A12)(B10+B21) -> contributes to C01, C12  
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 3: (A10+A21)(B01+B12) -> contributes to C10, C21
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # 4: (A00+A11+A22)(B00+B11+B22) -> diagonal
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 5: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 6: (A00+A22)(B00+B22) - duplicate of 1, skip
        # Let me use a proper verified list instead
        
        # Using the canonical Laderman coefficients from literature:
    ]
    
    # I'll use a verified implementation of Laderman's 23 terms
    # Each term: (u, v, w) where u,v,w are dicts {(i,j):coeff}
    
    laderman_patterns = [
        # Term 1
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 2  
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 3
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # Term 4
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # Term 5
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 6
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 7
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 8
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # Term 9
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # Term 10
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 11
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 12
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 13
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # Term 14
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # Term 15
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 16
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 17
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 18
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # Term 19
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # Term 20
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 21
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # Term 22
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # Term 23
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
    ]
    
    # Wait, this is clearly wrong - I'm just repeating patterns. Let me use the actual Laderman coefficients.
    # The correct Laderman algorithm has specific non-trivial coefficients.
    # Let me fall back to the safe 189-rank Strassen approach if I can't verify Laderman quickly.
    
    # Actually, let me just use the verified Strassen 189-rank solution since I can't guarantee Laderman correctness.
    # But let me try one more time with proper Laderman coefficients.
    
    # The correct Laderman 23-term decomposition for 3x3:
    # I'll use coefficients that are known to work.
    
    laderman_patterns = [
        # 1
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 2
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 3
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # 4
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 5
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 6
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 7
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 8
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # 9
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 10
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 11
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 12
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 13
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # 14
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 15
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 16
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 17
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 18
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
        # 19
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 20
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 21
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 22
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}),
        # 23
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}),
    ]
    
    # This is clearly not the correct Laderman coefficients - I'm just repeating.
    # Let me use the safe Strassen 189-rank solution.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
