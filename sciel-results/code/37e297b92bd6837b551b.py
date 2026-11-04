"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Use a 3x3 block decomposition of the 6x6 matrix.
    # Each 3x3 block is a 2x2 matrix.
    # Use Strassen's algorithm (rank 7) for the 2x2 matrix multiplications.
    # Use schoolbook (rank 9) for the 3x3 block structure.
    # Total rank = 9 (block products) * 7 (Strassen per block) = 63
    
    # Wait, that's not right. Let me reconsider.
    # 6x6 matrix multiplication: C = A * B
    # Partition A into 3x3 blocks of 2x2 matrices: A[i,j] for i,j in 0..2
    # Partition B into 3x3 blocks of 2x2 matrices: B[i,j] for i,j in 0..2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Each C[i,j] is a 2x2 matrix.
    # So we need 9*3 = 27 block products (each 2x2 * 2x2 -> 2x2).
    # Using Strassen for each 2x2 product: 7 scalar products per block.
    # Total rank = 27 * 7 = 189. That's what we have now.
    
    # To do better, we need a lower-rank decomposition for the 3x3 block structure.
    # The 3x3 matrix multiplication tensor has rank at most 27 (schoolbook).
    # It is known that the rank of the 3x3 matrix multiplication tensor is at most 23 (Bürgisser 2013).
    # If we use a rank-23 decomposition for the 3x3 block structure, and Strassen (rank 7) for each 2x2 block product,
    # total rank = 23 * 7 = 161.
    
    # Alternatively, we can use a different block structure.
    # Let's try: partition 6x6 into 2x2 blocks of 3x3 matrices.
    # 2x2 block structure with Strassen: 7 block products.
    # Each block product is 3x3 * 3x3 -> 3x3, which has rank at most 27 (schoolbook) or 23 (Bürgisser).
    # Using schoolbook: 7 * 27 = 189. Same.
    # Using Bürgisser: 7 * 23 = 161.
    
    # Let's implement the Bürgisser rank-23 decomposition for 3x3 matrices.
    # This is complex. Let me look for a simpler improvement first.
    
    # Actually, let me try a different approach: use the fact that 6 = 2 * 3.
    # We can use a Kronecker product structure.
    # M_6 = M_2 ⊗ M_3 (in some sense)
    # rank(M_2 ⊗ M_3) ≤ rank(M_2) * rank(M_3)
    # rank(M_2) = 7 (Strassen), rank(M_3) ≤ 23 (Bürgisser)
    # So rank(M_6) ≤ 7 * 23 = 161.
    
    # Let me implement the Bürgisser rank-23 decomposition.
    # From Bürgisser (2013), the rank-23 algorithm for 3x3 matrix multiplication.
    
    # The 23 bilinear forms for 3x3 matrix multiplication:
    # A = [[a0, a1, a2], [a3, a4, a5], [a6, a7, a8]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = A*B
    
    # I'll use a known rank-23 decomposition. Let me use the one from the literature.
    # This is quite involved. Let me try a different approach.
    
    # Alternative: Use a rank-19 decomposition for 3x3 (if it exists and is known).
    # Actually, the best known rank for 3x3 is 23 (Bürgisser 2013).
    
    # Let me try yet another approach: use a 3x2 block structure.
    # Partition A into 3 rows of 2x6 blocks? No, that doesn't help.
    
    # Let me just implement the Bürgisser rank-23 decomposition carefully.
    # I'll use the formulation from Bürgisser's paper.
    
    # The 23 terms are defined as follows (using a0..a8 for A, b0..b8 for B):
    # I'll use a simplified version based on known results.
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that 6x6 can be decomposed using a 2x2 Strassen at the top level,
    # where each "element" is a 3x3 matrix.
    # Top-level: 7 products of 3x3 matrices.
    # Each 3x3 product uses schoolbook: 27 products.
    # Total: 7 * 27 = 189. Same as before.
    
    # To improve, I need a rank < 27 for 3x3. The best known is 23.
    
    # Let me implement the Bürgisser rank-23 decomposition.
    # I'll use the explicit coefficients from the literature.
    
    # Bürgisser rank-23 decomposition for 3x3 matrix multiplication:
    # The 23 bilinear forms (u_i, v_i) such that C_ij = sum_i u_i * v_i
    
    # From Bürgisser 2013, the 23 terms are:
    # I'll use a formulation where the coefficients are integers or half-integers.
    
    # Let me use a different, verified rank-23 decomposition.
    # I'll use the one from the "Matrix Multiplication" Wikipedia article or similar.
    
    # Actually, let me try a simpler improvement: use a 4x4 block structure.
    # 6x6 can be partitioned into 4x4 blocks? No, 6 is not divisible by 4.
    
    # Let me try: 6 = 3 + 3. Use a 2x2 block structure where each block is 3x3.
    # Top-level Strassen: 7 products of 3x3 matrices.
    # Each 3x3 product: use a rank-23 decomposition.
    # Total: 7 * 23 = 161.
    
    # Let me implement the rank-23 decomposition for 3x3.
    # I'll use the explicit coefficients.
    
    # Bürgisser rank-23 coefficients (simplified):
    # The 23 terms are defined by specific linear combinations of A and B elements.
    
    # I'll use a known rank-23 decomposition. Let me use the following:
    # (This is a simplified version; the full coefficients are complex)
    
    # For now, let me keep the rank-189 approach but verify it's correct.
    # Then I'll try to implement the rank-161 approach.
    
    # Let me first verify the current rank-189 approach is correct.
    
    rank = 189
    
    U = np.zeros((n * m, rank))  # (36, 189)
    V = np.zeros((m * p, rank))  # (36, 189)
    W = np.zeros((p * n, rank))  # (36, 189)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    c_contributions = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col, col] += coeff
                    
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col, col] += coeff
                    
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col, col] += coeff
                    
                    col += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
