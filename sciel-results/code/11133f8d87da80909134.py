"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Rank-60 decomposition for 4x4 * 4x6 matrix multiplication
    # Strategy: split the 6 columns of B into two 3-column blocks
    # For each 3-column block, split into a 2-column part and a 1-column part
    # Actually, let's use a simpler approach:
    # Split B into two 3x4 blocks (B1: cols 0-2, B2: cols 3-5)
    # For B1 (4x3): use naive multiplication -> 4*4*3 = 48 mults
    # For B2 (4x3): use naive multiplication -> 4*4*3 = 48 mults
    # Total: 96 - too high
    
    # Better approach: Use Strassen-like for 4x4x4 part
    # Split C into two 4x3 blocks
    # For each 4x3 block, we need to compute A (4x4) * B_block (4x3)
    # This is equivalent to 4x4x3 multiplication
    
    # Let's use a different strategy:
    # Treat the problem as 4x4 * 4x6
    # Split into: 4x4 * 4x4 (first 4 cols) + 4x4 * 4x2 (last 2 cols)
    # For 4x4x4: use Strassen with rank 28 (7 Strassen mults, each 2x2x2 naive = 4 mults)
    # For 4x4x2: naive = 4*4*2 = 32 mults
    # Total: 28 + 32 = 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 16 x 60
    V = [[0] * rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * rank for _ in range(p * n)]  # 24 x 60
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B, columns 0-3 of C)
    # Use Strassen decomposition for 4x4x4
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # Split C into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    
    # Strassen 7 multiplications for 4x4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = A22 * B11
    # M3 = (A11 + A12) * B21
    # M4 = (A21 + A22) * B12
    # M5 = (A11 + A21) * B22
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A21 - A11)(B11 + B12)
    
    # Each Mi is a 2x2 * 2x2 product, done naively (4 mults each)
    # Total: 7 * 4 = 28 mults
    
    # Element indices for A (row-major: row*m + col)
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Element indices for B (row-major: row*p + col)
    # B11: rows 0-1, cols 0-1 -> indices 0,1,12,13
    # B12: rows 0-1, cols 2-3 -> indices 2,3,14,15
    # B21: rows 2-3, cols 0-1 -> indices 24,25,36,37
    # B22: rows 2-3, cols 2-3 -> indices 26,27,38,39
    
    # Element indices for C (W convention: col*n + row)
    # C11: rows 0-1, cols 0-1 -> W indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> W indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> W indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> W indices 10,11,14,15
    
    # Define Strassen combinations
    # A combinations (indices in A):
    # S1 = A11 + A22: indices [0,1,4,5] + [10,11,14,15]
    # S2 = A22: indices [10,11,14,15]
    # S3 = A11 + A12: indices [0,1,4,5] + [2,3,6,7]
    # S4 = A21 + A22: indices [8,9,12,13] + [10,11,14,15]
    # S5 = A11 + A21: indices [0,1,4,5] + [8,9,12,13]
    # S6 = A12 - A22: indices [2,3,6,7] - [10,11,14,15]
    # S7 = A21 - A11: indices [8,9,12,13] - [0,1,4,5]
    
    # B combinations (indices in B):
    # T1 = B11 + B22: indices [0,1,12,13] + [26,27,38,39]
    # T2 = B11: indices [0,1,12,13]
    # T3 = B21 + B22: indices [24,25,36,37] + [26,27,38,39]
    # T4 = B12 - B22: indices [2,3,14,15] - [26,27,38,39]
    # T5 = B11 + B12: indices [0,1,12,13] + [2,3,14,15]
    # T6 = B21 - B11: indices [24,25,36,37] - [0,1,12,13]
    # T7 = B12 + B22: indices [2,3,14,15] + [26,27,38,39]
    
    # For each Strassen multiplication Mi = Si * Ti, we do naive 2x2x2
    # Result Pi is a 2x2 matrix that contributes to C blocks
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the 7 Strassen products and their contributions
    
    # Product 1: M1 = (A11+A22)(B11+B22)
    # A side: A11+A22 -> indices [0,1,4,5,10,11,14,15] all with +1
    # B side: B11+B22 -> indices [0,1,12,13,26,27,38,39] all with +1
    # Naive 2x2x2: 4 mults
    # (A11[0,0]+A22[0,0])*(B11[0,0]+B22[0,0]) -> C11[0,0] += 1, C22[0,0] += 1
    # Wait, this is getting complex. Let me use a helper function.
    
    def add_strassen_product(a_indices, a_signs, b_indices, b_signs, c_indices, c_signs, current_col):
        """Add a naive 2x2x2 product where A and B are linear combinations."""
        nonlocal col
        # For each element in the 2x2 result
        for ci in range(4):
            # ci corresponds to (row, col) in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            c_row = ci // 2
            c_col = ci % 2
            c_idx = c_indices[ci]
            
            # For each pair (a_idx, b_idx) that contributes
            for ai in range(len(a_indices)):
                for bi in range(len(b_indices)):
                    # Check if this pair contributes to this c element
                    # In naive 2x2x2: C[i,j] = sum_k A[i,k] * B[k,j]
                    # So for c at (c_row, c_col), we need a at (c_row, k) and b at (k, c_col)
                    # But our a_indices and b_indices are for the full 2x2 matrices
                    # This approach is too complex. Let me use a different method.
                    pass
        return current_col
    
    # Let me use a simpler approach: directly enumerate all 28 mults
    
    # I'll define the 7 Strassen combinations explicitly and then do naive multiplication
    
    # A combinations (as list of (index, sign)):
    A_combos = [
        [(0,1),(1,1),(4,1),(5,1),(10,1),(11,1),(14,1),(15,1)],  # S1 = A11+A22
        [(10,1),(11,1),(14,1),(15,1)],  # S2 = A22
        [(0,1),(1,1),(4,1),(5,1),(2,1),(3,1),(6,1),(7,1)],  # S3 = A11+A12
        [(8,1),(9,1),(12,1),(13,1),(10,1),(11,1),(14,1),(15,1)],  # S4 = A21+A22
        [(0,1),(1,1),(4,1),(5,1),(8,1),(9,1),(12,1),(13,1)],  # S5 = A11+A21
        [(2,1),(3,1),(6,1),(7,1),(10,-1),(11,-1),(14,-1),(15,-1)],  # S6 = A12-A22
        [(8,1),(9,1),(12,1),(13,1),(0,-1),(1,-1),(4,-1),(5,-1)],  # S7 = A21-A11
    ]
    
    # B combinations (as list of (index, sign)):
    B_combos = [
        [(0,1),(1,1),(12,1),(13,1),(26,1),(27,1),(38,1),(39,1)],  # T1 = B11+B22
        [(0,1),(1,1),(12,1),(13,1)],  # T2 = B11
        [(24,1),(25,1),(36,1),(37,1),(26,1),(27,1),(38,1),(39,1)],  # T3 = B21+B22
        [(2,1),(3,1),(14,1),(15,1),(26,-1),(27,-1),(38,-1),(39,-1)],  # T4 = B12-B22
        [(0,1),(1,1),(12,1),(13,1),(2,1),(3,1),(14,1),(15,1)],  # T5 = B11+B12
        [(24,1),(25,1),(36,1),(37,1),(0,-1),(1,-1),(12,-1),(13,-1)],  # T6 = B21-B11
        [(2,1),(3,1),(14,1),(15,1),(26,1),(27,1),(38,1),(39,1)],  # T7 = B12+B22
    ]
    
    # C block contributions (for each of the 7 products, which C blocks they contribute to and with what sign)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C block indices (W convention: col*n + row):
    # C11: rows 0-1, cols 0-1 -> W[0], W[1], W[4], W[5]
    # C12: rows 0-1, cols 2-3 -> W[2], W[3], W[6], W[7]
    # C21: rows 2-3, cols 0-1 -> W[8], W[9], W[12], W[13]
    # C22: rows 2-3, cols 2-3 -> W[10], W[11], W[14], W[15]
    
    C_blocks = {
        'C11': [0, 1, 4, 5],
        'C12': [2, 3, 6, 7],
        'C21': [8, 9, 12, 13],
        'C22': [10, 11, 14, 15]
    }
    
    # For each of the 7 Strassen products, do naive 2x2x2 multiplication
    # Product i = A_combo[i] * B_combo[i]
    # The result is a 2x2 matrix Pi
    # Pi contributes to C blocks as specified
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for product i (0-indexed), the contributions are:
    # Product 0 (M1): C11 += Pi, C22 += Pi
    # Product 1 (M2): C21 += Pi, C22 -= Pi
    # Product 2 (M3): C12 += Pi, C22 += Pi
    # Product 3 (M4): C11 += Pi, C21 += Pi
    # Product 4 (M5): C11 -= Pi, C12 += Pi
    # Product 5 (M6): C22 += Pi
    # Product 6 (M7): C11 -= Pi
    
    contributions = [
        {'C11': 1, 'C22': 1},  # M1
        {'C21': 1, 'C22': -1},  # M2
        {'C12': 1, 'C22': 1},  # M3
        {'C11': 1, 'C21': 1},  # M4
        {'C11': -1, 'C12': 1},  # M5
        {'C22': 1},  # M6
        {'C11': -1},  # M7
    ]
    
    # For each product, do naive 2x2x2
    for prod_idx in range(7):
        a_combo = A_combos[prod_idx]
        b_combo = B_combos[prod_idx]
        contrib = contributions[prod_idx]
        
        # Naive 2x2x2: for each output element (i,j) in 2x2
        for i in range(2):
            for j in range(2):
                # C[i,j] = sum_k A[i,k] * B[k,j]
                # But A and B are linear combinations, so we need to expand
                # For each k in 0..1:
                for k in range(2):
                    # A[i,k] is a linear combination of A elements
                    # B[k,j] is a linear combination of B elements
                    # The product is a bilinear form
                    
                    # For each term in A[i,k]:
                    # For each term in B[k,j]:
                    # Add to U, V, W
                    
                    # But this would create too many terms.
                    # Instead, let's precompute the linear combinations as separate rows/cols
                    
                    pass
        
        # This approach is getting too complex. Let me use a different strategy.
        # I'll just use naive multiplication for the entire 4x4x4 part (rank 64)
        # and naive for 4x4x2 (rank 32), total 96.
        # But that's worse than 84.
        
        # Let me try a different approach: use the existing rank-84 solution
        # but try to optimize it.
        
        break
    
    # Given the complexity, let me just return the rank-84 solution from the current program
    # as it's known to be valid.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for k in range(3):
            for j in range(2):
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
