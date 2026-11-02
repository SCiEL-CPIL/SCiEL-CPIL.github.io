"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _get_schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition for matrix multiplication.
    The tensor T_{ijk} = sum_r U_{ir, r} V_{jr, r} W_{kr, r} must be 1 if indices match multiplication, 0 otherwise.
    Actually, the standard decomposition is:
    C_{ik} = sum_j A_{ij} B_{jk}
    Tensor form: T_{ijk} = 1.
    Decomposition: T_{ijk} = sum_r U_{i, r} V_{j, r} W_{k, r}
    Where U is (n x rank), V is (m x rank), W is (p x rank).
    But the interface requires U (n*m x rank), V (m*p x rank), W (p*n x rank).
    Let's map:
    Interface U[row = i*m+j, col=r]
    Interface V[row = j*p+k, col=r]
    Interface W[row = k*n+i, col=r]
    
    The condition is:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    Let's verify this.
    """
    rank = U.shape[1]
    # We need to check all 216 triples (i, j, k)
    # Since rank is small (216), and 216 triples, this is fast.
    # But we can do it vectorized for speed if needed.
    # For now, loop is fine for 216 * 216 ops.
    
    # Optimization: Pre-compute V and W products?
    # No, just loop.
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                # Compute sum of element-wise products
                prod = u_row * v_row * w_row
                s = np.sum(prod)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Initialize with Schoolbook
    U, V, W = _get_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # 2. Random Search for Rank Reduction
    # Strategy: Try to merge two columns (terms) into one by adding random coefficients.
    # If U[:, a] and U[:, b] are linearly dependent (or can be merged), we can reduce rank.
    # Actually, a simpler heuristic:
    # Pick two random columns a, b.
    # Try to replace them with a single column c such that the tensor is preserved.
    # This is hard.
    
    # Alternative Strategy: Random Linear Combination of Terms
    # We know Schoolbook is valid.
    # Can we find a lower rank decomposition?
    # Let's try a "Greedy Merging" approach.
    # If we have terms t1 and t2.
    # t1: (u1, v1, w1)
    # t2: (u2, v2, w2)
    # We want to see if we can represent the contribution of t1 and t2 with a single term t3.
    # This is generally impossible for arbitrary terms.
    
    # Better Strategy: Random Search in the space of rank-r decompositions.
    # Start with r = 216.
    # Try to reduce r by 1, 2, ...
    # For each r, generate random U, V, W and check if they form a valid decomposition.
    # This is unlikely to succeed with pure random init.
    
    # Let's use a "Perturbation" approach on the Schoolbook.
    # The Schoolbook has 216 terms.
    # Many of these terms are "independent" in the tensor space.
    # However, we can try to find a basis for the tensor space that is smaller.
    
    # Let's try a specific known decomposition if we can recall one?
    # For 6x6x6, the best known rank is likely close to 216 or slightly lower if using Strassen.
    # Strassen for 2x2 blocks on 6x6 (3x3 blocks of 2x2):
    # 6x6 = (3*2) x (3*2).
    # If we view 6x6 as 3x3 blocks of 2x2 matrices.
    # Block matrix multiplication: C_block = sum_k A_block_k * B_block_k.
    # Rank of block mult is 3*3*3 * Rank(2x2 mult).
    # Rank(2x2 mult) = 7 (Strassen).
    # So Rank(6x6 mult) <= 27 * 7 = 189.
    
    # Let's implement Strassen for 3x3 blocks of 2x2.
    # This is a structural change from the current 2x2 blocks of 3x3 (which was 7 * 27 = 189? No, wait.)
    # Current code: `_strassen_2x2_blocks` splits 6x6 into 2x2 blocks of 3x3.
    # Block size 3x3. Rank(3x3 schoolbook) = 27.
    # Total rank = 7 * 27 = 189.
    # The current code *does* implement this and verifies it.
    # Why did it score 1.0?
    # Maybe the verification failed? Or the code returned Schoolbook?
    # The current code returns Strassen if verified.
    # If it scored 1.0, it means the score was 216/216? Or maybe the score is relative to something else?
    # "score is the rank". "1.0 equals the seed program".
    # Seed program is Schoolbook (216).
    # If Strassen (189) worked, score would be 216/189 > 1.0.
    # Since score is 1.0, Strassen likely failed verification or wasn't returned.
    
    # Let's re-examine the current code's Strassen implementation.
    # It uses `c_combos` and `steps`.
    # The logic seems complex and prone to indexing errors.
    
    # My new strategy: Implement Strassen for 3x3 blocks of 2x2 explicitly and correctly.
    # This should yield rank 189.
    
    def _strassen_3x3_blocks_of_2x2(n, m, p):
        # n=6, m=6, p=6
        # Blocks are 2x2.
        # Number of blocks: 3x3.
        # A is 3x3 blocks of 2x2.
        # B is 3x3 blocks of 2x2.
        # C is 3x3 blocks of 2x2.
        
        # Strassen formulas for 3x3 block matrix multiplication?
        # Standard Strassen is for 2x2 blocks.
        # For 3x3 blocks, we can use the "Bini" or other 3x3 matrix multiplication algorithms?
        # Or just apply Strassen recursively?
        # 3x3 matrix multiplication can be done with rank 23 (Winograd) or 21?
        # Actually, the rank of 3x3 matrix multiplication is 23 (Winograd 1971).
        # If we use Winograd for the 3x3 block structure, and Strassen for the 2x2 inside:
        # Total Rank = Rank(3x3 block mult) * Rank(2x2 scalar mult)
        # Rank(3x3 block mult) = 23 (Winograd)
        # Rank(2x2 scalar mult) = 7 (Strassen)
        # Total Rank = 23 * 7 = 161.
        
        # This is a significant improvement over 189!
        
        # Let's implement Winograd for 3x3 block multiplication.
        # Winograd's algorithm for 3x3 matrix multiplication uses 23 multiplications.
        
        # The 3x3 matrix multiplication tensor has rank 23.
        # We need the explicit decomposition for 3x3.
        # This is complex to hardcode.
        
        # Alternative: Use the known rank 23 decomposition for 3x3.
        # Or, simpler: Just use Strassen on the 2x2 level for the whole 6x6?
        # If we view 6x6 as 2x2 blocks of 3x3.
        # Rank(3x3 schoolbook) = 27.
        # Total Rank = 7 * 27 = 189.
        
        # If we view 6x6 as 3x3 blocks of 2x2.
        # Rank(2x2 schoolbook) = 8.
        # Total Rank = Rank(3x3 mult) * 8.
        # If we use Winograd (23) for 3x3: 23 * 8 = 184.
        # If we use Strassen (7) for 2x2: We need Rank(3x3 mult) * 7.
        # But the 3x3 mult operates on "scalars" which are 2x2 matrices.
        # So we need a decomposition of 3x3 matrix multiplication where the "multiplication" operation is 2x2 matrix multiplication (rank 7).
        # This is exactly what the tensor product structure allows.
        
        # So, can we find a rank 23 decomposition for 3x3 matrix multiplication?
        # Yes, Winograd's algorithm.
        # But implementing Winograd's specific 23-term decomposition is tricky.
        
        # Let's try a simpler structural improvement first:
        # The current code's Strassen (189) might have a bug.
        # Let's write a clean Strassen for 2x2 blocks of 3x3.
        
        # Block size: 3x3.
        # A: 2x2 blocks of 3x3.
        # B: 2x2 blocks of 3x3.
        # C: 2x2 blocks of 3x3.
        
        # Strassen Steps:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each "multiplication" (X)(Y) is a 3x3 * 3x3 multiplication.
        # We use Schoolbook for 3x3 * 3x3 (Rank 27).
        
        # Total Rank = 7 * 27 = 189.
        
        # Let's implement this cleanly.
        
        U = np.zeros((n * m, 189))
        V = np.zeros((m * p, 189))
        W = np.zeros((p * n, 189))
        
        # Block indices:
        # 0: (0,0) -> rows 0-2, cols 0-2
        # 1: (0,1) -> rows 0-2, cols 3-5
        # 2: (1,0) -> rows 3-5, cols 0-2
        # 3: (1,1) -> rows 3-5, cols 3-5
        
        # Helper to get block row/col ranges
        def get_block_range(idx):
            r = idx // 2
            c = idx % 2
            return (r * 3, (r + 1) * 3, c * 3, (c + 1) * 3) # r_start, r_end, c_start, c_end
        
        # Define the 7 Strassen steps
        # Each step is a combination of A blocks and B blocks.
        # We need to generate the U, V, W entries for each term in the schoolbook expansion of each block multiplication.
        
        # For a block multiplication M = X * Y, where X is A_block_a (or sum) and Y is B_block_b (or sum).
        # X is 3x3, Y is 3x3.
        # Schoolbook: M_{ik} = sum_j X_{ij} Y_{jk}.
        # Term (i, j, k) in the 3x3 block.
        # Global indices:
        # A row: block_row_start + i
        # A col: block_col_start + j
        # B row: block_row_start + j
        # B col: block_col_start + k
        
        # The contribution to C is at global C row: C_block_row_start + i, C col: C_block_col_start + k.
        # C block index is determined by the result of the Strassen step.
        
        # Let's define the steps explicitly.
        # step: (a_coeffs, b_coeffs, c_block_target, c_sign)
        # Wait, the C combination is separate.
        # Strassen computes M1..M7. Then C is linear combo of M's.
        
        # Let's iterate over the 7 M's.
        # For each M_s, we have A_part and B_part.
        # A_part is a linear combination of A blocks.
        # B_part is a linear combination of B blocks.
        
        # For each (a_block, a_sign) in A_part and (b_block, b_sign) in B_part:
        # We perform schoolbook multiplication of A_block and B_block.
        # Each term in this multiplication contributes to the final C with a specific sign.
        
        # The sign for C is determined by which M_s it is and which C block it falls into.
        
        # Strassen Formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We will build the U, V, W matrices.
        
        steps = [
            # (A_coeffs, B_coeffs)
            ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
            ({2: 1, 3: 1}, {0: 1}),       # M2
            ({0: 1}, {1: 1, 3: -1}),      # M3
            ({3: 1}, {2: 1, 0: -1}),      # M4
            ({0: 1, 1: 1}, {3: 1}),       # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}), # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1})  # M7
        ]
        
        # C combination: c_block_idx -> {m_idx: sign}
        c_combos = [
            {0: 1, 3: -1, 5: 1, 6: -1}, # C00 (M1, M4, M6, M7)
            {2: 1, 4: 1},                # C01 (M3, M5)
            {1: 1, 3: 1},                # C10 (M2, M4)
            {0: 1, 1: -1, 2: 1, 6: 1}   # C11 (M1, M2, M3, M7)
        ]
        
        block_rank = 27 # 3*3*3
        total_rank = 7 * block_rank
        
        for s in range(7):
            a_coeffs, b_coeffs = steps[s]
            for ai, a_sign in a_coeffs.items():
                for bi, b_sign in b_coeffs.items():
                    # Schoolbook multiplication of A_block_ai and B_block_bi
                    # A_block_ai: rows [ai//2 * 3, ai//2 * 3 + 3), cols [ai%2 * 3, ai%2 * 3 + 3)
                    # B_block_bi: rows [bi//2 * 3, bi//2 * 3 + 3), cols [bi%2 * 3, bi%2 * 3 + 3)
                    
                    a_r0, a_r1, a_c0, a_c1 = get_block_range(ai)
                    b_r0, b_r1, b_c0, b_c1 = get_block_range(bi)
                    
                    # Loop over i, j, k in 0..2
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                # Global A index: (a_r0 + i, a_c0 + j)
                                ga_row = a_r0 + i
                                ga_col = a_c0 + j
                                
                                # Global B index: (b_r0 + j, b_c0 + k)
                                gb_row = b_r0 + j
                                gb_col = b_c0 + k
                                
                                # Global C index: (c_r0 + i, c_c0 + k)
                                # C block is determined by ga_row // 3 and gb_col // 3
                                c_block_row = ga_row // 3
                                c_block_col = gb_col // 3
                                c_block_idx = c_block_row * 2 + c_block_col
                                
                                c_sign = c_combos[c_block_idx].get(s, 0)
                                if c_sign == 0:
                                    continue
                                
                                # Column index in the decomposition
                                col = s * block_rank + i * 9 + j * 3 + k
                                
                                U[ga_row * m + ga_col, col] = a_sign
                                V[gb_row * p + gb_col, col] = b_sign
                                W[gb_col * n + ga_row, col] = c_sign # Note: W is (p*n, rank), index is k*n + i.
                                # Here gb_col is k (global), ga_row is i (global).
                                # So W index is gb_col * n + ga_row. Correct.
        
        return U, V, W

    # Try Strassen 3x3 blocks of 2x2 (Rank 189)
    try:
        U_s, V_s, W_s = _strassen_3x3_blocks_of_2x2(n, m, p)
        if _verify_decomp(U_s, V_s, W_s, n, m, p):
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = U_s.shape[1]
    except Exception:
        pass

    # If we have time, try to improve further?
    # For now, 189 is a solid improvement over 216.
    
    # Convert to list of lists for the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
