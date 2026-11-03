"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 4*4*8 = 128
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Build schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition using Strassen-style block decomposition
    # We can decompose the 4x4x8 multiplication into blocks.
    # Think of A as 4x4, B as 4x8. We can split B into two 4x4 blocks: B = [B1 | B2]
    # Then C = A*B = [A*B1 | A*B2]
    # Each 4x4x4 multiplication has rank at most 49 (actually Strassen gives 7 for 2x2x2, but for 4x4x4 we can use Strassen recursively)
    # Actually, for 4x4x4, we can use Strassen's algorithm on 2x2 blocks of 2x2 matrices.
    # 2x2x2 Strassen has rank 7. So 4x4x4 = 2x2 blocks of 2x2x2, each with rank 7, total rank = 7*4 = 28? No.
    # Actually, when we do Strassen on 2x2 blocks, we have 4 block multiplications, each 2x2x2 with rank 7.
    # But the combinations are linear, so total rank is 7 * 4 = 28? Let me think more carefully.
    
    # For a 4x4x4 matrix multiplication, we can view it as multiplying 2x2 block matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B, where each block is 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Using Strassen:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    # Each Pi is a 2x2x2 multiplication with rank 7.
    # So total rank for 4x4x4 is 7*7 = 49? No, each Pi involves one 2x2x2 mult, which has rank 7.
    # There are 7 Pi's, so total rank = 7*7 = 49.
    
    # Wait, that's not right either. Each Pi is computed as a 2x2 matrix multiplication, which requires rank 7 (Strassen).
    # So for each Pi, we need 7 scalar multiplications. With 7 Pi's, total = 49 scalar multiplications.
    # So 4x4x4 has rank at most 49.
    
    # For our problem: 4x4x8 = [4x4x4 | 4x4x4], so rank at most 49*2 = 98.
    # But we can do better by sharing structure.
    
    # Let me try a direct approach: use Strassen on the 4x4x8 problem by viewing it as 2x2 block multiplication
    # where A is 4x4 = [[A1, A2]] (each 2x4) and B is 4x8 = [[B1], [B2]] (each 2x8).
    # C = A*B = [A1*B1 + A2*B2] which is 4x8.
    # Hmm, this doesn't decompose nicely into independent blocks.
    
    # Let me try a different approach: directly construct a rank-98 decomposition using 4x4x4 Strassen.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Build a 4x4x4 Strassen decomposition
    # For 2x2x2 Strassen, the 7 rank-1 terms are:
    # (A11+A22)(B11+B22), (A21+A22)B11, A11(B12-B22), A22(B21-B11),
    # (A11+A12)B22, (A21-A11)(B11+B12), (A12-A22)(B21+B22)
    
    # For 2x2x2, U indices: (i,j) -> i*2+j, V indices: (j,k) -> j*2+k, W indices: (k,i) -> k*2+i
    # Strassen rank-1 terms for 2x2x2:
    # Term 1: U: (0,0)=1,(1,1)=1; V: (0,0)=1,(0,1)=1,(1,0)=1,(1,1)=1; W: (0,0)=1,(0,1)=1,(1,0)=1,(1,1)=1
    # Actually let me be more careful.
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, it's a product of a linear combo of A entries and a linear combo of B entries.
    # Mi = (sum_i u_i * A[i]) * (sum_j v_j * B[j]) where the result contributes to C entries.
    # Actually, Mi = uA * uB where uA is a vector over A entries and uB is a vector over B entries.
    # The contribution to C is w * Mi where w is a vector over C entries.
    # In the tensor decomposition: U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
    # where a_idx indexes A, b_idx indexes B, c_idx indexes C.
    
    # For 2x2x2:
    # A entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W convention: W[k*n+i, r] for n=2, so W[i*2+k, r]? Wait, the problem says W[k*n+i, r].
    # For 2x2x2: W[k*2+i, r]
    
    # Let me define the Strassen terms for 2x2x2:
    strassen_2x2x2 = []
    # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
    strassen_2x2x2.append((
        {0: 1, 3: 1},  # U: A00, A11
        {0: 1, 1: 1, 2: 1, 3: 1},  # V: B00, B01, B10, B11
        {0: 1, 3: 1}  # W: C00, C11 (W indices: k*2+i, C00: k=0,i=0 -> 0, C11: k=1,i=1 -> 3)
    ))
    # M2 = (A10+A11)B00, contributes to C10(+1), C11(-1)
    strassen_2x2x2.append((
        {2: 1, 3: 1},  # U: A10, A11
        {0: 1},  # V: B00
        {2: 1, 3: -1}  # W: C10(k=0,i=1->2), C11(k=1,i=1->3)
    ))
    # M3 = A00(B01-B11), contributes to C01(+1), C11(+1)
    strassen_2x2x2.append((
        {0: 1},  # U: A00
        {1: 1, 3: -1},  # V: B01, B11
        {1: 1, 3: 1}  # W: C01(k=0,i=1->2? no, C01: i=0,k=1 -> k*2+i = 1*2+0 = 2)
    ))
    # Wait, I need to be more careful with W indexing.
    # W[k*n+i, r] where n=2.
    # C[i,k] is at position k*n+i = k*2+i in W.
    # C00: i=0,k=0 -> 0
    # C01: i=0,k=1 -> 2
    # C10: i=1,k=0 -> 1
    # C11: i=1,k=1 -> 3
    
    strassen_2x2x2 = []
    # M1 = (A00+A11)(B00+B11), C00+=1, C11+=1
    strassen_2x2x2.append(({0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}, {0:1, 3:1}))
    # M2 = (A10+A11)B00, C10+=1, C11+=-1
    strassen_2x2x2.append(({2:1, 3:1}, {0:1}, {1:1, 3:-1}))
    # M3 = A00(B01-B11), C01+=1, C11+=1
    strassen_2x2x2.append(({0:1}, {1:1, 3:-1}, {2:1, 3:1}))
    # M4 = A11(B10-B00), C10+=1, C00+=1
    strassen_2x2x2.append(({3:1}, {2:1, 0:-1}, {1:1, 0:1}))
    # M5 = (A00+A01)B11, C01+=1, C00+=-1
    strassen_2x2x2.append(({0:1, 1:1}, {3:1}, {2:1, 0:-1}))
    # M6 = (A10-A00)(B00+B01), C11+=1
    strassen_2x2x2.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7 = (A01-A11)(B10+B11), C00+=1
    strassen_2x2x2.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    
    # Now build 4x4x4 using Strassen on 2x2 blocks.
    # 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    # A entries indexed by (i,j) -> i*4+j
    # B entries indexed by (j,k) -> j*4+k
    # C entries indexed by (i,k) -> k*4+i (W convention: k*n+i with n=4)
    
    # Block structure:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B = [[B11, B12], [B21, B22]]
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C = [[C11, C12], [C21, C22]]
    # C11: rows 0-1, cols 0-1 -> W indices: k*4+i for i in 0-1, k in 0-1 -> 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> W indices: k*4+i for i in 0-1, k in 2-3 -> 8,9,12,13
    # C21: rows 2-3, cols 0-1 -> W indices: k*4+i for i in 2-3, k in 0-1 -> 2,3,6,7
    # C22: rows 2-3, cols 2-3 -> W indices: k*4+i for i in 2-3, k in 2-3 -> 10,11,14,15
    
    def block_indices(block_type, block_pos, size=4):
        """Get indices for a 2x2 block in a 4x4 matrix."""
        # block_pos: 0=11, 1=12, 2=21, 3=22
        row_start = (block_pos // 2) * 2
        col_start = (block_pos % 2) * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                if block_type == 'A':  # A[i,j]
                    indices.append((row_start + dr) * size + (col_start + dc))
                elif block_type == 'B':  # B[j,k]
                    indices.append((row_start + dr) * size + (col_start + dc))
                elif block_type == 'C':  # C[i,k] -> W[k*n+i]
                    i = row_start + dr
                    k = col_start + dc
                    indices.append(k * size + i)
        return indices
    
    # Build 4x4x4 Strassen decomposition
    # Each Strassen term Mi involves a product of linear combos of A blocks and B blocks.
    # For each Mi in the 2x2x2 Strassen, we need to expand it into 2x2x2 Strassen terms.
    
    # The 7 Mi's for 4x4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mi is a 2x2 matrix multiplication (of 2x2 blocks), so each has rank 7.
    # Total rank for 4x4x4: 7*7 = 49.
    
    # For our problem 4x4x8, we split B into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left, C_right = A * B_right.
    # Each has rank at most 49, total rank 98.
    
    # But we can potentially share structure. Let's first build the rank-98 solution.
    
    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4x4 matrix multiplication."""
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        col = 0
        # For each of the 7 Strassen terms, expand into 7 2x2x2 terms
        for mi in range(7):
            # Determine which A blocks and B blocks are involved
            if mi == 0:  # M1 = (A11+A22)*(B11+B22)
                a_blocks = [(0, 1), (3, 1)]  # A11 coeff 1, A22 coeff 1
                b_blocks = [(0, 1), (3, 1)]  # B11 coeff 1, B22 coeff 1
            elif mi == 1:  # M2 = (A21+A22)*B11
                a_blocks = [(2, 1), (3, 1)]
                b_blocks = [(0, 1)]
            elif mi == 2:  # M3 = A11*(B12-B22)
                a_blocks = [(0, 1)]
                b_blocks = [(1, 1), (3, -1)]
            elif mi == 3:  # M4 = A22*(B21-B11)
                a_blocks = [(3, 1)]
                b_blocks = [(2, 1), (0, -1)]
            elif mi == 4:  # M5 = (A11+A12)*B22
                a_blocks = [(0, 1), (1, 1)]
                b_blocks = [(3, 1)]
            elif mi == 5:  # M6 = (A21-A11)*(B11+B12)
                a_blocks = [(2, 1), (0, -1)]
                b_blocks = [(0, 1), (1, 1)]
            elif mi == 6:  # M7 = (A12-A22)*(B21+B22)
                a_blocks = [(1, 1), (3, -1)]
                b_blocks = [(2, 1), (3, 1)]
            
            # For this Mi, it's a 2x2 block matrix multiplication.
            # The local 2x2 A matrix has entries from a_blocks, and local 2x2 B from b_blocks.
            # We need to apply 2x2x2 Strassen to this.
            
            # Map local 2x2 indices to global indices
            # Local A: [[a00, a01], [a10, a11]]
            # a00 comes from a_blocks[0]'s (0,0) element
            # etc.
            
            # Build local index mappings
            local_A_map = {}
            local_B_map = {}
            local_C_map = {}
            
            # For local 2x2 matrices, the block positions are 0,1,2,3 (11,12,21,22)
            # But actually, each Mi involves specific blocks. Let me rethink.
            
            # Actually, each Mi is a product of two 2x2 matrices (composed of 2x2 blocks).
            # The 2x2x2 Strassen applies to the 2x2 matrices, not the blocks.
            
            # Let me take a different approach. For each Mi, I have:
            # Mi = (sum of c_a * A_block) * (sum of c_b * B_block)
            # where each A_block and B_block is a 2x2 matrix.
            # The product is a 2x2 matrix.
            # I can expand this using 2x2x2 Strassen.
            
            # For simplicity, let me just use the schoolbook for each 2x2x2 sub-multiplication.
            # This gives rank 4*7 = 28 for each Mi, total 7*28 = 196. That's worse than schoolbook.
            
            # Instead, let me use Strassen for each 2x2x2, giving rank 7 per Mi, total 49.
            
            # For each Mi, I need to express it as a sum of 7 rank-1 terms.
            # Each rank-1 term involves linear combos of the local 2x2 A and B entries.
            
            # This is getting complex. Let me just use the rank-98 solution (two 4x4x4 Strassens)
            # and verify it.
            
            pass
        
        return None
    
    # Let me try a simpler approach: just use two 4x4x4 Strassen decompositions.
    # For C_left = A * B_left and C_right = A * B_right.
    
    # Actually, let me try to directly construct a better decomposition.
    # Key insight: for 4x4x8, we can view it as multiplying a 4x4 matrix by an 8-column matrix.
    # We can use the fact that 4x4x4 has rank at most 49 (Strassen).
    # So 4x4x8 has rank at most 98.
    
    # Let me build the rank-98 solution and verify it.
    
    def build_strassen_4x4x4():
        """Build a rank-49 decomposition for 4x4x4."""
        n = 4
        rank = 49
        U = [[0] * rank for _ in range(n*n)]
        V = [[0] * rank for _ in range(n*n)]
        W = [[0] * rank for _ in range(n*n)]
        
        col = 0
        # 7 Strassen terms, each expanded into 7 2x2x2 terms
        for mi in range(7):
            # Determine A block coefficients and B block coefficients
            if mi == 0:
                a_coeffs = {0: 1, 3: 1}  # A11, A22
                b_coeffs = {0: 1, 3: 1}  # B11, B22
            elif mi == 1:
                a_coeffs = {2: 1, 3: 1}  # A21, A22
                b_coeffs = {0: 1}  # B11
            elif mi == 2:
                a_coeffs = {0: 1}  # A11
                b_coeffs = {1: 1, 3: -1}  # B12, B22
            elif mi == 3:
                a_coeffs = {3: 1}  # A22
                b_coeffs = {2: 1, 0: -1}  # B21, B11
            elif mi == 4:
                a_coeffs = {0: 1, 1: 1}  # A11, A12
                b_coeffs = {3: 1}  # B22
            elif mi == 5:
                a_coeffs = {2: 1, 0: -1}  # A21, A11
                b_coeffs = {0: 1, 1: 1}  # B11, B12
            elif mi == 6:
                a_coeffs = {1: 1, 3: -1}  # A12, A22
                b_coeffs = {2: 1, 3: 1}  # B21, B22
            
            # For each of the 7 2x2x2 Strassen terms
            for si in range(7):
                # 2x2x2 Strassen term si
                # This multiplies local A (2x2) and local B (2x2)
                # The local A is formed by combining the blocks in a_coeffs
                # Similarly for B.
                
                # For the 2x2x2 Strassen term, we need:
                # u_a: linear combo of local A entries (2x2 = 4 entries)
                # u_b: linear combo of local B entries
                # u_c: linear combo of local C entries
                
                # The local 2x2 A matrix has entries:
                # Local A[0,0] = sum of a_coeffs[block] * A_block[0,0]
                # etc.
                
                # This is getting very complex. Let me use a different strategy.
                
                pass
        
        return U, V, W, rank
    
    # Given the complexity, let me try a simpler optimization: 
    # Use the schoolbook decomposition but try to merge terms.
    # Or try a direct search for a lower-rank decomposition.
    
    # For now, let me just return the schoolbook solution.
    # The task says score is the rank, so lower is better.
    # Schoolbook gives rank 128.
    
    # Let me try to implement a rank-98 solution using two 4x4x4 Strassens.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank-98: two 4x4x4 Strassen decompositions
    # I'll implement a helper to build a 4x4x4 Strassen decomposition.
    
    def build_4x4x4_strassen_decomposition():
        """
        Build rank-49 decomposition for 4x4x4 matrix multiplication.
        Uses Strassen's algorithm on 2x2 blocks.
        Returns U, V, W matrices.
        """
        n = 4
        rank = 49
        U = [[0] * rank for _ in range(n*n)]
        V = [[0] * rank for _ in range(n*n)]
        W = [[0] * rank for _ in range(n*n)]
        
        # 2x2x2 Strassen terms
        # For 2x2x2: A entries (i,j) -> i*2+j, B entries (j,k) -> j*2+k, W entries k*2+i
        strassen_terms = [
            # (U_dict, V_dict, W_dict)
            ({0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}, {0:1, 3:1}),  # M1
            ({2:1, 3:1}, {0:1}, {1:1, 3:-1}),  # M2
            ({0:1}, {1:1, 3:-1}, {2:1, 3:1}),  # M3
            ({3:1}, {2:1, 0:-1}, {1:1, 0:1}),  # M4
            ({0:1, 1:1}, {3:1}, {2:1, 0:-1}),  # M5
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),  # M6
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),  # M7
        ]
        
        # 7 outer Strassen terms for 4x4x4
        outer_terms = [
            # (a_block_coeffs, b_block_coeffs)
            # block indices: 0=11, 1=12, 2=21, 3=22
            ({0:1, 3:1}, {0:1, 3:1}),  # M1 = (A11+A22)*(B11+B22)
            ({2:1, 3:1}, {0:1}),  # M2 = (A21+A22)*B11
            ({0:1}, {1:1, 3:-1}),  # M3 = A11*(B12-B22)
            ({3:1}, {2:1, 0:-1}),  # M4 = A22*(B21-B11)
            ({0:1, 1:1}, {3:1}),  # M5 = (A11+A12)*B22
            ({2:1, 0:-1}, {0:1, 1:1}),  # M6 = (A21-A11)*(B11+B12)
            ({1:1, 3:-1}, {2:1, 3:1}),  # M7 = (A12-A22)*(B21+B22)
        ]
        
        col = 0
        for oi, (a_coeff, b_coeff) in enumerate(outer_terms):
            for si, (u_2x2, v_2x2, w_2x2) in enumerate(strassen_terms):
                # For this combined term, we need to map local 2x2 indices to global indices.
                # Local 2x2 A: entries (li, lj) where li, lj in {0,1}
                # Global A index for block b, local (li, lj): 
                #   block b has row_start = (b//2)*2, col_start = (b%2)*2
                #   global A index = (row_start+li)*4 + (col_start+lj)
                
                # For local 2x2 A entry (li, lj), it's a sum over blocks:
                # sum_b a_coeff[b] * A_global[(row_start(b)+li)*4 + (col_start(b)+lj)]
                
                # Similarly for B.
                
                # For the U matrix, we add to U[global_A_idx, col] the coefficient.
                for li in range(2):
                    for lj in range(2):
                        local_a_idx = li*2 + lj
                        if local_a_idx in u_2x2:
                            u_val = u_2x2[local_a_idx]
                            for b, coeff in a_coeff.items():
                                row_start = (b // 2) * 2
                                col_start = (b % 2) * 2
                                global_a = (row_start + li) * 4 + (col_start + lj)
                                U[global_a][col] += u_val * coeff
                
                for lj in range(2):  # local B row (j)
                    for lk in range(2):  # local B col (k)
                        local_b_idx = lj*2 + lk
                        if local_b_idx in v_2x2:
                            v_val = v_2x2[local_b_idx]
                            for b, coeff in b_coeff.items():
                                row_start = (b // 2) * 2
                                col_start = (b % 2) * 2
                                global_b = (row_start + lj) * 4 + (col_start + lk)
                                V[global_b][col] += v_val * coeff
                
                for li in range(2):  # local C row (i)
                    for lk in range(2):  # local C col (k)
                        local_c_idx = lk*2 + li  # W convention: k*2+i
                        if local_c_idx in w_2x2:
                            w_val = w_2x2[local_c_idx]
                            # C block: C11 (block 0), C12 (block 1), C21 (block 2), C22 (block 3)
                            # Which C block does this contribute to?
                            # The outer Strassen term determines which C blocks are affected.
                            # For Mi, the result is added to specific C blocks.
                            # Let me determine this.
                            
                            # M1 = (A11+A22)*(B11+B22) -> contributes to C11 and C22
                            # M2 = (A21+A22)*B11 -> contributes to C21 and C22
                            # M3 = A11*(B12-B22) -> contributes to C12 and C22
                            # M4 = A22*(B21-B11) -> contributes to C21 and C11
                            # M5 = (A11+A12)*B22 -> contributes to C12 and C11
                            # M6 = (A21-A11)*(B11+B12) -> contributes to C22
                            # M7 = (A12-A22)*(B21+B22) -> contributes to C11
                            
                            if oi == 0:  # M1
                                c_blocks = {0: 1, 3: 1}  # C11 +1, C22 +1
                            elif oi == 1:  # M2
                                c_blocks = {2: 1, 3: -1}  # C21 +1, C22 -1
                            elif oi == 2:  # M3
                                c_blocks = {1: 1, 3: 1}  # C12 +1, C22 +1
                            elif oi == 3:  # M4
                                c_blocks = {2: 1, 0: 1}  # C21 +1, C11 +1
                            elif oi == 4:  # M5
                                c_blocks = {1: 1, 0: -1}  # C12 +1, C11 -1
                            elif oi == 5:  # M6
                                c_blocks = {3: 1}  # C22 +1
                            elif oi == 6:  # M7
                                c_blocks = {0: 1}  # C11 +1
                            
                            for cb, ccoeff in c_blocks.items():
                                row_start = (cb // 2) * 2
                                col_start = (cb % 2) * 2
                                global_i = row_start + li
                                global_k = col_start + lk
                                global_w = global_k * 4 + global_i
                                W[global_w][col] += w_val * ccoeff
                
                col += 1
        
        return U, V, W, rank
    
    try:
        U4, V4, W4, rank4 = build_4x4x4_strassen_decomposition()
        # Verify: check that U * V * W reconstructs the 4x4x4 multiplication tensor
        # For random A, B, check A*B == sum_r U[:,r] @ V[:,r] @ W[:,r]
        
        import random
        random.seed(42)
        valid = True
        for trial in range(5):
            A = np.array([[random.randint(-5,5) for _ in range(4)] for _ in range(4)])
            B = np.array([[random.randint(-5,5) for _ in range(4)] for _ in range(4)])
            C_expected = A @ B
            
            C_computed = np.zeros((4,4))
            U_np = np.array(U4).T  # (rank, 16)
            V_np = np.array(V4).T  # (rank, 16)
            W_np = np.array(W4).T  # (rank, 16)
            
            for r in range(rank4):
                A_vec = U_np[r, :]  # (16,)
                B_vec = V_np[r, :]  # (16,)
                W_vec = W_np[r, :]  # (16,)
                
                # Reshape A_vec to 4x4, B_vec to 4x4
                A_mat = A_vec.reshape(4,4)
                B_mat = B_vec.reshape(4,4)
                # Product is a scalar: sum of A_vec[i*4+j] * B_vec[j*4+k] for all i,j,k
                # Actually, the rank-1 term is: sum_{i,j,k} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                # This equals (sum_{i,j} U[i*4+j, r] * A[i,j]) * (sum_{j,k} V[j*4+k, r] * B[j,k])
                # times the W contribution... no.
                
                # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For 4x4x4: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                # And T[i,j,k] should equal A[i,j] * B[j,k] for the matrix multiplication.
                
                # So C[i,k] = sum_j A[i,j] * B[j,k]
                # = sum_r sum_j U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                
                # Let me compute this directly.
                pass
        
        # Let me verify differently
        C_computed = np.zeros((4,4))
        for r in range(rank4):
            u_col = [U4[i*4+j][r] for i in range(4) for j in range(4)]
            v_col = [V4[j*4+k][r] for j in range(4) for k in range(4)]
            w_col = [W4[k*4+i][r] for i in range(4) for k in range(4)]
            
            # For each (i,k), C[i,k] += sum_j u[i*4+j] * v[j*4+k] * w[k*4+i]
            for i in range(4):
                for k in range(4):
                    s = 0
                    for j in range(4):
                        s += u_col[i*4+j] * v_col[j*4+k] * w_col[k*4+i]
                    C_computed[i,k] += s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            valid = False
        
        if valid:
            # Build 4x4x8 solution: two 4x4x4 Strassens
            rank_total = 2 * rank4
            U_full = [[0] * rank_total for _ in range(16)]
            V_full = [[0] * rank_total for _ in range(32)]
            W_full = [[0] * rank_total for _ in range(32)]
            
            # Left half: B columns 0-3
            for i in range(16):
                for r in range(rank4):
                    U_full[i][r] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    for r in range(rank4):
                        V_full[j*8+k][r] = V4[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    for r in range(rank4):
                        W_full[k*4+i][r] = W4[k*4+i][r]
            
            # Right half: B columns 4-7
            for i in range(16):
                for r in range(rank4):
                    U_full[i][rank4+r] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    for r in range(rank4):
                        V_full[j*8+(k+4)][rank4+r] = V4[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    for r in range(rank4):
                        W_full[(k+4)*4+i][rank4+r] = W4[k*4+i][r]
            
            if rank_total < best_rank:
                best_rank = rank_total
                best_U = U_full
                best_V = V_full
                best_W = W_full
        
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
