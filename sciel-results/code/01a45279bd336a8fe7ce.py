import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 88.
    # 58 for 3x5x5 part, 30 for 3x5x2 part.
    
    # U shape: (n*m, rank) -> (15, rank)
    # V shape: (m*p, rank) -> (35, rank)
    # W shape: (p*n, rank) -> (21, rank)
    
    # We will build lists of (u, v, w) triples
    triples = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict {index: val}
        # v_vals: dict {index: val}
        # w_vals: dict {index: val}
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, val in u_vals.items():
            u_vec[idx] = val
        for idx, val in v_vals.items():
            v_vec[idx] = val
        for idx, val in w_vals.items():
            w_vec[idx] = val
            
        triples.append((u_vec, v_vec, w_vec))

    # 1. Strassen 4x4 for A_TL (4x4) * B_TL (4x4)
    # A_TL indices: i in 0..3, j in 0..3. We only use i in 0..2.
    # B_TL indices: j in 0..3, k in 0..3.
    # C_TL indices: k in 0..3, i in 0..3. We only use i in 0..2.
    
    # Generate Strassen 2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to expand these for 4x4.
    # Let's define a function to get the terms for a 2x2 block at offset (row_offset, col_offset)
    # This is getting complex to do manually.
    # Let's use a recursive function or just hardcode the 49 terms for 4x4 Strassen.
    
    # Actually, we can just use the known rank 49 decomposition for 4x4.
    # Or simpler: use the rank 16 schoolbook for 4x4? No, 49 is better.
    # Let's implement the recursion.
    
    def get_strassen_2x2_terms():
        # Returns list of (u_dict, v_dict, w_dict) for 2x2
        # Indices: A00:0, A01:1, A10:2, A11:3
        # B00:0, B01:1, B10:2, B11:3
        # C00:0, C01:1, C10:2, C11:3
        
        terms = []
        
        # M1
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1} # C00, C11
        terms.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1} # C10, C11
        terms.append((u, v, w))
        
        # M3
        u = {0: 1}
        v = {2: 1, 3: -1}
        w = {1: 1, 3: 1} # C01, C11
        terms.append((u, v, w))
        
        # M4
        u = {3: 1}
        v = {1: 1, 0: -1}
        w = {0: 1, 2: 1} # C00, C10
        terms.append((u, v, w))
        
        # M5
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1} # C00, C01
        terms.append((u, v, w))
        
        # M6
        u = {2: 1, 0: -1}
        v = {0: 1, 2: 1}
        w = {3: 1} # C11
        terms.append((u, v, w))
        
        # M7
        u = {1: 1, 3: -1}
        v = {3: 1, 1: 1}
        w = {0: 1} # C00
        terms.append((u, v, w))
        
        return terms

    base_2x2 = get_strassen_2x2_terms()
    
    # Now build 4x4 from 2x2
    # A_4x4 blocks:
    # A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2) in 4x4 grid?
    # Indices in 4x4:
    # A: 0..15. 0:0-3, 1:4-7, 2:8-11, 3:12-15?
    # Let's use row-major index i*4+j.
    # Block (0,0): i in 0..1, j in 0..1. Indices 0,1,4,5.
    # Block (0,1): i in 0..1, j in 2..3. Indices 2,3,6,7.
    # Block (1,0): i in 2..3, j in 0..1. Indices 8,9,12,13.
    # Block (1,1): i in 2..3, j in 2..3. Indices 10,11,14,15.
    
    def map_2x2_to_4x4(u_dict, v_dict, w_dict, a_block, b_block, c_block):
        # a_block: (row_offset, col_offset) for A
        # b_block: (row_offset, col_offset) for B
        # c_block: (row_offset, col_offset) for C
        
        u_new = {}
        for idx, val in u_dict.items():
            i, j = divmod(idx, 2)
            new_i = i + a_block[0]
            new_j = j + a_block[1]
            new_idx = new_i * 4 + new_j
            u_new[new_idx] = u_new.get(new_idx, 0) + val
            
        v_new = {}
        for idx, val in v_dict.items():
            i, j = divmod(idx, 2)
            new_i = i + b_block[0]
            new_j = j + b_block[1]
            new_idx = new_i * 4 + new_j
            v_new[new_idx] = v_new.get(new_idx, 0) + val
            
        w_new = {}
        for idx, val in w_dict.items():
            i, j = divmod(idx, 2)
            new_i = i + c_block[0]
            new_j = j + c_block[1]
            new_idx = new_i * 4 + new_j
            w_new[new_idx] = w_new.get(new_idx, 0) + val
            
        return u_new, v_new, w_new

    # Strassen 4x4 terms
    strassen_4x4_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is (0,0), A11 is (1,1)
    # B00 is (0,0), B11 is (1,1)
    # C00 is (0,0), C11 is (1,1)
    
    # We need to sum the terms for A00 and A11, etc.
    # This is easier to just iterate over the 7 base terms and apply the block mappings.
    
    # Actually, the 7 terms of Strassen 2x2 operate on the blocks.
    # Term 1: (A00+A11)(B00+B11)
    # This is not a single bilinear term, it's a sum of 4 products.
    # Strassen replaces 8 products with 7.
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Each of these is a product of two 2x2 matrices (or sums/diffs).
    # To get the scalar terms, we need to expand these using the base_2x2 terms?
    # No, the base_2x2 terms ARE the scalar multiplications.
    # So for M1, we need to compute (A00+A11)(B00+B11).
    # This is equivalent to taking the sum of the scalar terms of A00 and A11, and B00 and B11, and multiplying?
    # No, it's (A00+B00) ... wait.
    # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # Their product is computed using the 7 scalar multiplications of Strassen 2x2.
    # So for each of the 7 high-level terms (M1..M7), we need to generate 7 low-level scalar terms.
    # Total 49 terms.
    
    # Let's define the high-level terms' A and B components.
    # High-level term 1: A_comp = A00+A11, B_comp = B00+B11
    # We need to map A_comp to a 2x2 matrix of coefficients.
    
    # This is getting too complex to do generically in the code without errors.
    # Given the time limit, I will fall back to a simpler, slightly higher rank decomposition if 88 is too hard to code correctly.
    # Schoolbook for 3x5x7 is 105.
    # Schoolbook for 3x5x5 (75) + 3x5x2 (30) = 105.
    # Strassen for 4x4 (49) + 4 (4x1x1x4) + 4 (4x4x4x1) + 1 (4x1x1x1) = 58.
    # 58 + 30 = 88.
    
    # Let's try to implement the 49 terms of Strassen 4x4 by explicitly listing them.
    # Or use a library? No.
    
    # Alternative: Use the rank 16 decomposition for 4x4? No, 49 is standard.
    # Let's just use Schoolbook for the 4x4 part to be safe, then optimize if time permits.
    # Schoolbook 4x4 is 16.
    # 16 + 4 + 4 + 1 = 25.
    # 25 + 30 = 55.
    # Wait, Schoolbook 4x4 is 16. Strassen is 49.
    # 16 < 49.
    # So Schoolbook 4x4 is BETTER for 4x4?
    # No, Strassen is for large matrices. For 4x4, 16 multiplications is better than 49.
    # Yes, $4^3 = 64$ schoolbook? No, $4 \times 4$ matrix multiplication has 64 scalar multiplications.
    # Strassen has 49.
    # So 49 < 64.
    # My previous "16" was wrong. 4x4 schoolbook is 64.
    # So 49 is correct.
    
    # So 88 is the target.
    
    # I will implement the 49 terms by generating them from the 2x2 Strassen terms.
    
    high_level_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_idx, coeff)
        # Block indices: 0:00, 1:01, 2:10, 3:11
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]), # M2
        ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (1, 1), (3, 1) ]), # M3
        ([ (3, 1) ], [ (1, 1), (0, -1) ], [ (0, 1), (2, 1) ]), # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]), # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (3, 1) ]), # M6
        ([ (1, 1), (3, -1) ], [ (3, 1), (1, 1) ], [ (0, 1) ])  # M7
    ]
    
    # For each high-level term, we need to compute the product of the combined A and B blocks.
    # This is done by applying the 2x2 Strassen decomposition to the combined blocks.
    # This is effectively what the recursion does.
    
    # I will skip the full 4x4 Strassen implementation to avoid bugs and use a known valid decomposition.
    # I will use the Schoolbook for 3x5x7 (105) as a fallback, but try to implement 88.
    
    # Given the complexity, I will implement the 88 rank solution carefully.
    
    # 1. 3x5x5 part (58 terms)
    # 2. 3x5x2 part (30 terms)
    
    # I will write the code to generate these.
    
    # ... (code generation logic) ...
    
    # For the sake of the response, I will provide the code that constructs the 88 rank decomposition.
    
    pass

# The code above is a sketch. I will write the full code now.
