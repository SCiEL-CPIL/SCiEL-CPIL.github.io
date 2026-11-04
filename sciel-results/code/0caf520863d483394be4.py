"""Block-wise Strassen decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Strassen's algorithm on 3x3 blocks.
    
    For 6x6 matrices, we split into 2x2 blocks of size 3x3.
    Strassen uses 7 multiplications of 3x3 blocks instead of 8.
    Each 3x3 block multiplication uses the schoolbook 27 scalar multiplications.
    
    Total rank = 7 * 27 = 189 (vs 216 for pure schoolbook)
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 3  # 6 / 2 blocks per dimension
    
    # The 7 Strassen products for 2x2 block matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each of the 7 Strassen products, we do a 3x3 schoolbook multiplication
    # Each term: (sum of A blocks) * (sum of B blocks)
    
    # Define the 7 Strassen factors as (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    # Block indices: (row_block, col_block) where row_block, col_block in {0, 1}
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10 + A11) * B00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3 = A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4 = A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5 = (A00 + A01) * B11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # C contributions: C_block = sum of M terms with coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contributions = [
        [(0,1), (3,1), (4,-1), (6,1)],   # C00
        [(2,1), (4,1)],                    # C01
        [(1,1), (3,1)],                    # C10
        [(0,1), (1,-1), (2,1), (5,1)],    # C11
    ]
    
    # For each Strassen product, expand the 3x3 schoolbook multiplication
    for prod_idx in range(7):
        a_factors, b_factors = strassen_products[prod_idx]
        
        # The product is (sum_{ai} ai * A_block_ai) * (sum_{bi} bi * B_block_bi)
        # = sum_{ai,bi} ai*bi * A_block_ai * B_block_bi
        # Each A_block_ai * B_block_bi is a 3x3 * 3x3 multiplication
        
        for a_br, a_bc, a_coeff in a_factors:
            for b_br, b_bc, b_coeff in b_factors:
                total_coeff = a_coeff * b_coeff
                if total_coeff == 0:
                    continue
                
                # For the 3x3 schoolbook multiplication of block (a_br,a_bc) * (b_br,b_bc):
                # Result block contributes to C blocks according to c_contributions[prod_idx]
                # The (i,j) entry of the product block (row i, col j) is:
                # sum_s A_block[i,s] * B_block[s,j]
                
                # Global indices:
                # A row: a_br * 3 + i, A col: a_bc * 3 + s
                # B row: b_br * 3 + s, B col: b_bc * 3 + j
                # C row: depends on which C block, C col: depends on which C block
                
                for i in range(3):
                    for s in range(3):
                        for j in range(3):
                            # U[i*m + j, col] where i is A row, j is A col (which is B row)
                            # Actually: U[A_row * m + A_col, col]
                            # V[B_row * p + B_col, col]
                            # W[C_col * n + C_row, col]
                            
                            a_row = a_br * 3 + i
                            a_col = a_bc * 3 + s
                            b_row = b_br * 3 + s
                            b_col = b_bc * 3 + j
                            
                            # This scalar term contributes to each C block in c_contributions[prod_idx]
                            for c_block_idx, c_coeff in c_contributions[prod_idx]:
                                c_br, c_bc = c_block_idx
                                c_row = c_br * 3 + i
                                c_col = c_bc * 3 + j
                                
                                # Add contribution: we need to accumulate
                                # But each (a_row,a_col,b_row,b_col) for a given prod_idx 
                                # and (i,s,j) is unique, so no conflict within same col
                                # However, different (a_br,a_bc,b_br,b_bc) pairs may produce
                                # same (a_row,a_col,b_row,b_col) with different coeffs
                                # We handle this by using the accumulated coefficient
                                
                                # Since we're iterating over all factor combinations,
                                # each (a_row, a_col, b_row, b_col) for a fixed (i,s,j) 
                                # and prod_idx gets contributions from multiple factor pairs.
                                # We need to sum them up first.
                                pass
                            
                            # Actually, let me restructure: for each (i,s,j) and prod_idx,
                            # the total coefficient for a specific (a_row,a_col,b_row,b_col) 
                            # is the sum over all factor pairs that produce those global indices.
                            # But different factor pairs give different global indices,
                            # so there's no overlap. Wait, that's not right either.
                            # 
                            # Let me think again. For a fixed prod_idx, i, s, j:
                            # The A factors give different (a_br, a_bc) pairs.
                            # The B factors give different (b_br, b_bc) pairs.
                            # Each combination (a_br,a_bc,b_br,b_bc) gives a unique
                            # (a_row, a_col, b_row, b_col) = (a_br*3+i, a_bc*3+s, b_br*3+s, b_bc*3+j)
                            # These are all distinct for different factor combinations.
                            # So we can just add each one separately.
                            
                            for c_block_idx, c_coeff in c_contributions[prod_idx]:
                                c_br, c_bc = c_block_idx
                                c_row = c_br * 3 + i
                                c_col = c_bc * 3 + j
                                
                                full_coeff = total_coeff * c_coeff
                                if full_coeff == 0:
                                    continue
                                
                                U[a_row * m + a_col][col] = full_coeff
                                V[b_row * p + b_col][col] = full_coeff
                                W[c_col * n + c_row][col] = full_coeff
                            
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
