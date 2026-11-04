import numpy as np

def strassen_factors(n):
    """
    Generate U, V, W factors for n x n matrix multiplication using Strassen's algorithm.
    n must be a power of 2.
    Returns U, V, W as lists of lists.
    """
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    # Recursive call for n/2
    U_sub, V_sub, W_sub = strassen_factors(n // 2)
    
    # Dimensions
    half = n // 2
    rank_sub = len(U_sub[0])
    
    # Strassen has 7 multiplications at each level
    rank = 7 * rank_sub
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Define the 7 combinations
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Helper to add a block of U, V, W
    def add_block(U_block, V_block, W_block, base_col):
        for i in range(n * n):
            for j in range(rank_sub):
                U[i][base_col + j] += U_block[i]
                V[i][base_col + j] += V_block[i]
                W[i][base_col + j] += W_block[i]
                
    # This is tricky because U, V, W for the sub-problems are for half x half matrices.
    # We need to map the indices.
    
    # Let's build the factors explicitly for the 7 terms.
    # Each term involves a specific combination of A and B blocks.
    
    # Term 1: (A11+A22)(B11+B22)
    # U part: A11 + A22
    # V part: B11 + B22
    # W part: C11 + C12 + C21 + C22 ? No.
    # The result of the multiplication is added to C.
    # M1 contributes to C11, C12, C21, C22?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes +1 to C11 and +1 to C22.
    # M2 contributes +1 to C21 and -1 to C22.
    # M3 contributes +1 to C12 and +1 to C22.
    # M4 contributes +1 to C11 and +1 to C21.
    # M5 contributes -1 to C11 and +1 to C12.
    # M6 contributes +1 to C22.
    # M7 contributes +1 to C11.
    
    # We need to construct U, V, W for the 7 terms.
    # For each term t (0..6), we have a sub-decomposition of rank rank_sub.
    # The U, V, W for term t are scaled and placed into the global U, V, W.
    
    # Let's define the blocks for A and B.
    # A11: rows 0..half-1, cols 0..half-1
    # A12: rows 0..half-1, cols half..n-1
    # A21: rows half..n-1, cols 0..half-1
    # A22: rows half..n-1, cols half..n-1
    
    # Same for B.
    
    # For each term, we define the linear combination of A blocks and B blocks.
    # Then we use the sub-factors U_sub, V_sub, W_sub.
    
    # Term 0: M1
    # U: A11 + A22
    # V: B11 + B22
    # W: C11 + C22
    # We add rank_sub columns.
    base = 0
    for i in range(half * half):
        r_a = i // half
        c_a = i % half
        # A11 index: r_a * half + c_a
        # A22 index: (half + r_a) * half + (half + c_a)
        U[r_a * n + c_a][base + i] = U_sub[i][0] # Wait, U_sub is for half x half.
        # This logic is getting complicated to map indices correctly.
        
    # Alternative: Just generate the factors for 2x2 Strassen and recurse.
    # But we need to handle the index mapping.
    
    # Let's try a different approach:
    # Generate the factors for 2x2 Strassen explicitly.
    # Then use a recursive function that takes the factors for n/2 and produces factors for n.
    
    def combine_factors(U_sub, V_sub, W_sub, half):
        n = 2 * half
        rank_sub = len(U_sub[0])
        rank = 7 * rank_sub
        U = [[0] * rank for _ in range(n * n)]
        V = [[0] * rank for _ in range(n * n)]
        W = [[0] * rank for _ in range(n * n)]
        
        # Helper to add a block
        def add_block(u_block, v_block, w_block, base_col):
            for i in range(half * half):
                for j in range(rank_sub):
                    U[i][base_col + j] += u_block[i]
                    V[i][base_col + j] += v_block[i]
                    W[i][base_col + j] += w_block[i]
                    
        # We need to map the sub-factors to the 4 blocks.
        # For a term like (A11+A22)(B11+B22), the U part is A11+A22.
        # This means for each row in A11, we add the corresponding U_sub row.
        # For each row in A22, we add the corresponding U_sub row.
        
        # Let's define the 7 terms.
        # Term 0: M1 = (A11+A22)(B11+B22)
        # U: A11 + A22
        # V: B11 + B22
        # W: C11 + C22
        
        # We iterate over the 7 terms.
        for t in range(7):
            base_col = t * rank_sub
            # Determine U, V, W blocks for this term
            # This is complex to generalize.
            
        # Given the time, it's safer to just use the schoolbook for 5x5 but with a small optimization?
        # No, 74 is a good target.
        # Let's hardcode the 2x2 Strassen factors and use a recursive builder.
        
        return U, V, W
        
    # This is too complex to implement correctly in the time remaining.
    # Let's fall back to a simpler improvement:
    # Use Strassen for 2x2 blocks within the 5x5 matrix.
    # But 5 is not a power of 2.
    
    # Alternative: Use the rank-49 decomposition for 4x4 and add the 25 extra terms.
    # I will implement the 4x4 Strassen factors by hardcoding the 2x2 Strassen and recursing.
    
    return U, V, W

def construct():
    n, m, p = 5, 5, 5
    # Start with schoolbook for 5x5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will replace the schoolbook with Strassen-based decomposition.
    # For now, return schoolbook to ensure correctness.
    # TODO: Implement Strassen 4x4 + extra terms.
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
