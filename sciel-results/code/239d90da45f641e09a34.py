import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Part 1: M_{1,4,8} (i=0)
    # Part 2: M_{2,4,8} (i=1,2) split into M_{2,2,8} (j=0,1) and M_{2,2,8} (j=2,3)
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7
    # Schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    # col is now 32
    
    # Part 2: i=1,2
    # Sub-part 2a: j=0,1
    # Sub-part 2b: j=2,3
    
    # We need a helper to add M_{2,2,8} decomposition
    # M_{2,2,8} splits into two M_{2,2,4}
    # M_{2,2,4} splits into two M_{2,2,2}
    # M_{2,2,2} is Strassen rank 7
    
    def add_strassen_block(i_base, j_base, k_base, col_offset):
        """
        Adds Strassen decomposition for M_{2,2,2} block.
        i_base: start index for i (0 or 1)
        j_base: start index for j (0, 1, 2, or 3)
        k_base: start index for k (0..7)
        col_offset: current column index
        Returns new col_offset
        """
        # Strassen coefficients
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W
        # U indices: (i_base + di, j_base + dj)
        # V indices: (j_base + dj, k_base + dk)
        # W indices: (k_base + dk, i_base + di)
        
        # M1
        # A11+A22 -> U[i_base, j_base]=1, U[i_base+1, j_base+1]=1
        # B11+B22 -> V[j_base, k_base]=1, V[j_base+1, k_base+1]=1
        # W11+W22? No, W is output.
        # The term is (A11+A22) * (B11+B22).
        # This expands to 4 terms in schoolbook, but Strassen treats it as one product.
        # Wait, the rank-1 tensor for M1 is:
        # U has 1s at (i_base, j_base) and (i_base+1, j_base+1)
        # V has 1s at (j_base, k_base) and (j_base+1, k_base+1)
        # W has 1s at (k_base, i_base) and (k_base+1, i_base+1)
        # This is NOT a rank-1 tensor in the standard sense if we sum them.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # Here $u$ has two non-zeros. So it's not rank 1.
        # Strassen uses 7 products of *sums*.
        # Each product is a scalar $M_x$.
        # The decomposition is $C = \sum_{r=1}^7 M_r \otimes W_r$?
        # No, the tensor decomposition is:
        # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$
        # where $u_r, v_r, w_r$ are vectors.
        # For M1:
        # $u_1$ has 1 at (0,0) and (1,1).
        # $v_1$ has 1 at (0,0) and (1,1).
        # $w_1$ has 1 at (0,0) and (1,1).
        # This is a valid term in the decomposition.
        
        # Let's define the 7 terms for M_{2,2,2}
        
        # Term 1: M1
        U[i_base + 0 * m + j_base][col_offset] = 1
        U[i_base + 1 * m + j_base + 1][col_offset] = 1
        V[j_base * p + k_base][col_offset] = 1
        V[(j_base + 1) * p + (k_base + 1)][col_offset] = 1
        W[k_base * n + i_base][col_offset] = 1
        W[(k_base + 1) * n + (i_base + 1)][col_offset] = 1
        col_offset += 1
        
        # Term 2: M2 = (A21+A22)B11
        U[i_base + 1 * m + j_base][col_offset] = 1
        U[i_base + 1 * m + j_base + 1][col_offset] = 1
        V[j_base * p + k_base][col_offset] = 1
        W[k_base * n + i_base + 1][col_offset] = 1
        col_offset += 1
        
        # Term 3: M3 = A11(B21-B22)
        U[i_base + 0 * m + j_base][col_offset] = 1
        V[j_base + 1 * p + k_base][col_offset] = 1
        V[j_base + 1 * p + k_base + 1][col_offset] = -1
        W[k_base * n + i_base][col_offset] = 1
        W[(k_base + 1) * n + i_base][col_offset] = -1
        col_offset += 1
        
        # Term 4: M4 = A22(B12-B11)
        U[i_base + 1 * m + j_base + 1][col_offset] = 1
        V[j_base * p + k_base + 1][col_offset] = 1
        V[j_base * p + k_base][col_offset] = -1
        W[(k_base + 1) * n + i_base + 1][col_offset] = 1
        W[k_base * n + i_base + 1][col_offset] = -1
        col_offset += 1
        
        # Term 5: M5 = (A11+A12)B22
        U[i_base + 0 * m + j_base][col_offset] = 1
        U[i_base + 0 * m + j_base + 1][col_offset] = 1
        V[j_base + 1 * p + k_base + 1][col_offset] = 1
        W[(k_base + 1) * n + i_base][col_offset] = 1
        col_offset += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        U[i_base + 1 * m + j_base][col_offset] = 1
        U[i_base + 0 * m + j_base][col_offset] = -1
        V[j_base * p + k_base][col_offset] = 1
        V[j_base * p + k_base + 1][col_offset] = 1
        W[k_base * n + i_base + 1][col_offset] = 1
        W[(k_base + 1) * n + i_base + 1][col_offset] = 1
        col_offset += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        U[i_base + 0 * m + j_base + 1][col_offset] = 1
        U[i_base + 1 * m + j_base + 1][col_offset] = -1
        V[j_base + 1 * p + k_base][col_offset] = 1
        V[j_base + 1 * p + k_base + 1][col_offset] = 1
        W[k_base * n + i_base][col_offset] = 1
        W[(k_base + 1) * n + i_base][col_offset] = 1
        col_offset += 1
        
        return col_offset

    # Now we need to assemble the M_{2,2,8} blocks
    # M_{2,2,8} at (i_base, j_base)
    # Splits into two M_{2,2,4} at k_base and k_base+4
    # Each M_{2,2,4} splits into two M_{2,2,2} at k_base and k_base+2
    
    def add_m228(i_base, j_base, k_base, col_offset):
        # First M_{2,2,4}
        # First M_{2,2,2}
        col_offset = add_strassen_block(i_base, j_base, k_base, col_offset)
        # Second M_{2,2,2}
        col_offset = add_strassen_block(i_base, j_base, k_base + 2, col_offset)
        # Second M_{2,2,4}
        # First M_{2,2,2}
        col_offset = add_strassen_block(i_base, j_base, k_base + 4, col_offset)
        # Second M_{2,2,2}
        col_offset = add_strassen_block(i_base, j_base, k_base + 6, col_offset)
        return col_offset

    # Part 2a: i_base=1, j_base=0, k_base=0
    col = add_m228(1, 0, 0, col)
    
    # Part 2b: i_base=1, j_base=2, k_base=0
    col = add_m228(1, 2, 0, col)
    
    # Check if we used all columns
    # 32 + 28 + 28 = 88
    # col should be 88
    
    return {"U": U, "V": V, "W": W}

