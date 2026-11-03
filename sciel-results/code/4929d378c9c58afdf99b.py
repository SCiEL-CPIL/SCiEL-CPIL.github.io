"""Matrix multiplication 2x4x6 rank minimization via WFTA / tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x6 matrix.
    
    Strategy: Use the Winograd Fourier Transform Algorithm (WFTA) exploiting the
    cyclic group structure. For 2x4x6, we can use a combination of:
    - 2x2 blocks with Strassen (rank 7 each)
    - Or use a direct WFTA-based decomposition
    
    The WFTA approach for matrix multiplication A(n x m) * B(m x p):
    We decompose into smaller blocks and use fast transform methods.
    
    For 2x4x6:
    - Split into 2x2 blocks: A is 1x2 blocks, B is 2x3 blocks, C is 1x3 blocks
    - Each 2x2x2 sub-problem can use Strassen (rank 7)
    - But we need to handle the 4 and 6 dimensions carefully
    
    Alternative: Direct WFTA using Z_2 x Z_2 x Z_3 structure
    
    Let's use a block decomposition:
    A (2x4) = [A11 A12] where A11, A12 are 2x2
    B (4x6) = [B11; B12] where B11, B12 are 2x6
    
    C = A11*B11 + A12*B12
    
    Each A*B is 2x2 times 2x6 = 2x6 result
    We can further decompose the 2x6 into 2x2 and 2x4 blocks.
    
    Actually, let's use a more systematic WFTA approach.
    
    For the 2x4x6 problem, one known approach:
    - Use 2x2x2 Strassen for the 2x2 blocks
    - Handle the 4 and 6 dimensions with appropriate transforms
    
    Let me construct a decomposition using the following strategy:
    Split A (2x4) into 2 blocks of 2x2: A = [A0 | A1]
    Split B (4x6) into 2 blocks of 2x6: B = [B0; B1]
    Split C (2x6) into 3 blocks of 2x2: C = [C0 | C1 | C2]
    
    C = A0*B0 + A1*B1
    
    Now A0*B0 is 2x2 * 2x6 = 2x6
    A1*B1 is 2x2 * 2x6 = 2x6
    
    For 2x2 * 2x6:
    Split B into 3 blocks of 2x2: B = [B0 | B1 | B2]
    Result = [R0 | R1 | R2] where Ri = A*B_i
    
    Each A*B_i is 2x2 * 2x2, which can be done with Strassen (rank 7)
    
    Total: 2 (for A0, A1) * 3 (for B0, B1, B2) * 7 (Strassen) = 42
    
    But we can do better by sharing Strassen intermediates.
    
    Let me use a direct WFTA construction. The key insight is that for
    matrix multiplication over Z_n x Z_m x Z_p, we can use the discrete
    Fourier transform to diagonalize the tensor.
    
    For 2x4x6, the group is Z_2 x Z_4 x Z_6, but we can use a different
    decomposition.
    
    Let me try a concrete construction with rank 40 or lower.
    
    Actually, let's use the known result: the rank of 2x4x6 matrix multiplication
    is at most 40 (and possibly lower).
    
    Here's a construction using WFTA:
    
    We view the multiplication as a tensor T in R^{2x4} x R^{4x6} x R^{2x6}
    T[i,j,k,l,m,n] = delta(j,k) * delta(l,m) ... no, let me be more careful.
    
    The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p):
    T_{(i,j), (j',k), (i',k')} = delta(i,i') * delta(j,j') * delta(k,k')
    
    For n=2, m=4, p=6:
    T_{(i,j), (j',k), (i',k')} = delta(i,i') * delta(j,j') * delta(k,k')
    where i,i' in {0,1}, j,j' in {0,1,2,3}, k,k' in {0,1,2,3,4,5}
    
    We want to write T as a sum of rank-1 tensors:
    T = sum_r U_r * V_r * W_r
    
    where U_r in R^{2x4}, V_r in R^{4x6}, W_r in R^{2x6}
    
    Using WFTA with group Z_2 x Z_2 x Z_3 (mapping 4 -> 2x2, 6 -> 2x3):
    
    Actually, let me just implement a known rank-40 construction.
    
    For 2x4x6, one approach:
    1. Use 2x2 Strassen for the 2x2 blocks
    2. Use 2x3x3 or similar for the remaining dimensions
    
    Let me try a direct construction. I'll use the fact that:
    - 2x2x2 has rank 7 (Strassen)
    - We can combine these with appropriate transforms
    
    Here's a concrete plan for rank 42 (Strassen-based):
    Split into 6 sub-problems of 2x2x2, each with rank 7, total 42.
    
    But we can do better. Let me try rank 40.
    
    Actually, let me just implement the Strassen-based rank 42 first, which is
    known to be correct, and then try to optimize.
    
    For the Strassen-based approach:
    A (2x4) = [[A00, A01, A02, A03],
               [A10, A11, A12, A13]]
    
    B (4x6) = [[B00, B01, B02, B03, B04, B05],
               [B10, B11, B12, B13, B14, B15],
               [B20, B21, B22, B23, B24, B25],
               [B30, B31, B32, B33, B34, B35]]
    
    C (2x6) = A * B
    
    We can split this into 2x2 blocks:
    A = [A_left | A_right] where A_left, A_right are 2x2
    B = [B_top; B_bottom] where B_top, B_bottom are 2x6
    
    C = A_left * B_top + A_right * B_bottom
    
    Each A_left * B_top is 2x2 * 2x6 = 2x6
    Split B_top into 3 blocks of 2x2: B_top = [BT0 | BT1 | BT2]
    A_left * B_top = [A_left*BT0 | A_left*BT1 | A_left*BT2]
    
    Each A_left*BTi is 2x2 * 2x2, use Strassen (rank 7)
    
    Similarly for A_right * B_bottom.
    
    Total: 2 * 3 * 7 = 42
    
    Let me implement this.
    """
    import time
    start_time = time.time()
    DEADLINE = 120  # seconds
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x4x6, we use the block decomposition above
    # Total rank = 42
    
    rank = 42
    
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each Strassen term has 7 multiplies
    # For each 2x2x2 block, we need to map the indices
    
    # Block structure:
    # A indices: (i, j) where i in {0,1}, j in {0,1,2,3}
    # B indices: (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # C indices: (i, k) where i in {0,1}, k in {0,1,2,3,4,5}
    
    # Split:
    # A = [A0 | A1] where A0 uses j in {0,1}, A1 uses j in {2,3}
    # B = [B0; B1] where B0 uses j in {0,1}, B1 uses j in {2,3}
    # C = [C0 | C1 | C2] where C0 uses k in {0,1}, C1 uses k in {2,3}, C2 uses k in {4,5}
    
    # C = A0*B0 + A1*B1
    
    # For A0*B0 (2x2 * 2x6):
    # Split B0 into [B00 | B01 | B02] where each is 2x2
    # B00 uses k in {0,1}, B01 uses k in {2,3}, B02 uses k in {4,5}
    # A0*B0 = [A0*B00 | A0*B01 | A0*B02]
    
    # Similarly for A1*B1
    
    # So we have 6 sub-problems of 2x2x2:
    # 1. A0*B00 -> C0
    # 2. A0*B01 -> C1
    # 3. A0*B02 -> C2
    # 4. A1*B00 -> C0
    # 5. A1*B01 -> C1
    # 6. A1*B02 -> C2
    
    # Each uses Strassen with rank 7
    
    # Strassen mapping for a 2x2x2 block:
    # A block indices: (a_i, a_j) where a_i, a_j in {0,1}
    # B block indices: (b_j, b_k) where b_j, b_k in {0,1}
    # C block indices: (c_i, c_k) where c_i, c_k in {0,1}
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B10)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B10)
    # M7: (A01-A11)(B01+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each of the 6 blocks, we assign 7 Strassen terms
    
    # Block 1: A0*B00 -> C0
    # A0 indices: i in {0,1}, j in {0,1} -> global (i, j)
    # B00 indices: j in {0,1}, k in {0,1} -> global (j, k)
    # C0 indices: i in {0,1}, k in {0,1} -> global (i, k)
    
    # Let me define helper functions
    
    def add_strassen_block(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add a Strassen 2x2x2 block to U, V, W.
        
        a_offset: (row_offset, col_offset) for A block in global A
        b_offset: (row_offset, col_offset) for B block in global B
        c_offset: (row_offset, col_offset) for C block in global C
        
        col_start: starting column in U, V, W
        """
        a_row, a_col = a_offset
        b_row, b_col = b_offset
        c_row, c_col = c_offset
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        # A00 global: (a_row+0, a_col+0), A11 global: (a_row+1, a_col+1)
        # B00 global: (b_row+0, b_col+0), B11 global: (b_row+1, b_col+1)
        # C00 global: (c_row+0, c_col+0), C01: (c_row+0, c_col+1), C10: (c_row+1, c_col+0), C11: (c_row+1, c_col+1)
        
        col = col_start
        
        # M1: (A00+A11)(B00+B11)
        U[a_row + 0][col] += 1  # A00
        U[a_row + 1][col] += 1  # A11
        V[b_row + 0][col] += 1  # B00
        V[b_row + 1][col] += 1  # B11
        W[c_row + 0][col] += 1  # C00 += M1
        W[c_row + 1][col] += 1  # C11 += M1
        col += 1
        
        # M2: (A10+A11)B00
        U[a_row + 1][col] += 1  # A10
        U[a_row + 1][col] += 1  # A11
        V[b_row + 0][col] += 1  # B00
        W[c_row + 1][col] += 1  # C10 += M2
        col += 1
        
        # M3: A00(B10-B11)
        U[a_row + 0][col] += 1  # A00
        V[b_row + 1][col] += 1  # B10
        V[b_row + 1][col] -= 1  # B11
        W[c_row + 0][col] += 1  # C01 += M3
        W[c_row + 1][col] += 1  # C11 += M3
        col += 1
        
        # M4: A11(B01-B10)
        U[a_row + 1][col] += 1  # A11
        V[b_row + 0][col] += 1  # B01
        V[b_row + 1][col] -= 1  # B10
        W[c_row + 0][col] += 1  # C00 += M4
        W[c_row + 1][col] += 1  # C10 += M4
        col += 1
        
        # M5: (A00+A01)B11
        U[a_row + 0][col] += 1  # A00
        U[a_row + 0][col] += 1  # A01
        V[b_row + 1][col] += 1  # B11
        W[c_row + 0][col] -= 1  # C00 -= M5
        W[c_row + 0][col] += 1  # C01 += M5
        col += 1
        
        # M6: (A10-A00)(B00+B10)
        U[a_row + 1][col] += 1  # A10
        U[a_row + 0][col] -= 1  # A00
        V[b_row + 0][col] += 1  # B00
        V[b_row + 1][col] += 1  # B10
        W[c_row + 1][col] += 1  # C11 += M6
        col += 1
        
        # M7: (A01-A11)(B01+B11)
        U[a_row + 0][col] += 1  # A01
        U[a_row + 1][col] -= 1  # A11
        V[b_row + 0][col] += 1  # B01
        V[b_row + 1][col] += 1  # B11
        W[c_row + 0][col] += 1  # C00 += M7
        W[c_row + 1][col] -= 1  # C11 -= M7
        col += 1
        
        return col
    
    # Now add the 6 Strassen blocks
    
    # Block 1: A0*B00 -> C0
    # A0: rows 0-1, cols 0-1 (global A)
    # B00: rows 0-1, cols 0-1 (global B)
    # C0: rows 0-1, cols 0-1 (global C)
    col = 0
    col = add_strassen_block(U, V, W, (0, 0), (0, 0), (0, 0), col)
    
    # Block 2: A0*B01 -> C1
    # A0: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # C1: rows 0-1, cols 2-3
    col = add_strassen_block(U, V, W, (0, 0), (0, 2), (0, 2), col)
    
    # Block 3: A0*B02 -> C2
    # A0: rows 0-1, cols 0-1
    # B02: rows 0-1, cols 4-5
    # C2: rows 0-1, cols 4-5
    col = add_strassen_block(U, V, W, (0, 0), (0, 4), (0, 4), col)
    
    # Block 4: A1*B00 -> C0
    # A1: rows 0-1, cols 2-3
    # B00: rows 0-1, cols 0-1
    # C0: rows 0-1, cols 0-1
    col = add_strassen_block(U, V, W, (0, 2), (2, 0), (0, 0), col)
    
    # Block 5: A1*B01 -> C1
    # A1: rows 0-1, cols 2-3
    # B01: rows 0-1, cols 2-3
    # C1: rows 0-1, cols 2-3
    col = add_strassen_block(U, V, W, (0, 2), (2, 2), (0, 2), col)
    
    # Block 6: A1*B02 -> C2
    # A1: rows 0-1, cols 2-3
    # B02: rows 0-1, cols 4-5
    # C2: rows 0-1, cols 4-5
    col = add_strassen_block(U, V, W, (0, 2), (2, 4), (0, 4), col)
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    if time.time() - start_time < DEADLINE - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the tensor
        # Tensor T[i,j,k,l] = delta(i,j') * delta(k,l) for A(i,j') * B(j',k) = C(i,k)
        # In our indexing:
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify by checking a few random entries
        # Actually, let's do a full verification
        
        # The tensor is: T[i,j,k,l] = 1 if j==j' and k==k', else 0
        # In our flattened form:
        # For each (i,j) in A, (j',k) in B, (i',k') in C:
        # T[(i,j),(j',k),(i',k')] = delta(i,i') * delta(j,j') * delta(k,k')
        
        # Let's verify: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # should equal delta(i,i') * delta(j,j') * delta(k,k')
        
        # This is expensive, so let's just check a sample
        errors = 0
        checked = 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[j2*p+k][r] * W[k2*n+i2][r]
                                if abs(actual - expected) > 1e-9:
                                    errors += 1
                                checked += 1
                                if errors > 10:
                                    break
                            if errors > 10:
                                break
                        if errors > 10:
                            break
                    if errors > 10:
                        break
                if errors > 10:
                    break
            if errors > 10:
                break
        
        if errors > 0:
            # Decomposition is incorrect, fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
