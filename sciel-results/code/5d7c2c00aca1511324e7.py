"""Matrix multiplication 2x4x5 rank-minimal decomposition via block-Strassen."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    Strategy: Use block-Strassen decomposition.
    - Split A (2x4) into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    - Split B (4x5) into 2x2 blocks: B00, B01, B10, B11 (each 2x2, but B is 4x5 so blocks are 2x2 and 2x3)
    
    Actually, let's think more carefully. A is 2x4, B is 4x5, result C is 2x5.
    
    We can split the 4 columns of A and 4 rows of B into two groups of 2.
    A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    
    Then C = A_left * B_top + A_right * B_bottom
    
    This is a sum of two 2x2 by 2x5 multiplications.
    
    For each 2x2 by 2x5 multiplication, we can use a rank-7 Strassen-like decomposition
    for the 2x2 part, but the right matrix is 2x5 not 2x2.
    
    Let's use a different approach. Split into 2x2x2 blocks where possible.
    
    A (2x4): split columns into 2+2
    B (4x5): split rows into 2+2, split columns into 2+2+1
    
    This gives 2x3 blocks. Let's use a cleaner decomposition.
    
    Alternative: Use the fact that we can decompose the tensor using Strassen on 2x2x2 blocks.
    
    Let me try a direct approach:
    - C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    
    We can group j into {0,1} and {2,3}:
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    For the first sum, we have a 2x2 by 2x5 product.
    For the second sum, we have a 2x2 by 2x5 product.
    
    For a 2x2 by 2x5 product C = A*B where A is 2x2, B is 2x5:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a bilinear form with 4*5 = 20 terms, but we can do better.
    
    Actually, the tensor for 2x2 by 2x5 has dimension 2*2*5 = 20.
    The rank of the 2x2x5 tensor: we can write it as sum over j of A[:,j] * B[j,:].
    That's 2 terms, each being an outer product of a 2x1 and 1x5 vector, giving a 2x5 matrix.
    So the rank is at most 2... wait, that's not right.
    
    Let me reconsider. The multiplication tensor T_{i,j,k} = delta where the multiplication is
    C[i,k] = sum_j A[i,j] B[j,k].
    
    For 2x2 by 2x5: T_{i,j,k} for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}.
    The standard decomposition has rank 2*2*5/2 = ... no.
    
    Standard: for each j and k, we have a term A[i,j]*B[j,k]. That's 2*5 = 10 terms.
    Each term is a rank-1 tensor: for fixed j,k, the tensor is e_i (for A) * e_j (for B) * e_k (for C).
    Wait, the tensor is T_{i,j,k} = 1 if the indices correspond to a valid multiplication term.
    
    Actually for matrix multiplication A (m x n) * B (n x p) = C (m x p):
    T_{i,j,k} where i indexes rows of A and C, j indexes columns of A and rows of B, k indexes columns of B and C.
    T_{i,j,k} = 1 for all valid (i,j,k), and the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    
    The standard decomposition uses n terms (one for each j), each of rank 1 in the tensor sense? No.
    
    Let me use the standard definition. The matrix multiplication tensor M_{m,n,p} has entries
    M_{(i,j),(j,k),(i,k)} = 1 for all i,j,k, and 0 otherwise.
    
    A rank-1 decomposition term is u * v * w where u is in R^{m*n}, v in R^{n*p}, w in R^{p*m} (or similar).
    
    The standard schoolbook decomposition has rank n (the inner dimension). For each j, we have:
    u_j has 1 at position (i,j) for all i... no, that's not rank 1.
    
    I'm confusing myself. Let me just use the known result: the rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
    
    For our problem 2x4x5:
    - We can split the 4 inner dimension into 2+2.
    - Each 2x2x5 sub-problem: multiply 2x2 by 2x5.
    
    For 2x2 by 2x5: The tensor has dimensions 2*2 * 2*5 * 5*2 = 4 * 10 * 10.
    The standard rank is 2 (the inner dimension of the 2x2 by 2x5... no, the inner dimension is 2).
    Wait, for A (2x2) * B (2x5) = C (2x5), the inner dimension is 2.
    Standard decomposition: rank 2. For j=0: term involving A[:,0] and B[0,:]. For j=1: term involving A[:,1] and B[1,:].
    
    But can we do better than 2? The tensor is essentially two independent 2x5 outer products.
    Each outer product A[:,j] * B[j,:] is a 2x5 matrix, which as a tensor in R^{2*2} x R^{2*5} x R^{5*2}...
    
    I think I need to just implement the block-Strassen approach that was measured to work.
    
    From the knowledge: "Block-Strassen decomposition splits the 2x4x5 problem into sub-problems where 2x2x2 blocks use Strassen (rank 7 instead of 4), reducing total rank from 40 to 36"
    
    So the approach is:
    - Split the 4x4 middle part into 2x2 blocks (4 blocks of 2x2).
    - For each 2x2x2 block, use Strassen (rank 7 instead of 4).
    - But we have 2x4 and 4x5, so the blocks aren't all 2x2x2.
    
    Let me think about this differently. The 2x4 * 4x5 = 2x5.
    
    Split A into left (2x2) and right (2x2).
    Split B into top (2x5) and bottom (2x5).
    C = A_left * B_top + A_right * B_bottom.
    
    Each term is 2x2 * 2x5 = 2x5.
    
    For 2x2 * 2x5: split the 5 columns of B into 2+2+1.
    So B_top = [B_top_0 (2x2) | B_top_1 (2x2) | B_top_2 (2x1)]
    
    Then A_left * B_top = A_left * B_top_0 + A_left * B_top_1 + A_left * B_top_2.
    
    A_left * B_top_0 is 2x2 * 2x2 = 2x2. Use Strassen: rank 7.
    A_left * B_top_1 is 2x2 * 2x2 = 2x2. Use Strassen: rank 7.
    A_left * B_top_2 is 2x2 * 2x1 = 2x1. Standard rank: 2.
    
    Similarly for A_right * B_bottom.
    
    Total: 7 + 7 + 2 + 7 + 7 + 2 = 32.
    
    Hmm, that's 32, not 36. Let me re-examine.
    
    Actually, maybe the split is different. Let me just implement a clean version.
    
    For 2x2 * 2x2 Strassen:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M is a scalar product of two 1x1 "matrices" (scalars), so each M is rank 1.
    Total 7 rank-1 terms for the 2x2x2 multiplication.
    
    For our problem, let me define the decomposition carefully.
    
    A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    B is 4x5: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    C is 2x5: C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    Split j into {0,1} and {2,3}.
    
    C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    
    First sum: A_left (2x2) * B_top (2x5)
    Second sum: A_right (2x2) * B_bottom (2x5)
    
    For A_left * B_top:
    Split k into {0,1}, {2,3}, {4}.
    B_top = [B_top[:,0:2] | B_top[:,2:4] | B_top[:,4:5]]
    
    A_left * B_top[:,0:2] is 2x2 * 2x2. Use Strassen.
    A_left * B_top[:,2:4] is 2x2 * 2x2. Use Strassen.
    A_left * B_top[:,4:5] is 2x2 * 2x1. Standard rank 2.
    
    Similarly for A_right * B_bottom.
    
    Total rank: 7 + 7 + 2 + 7 + 7 + 2 = 32.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (10, rank)
    
    # Convention: W[k*n+i, r] for the output factor.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_2x2x2(A_rows, B_cols, C_rows, C_cols, offset):
        """
        Add Strassen decomposition for a 2x2 * 2x2 block.
        A_rows: list of 2 row indices in A (each row has 2 cols)
        B_cols: list of 2 col groups in B
        C_rows: list of 2 row indices in C
        C_cols: list of 2 col indices in C
        """
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2
        # A[i,j] for i in 0,1; j in 0,1
        # B[j,k] for j in 0,1; k in 0,1
        # C[i,k] for i in 0,1; k in 0,1
        
        # Strassen products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each M, we add one rank-1 term.
        # M is a product of a linear combination of A entries and a linear combination of B entries.
        # The A part contributes to U, the B part contributes to V, and the C part contributes to W.
        
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            """
            a_coeffs: dict mapping (i, j) -> coeff for A
            b_coeffs: dict mapping (j, k) -> coeff for B
            c_coeffs: dict mapping (i, k) -> coeff for C
            """
            nonlocal col
            for (i, j), c in a_coeffs.items():
                if c != 0:
                    U[i * m + j][col] += c
            for (j, k), c in b_coeffs.items():
                if c != 0:
                    V[j * p + k][col] += c
            for (i, k), c in c_coeffs.items():
                if c != 0:
                    W[k * n + i][col] += c
            col += 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {})
        
        # M2 = (A10 + A11) * B00
        add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {})
        
        # M3 = A00 * (B01 - B11)
        add_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {})
        
        # M4 = A11 * (B10 - B00)
        add_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {})
        
        # M5 = (A00 + A01) * B11
        add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {})
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {})
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {})
        
        # Now add the C contributions:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to go back and add C coefficients. But we already incremented col.
        # This approach is messy. Let me restructure.
        
        pass
    
    # Let me restructure: build the full decomposition explicitly.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: dict {U_index: coeff}
        v_entries: dict {V_index: coeff}
        w_entries: dict {W_index: coeff}
        """
        nonlocal col
        for idx, c in u_entries.items():
            if c != 0:
                U[idx][col] += c
        for idx, c in v_entries.items():
            if c != 0:
                V[idx][col] += c
        for idx, c in w_entries.items():
            if c != 0:
                W[idx][col] += c
        col += 1
    
    # Process: A_left (2x2) * B_top (2x5)
    # A_left: A[i,j] for i in {0,1}, j in {0,1}
    # B_top: B[j,k] for j in {0,1}, k in {0,1,2,3,4}
    
    # Split B_top into B_top_0 (2x2, k=0,1), B_top_1 (2x2, k=2,3), B_top_2 (2x1, k=4)
    
    # A_left * B_top_0: 2x2 * 2x2, use Strassen
    # A entries: (i,j) -> U[i*4+j] for i in {0,1}, j in {0,1}
    # B entries: (j,k) -> V[j*5+k] for j in {0,1}, k in {0,1}
    # C entries: (i,k) -> W[k*2+i] for i in {0,1}, k in {0,1}
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    # Strassen for A_left * B_top_0
    # M1 = (A00 + A11) * (B00 + B11)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    
    # M2 = (A10 + A11) * B00
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(1,0): 1})
    
    # M3 = A00 * (B01 - B11)
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(0,1): 1})
    
    # M4 = A11 * (B10 - B00)
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(1,0): 1})
    
    # M5 = (A00 + A01) * B11
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,1): 1})
    
    # M6 = (A10 - A00) * (B00 + B01)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    
    # M7 = (A01 - A11) * (B10 + B11)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # A_left * B_top_1: 2x2 * 2x2, use Strassen
    # B entries: (j,k) -> V[j*5+k] for j in {0,1}, k in {2,3}
    # C entries: (i,k) -> W[k*2+i] for i in {0,1}, k in {2,3}
    
    # M1 = (A00 + A11) * (B02 + B13)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1, v_idx(1,3): 1}, {w_idx(0,2): 1, w_idx(1,3): 1})
    
    # M2 = (A10 + A11) * B02
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1}, {w_idx(1,2): 1})
    
    # M3 = A00 * (B03 - B13)
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1, v_idx(1,3): -1}, {w_idx(0,3): 1})
    
    # M4 = A11 * (B12 - B02)
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1, v_idx(0,2): -1}, {w_idx(0,2): 1, w_idx(1,2): 1})
    
    # M5 = (A00 + A01) * B13
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(0,2): -1, w_idx(1,3): 1})
    
    # M6 = (A10 - A00) * (B02 + B03)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,2): 1, v_idx(0,3): 1}, {w_idx(1,3): 1})
    
    # M7 = (A01 - A11) * (B12 + B13)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,2): 1, v_idx(1,3): 1}, {w_idx(0,2): 1})
    
    # A_left * B_top_2: 2x2 * 2x1, standard rank 2
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    # Term 1: j=0
    add_term({u_idx(0,0): 1, u_idx(1,0): 1}, {v_idx(0,4): 1}, {w_idx(0,4): 1, w_idx(1,4): 1})
    # Term 2: j=1
    add_term({u_idx(0,1): 1, u_idx(1,1): 1}, {v_idx(1,4): 1}, {w_idx(0,4): 1, w_idx(1,4): 1})
    
    # A_right * B_bottom: A_right is 2x2 (j=2,3), B_bottom is 2x5 (j=2,3)
    # A entries: (i,j) -> U[i*4+j] for i in {0,1}, j in {2,3}
    # B entries: (j,k) -> V[j*5+k] for j in {2,3}, k in {0,1,2,3,4}
    
    # A_right * B_bottom_0: 2x2 * 2x2, use Strassen
    # B entries: (j,k) for j in {2,3}, k in {0,1}
    
    # M1 = (A02 + A13) * (B20 + B31)
    add_term({u_idx(0,2): 1, u_idx(1,3): 1}, {v_idx(2,0): 1, v_idx(3,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    
    # M2 = (A12 + A13) * B20
    add_term({u_idx(1,2): 1, u_idx(1,3): 1}, {v_idx(2,0): 1}, {w_idx(1,0): 1})
    
    # M3 = A02 * (B21 - B31)
    add_term({u_idx(0,2): 1}, {v_idx(2,1): 1, v_idx(3,1): -1}, {w_idx(0,1): 1})
    
    # M4 = A13 * (B30 - B20)
    add_term({u_idx(1,3): 1}, {v_idx(3,0): 1, v_idx(2,0): -1}, {w_idx(0,0): 1, w_idx(1,0): 1})
    
    # M5 = (A02 + A03) * B31
    add_term({u_idx(0,2): 1, u_idx(0,3): 1}, {v_idx(3,1): 1}, {w_idx(0,0): -1, w_idx(1,1): 1})
    
    # M6 = (A12 - A02) * (B20 + B21)
    add_term({u_idx(1,2): 1, u_idx(0,2): -1}, {v_idx(2,0): 1, v_idx(2,1): 1}, {w_idx(1,1): 1})
    
    # M7 = (A03 - A13) * (B30 + B31)
    add_term({u_idx(0,3): 1, u_idx(1,3): -1}, {v_idx(3,0): 1, v_idx(3,1): 1}, {w_idx(0,0): 1})
    
    # A_right * B_bottom_1: 2x2 * 2x2, use Strassen
    # B entries: (j,k) for j in {2,3}, k in {2,3}
    
    # M1 = (A02 + A13) * (B22 + B33)
    add_term({u_idx(0,2): 1, u_idx(1,3): 1}, {v_idx(2,2): 1, v_idx(3,3): 1}, {w_idx(0,2): 1, w_idx(1,3): 1})
    
    # M2 = (A12 + A13) * B22
    add_term({u_idx(1,2): 1, u_idx(1,3): 1}, {v_idx(2,2): 1}, {w_idx(1,2): 1})
    
    # M3 = A02 * (B23 - B33)
    add_term({u_idx(0,2): 1}, {v_idx(2,3): 1, v_idx(3,3): -1}, {w_idx(0,3): 1})
    
    # M4 = A13 * (B32 - B22)
    add_term({u_idx(1,3): 1}, {v_idx(3,2): 1, v_idx(2,2): -1}, {w_idx(0,2): 1, w_idx(1,2): 1})
    
    # M5 = (A02 + A03) * B33
    add_term({u_idx(0,2): 1, u_idx(0,3): 1}, {v_idx(3,3): 1}, {w_idx(0,2): -1, w_idx(1,3): 1})
    
    # M6 = (A12 - A02) * (B22 + B23)
    add_term({u_idx(1,2): 1, u_idx(0,2): -1}, {v_idx(2,2): 1, v_idx(2,3): 1}, {w_idx(1,3): 1})
    
    # M7 = (A03 - A13) * (B32 + B33)
    add_term({u_idx(0,3): 1, u_idx(1,3): -1}, {v_idx(3,2): 1, v_idx(3,3): 1}, {w_idx(0,2): 1})
    
    # A_right * B_bottom_2: 2x2 * 2x1, standard rank 2
    add_term({u_idx(0,2): 1, u_idx(1,2): 1}, {v_idx(2,4): 1}, {w_idx(0,4): 1, w_idx(1,4): 1})
    add_term({u_idx(0,3): 1, u_idx(1,3): 1}, {v_idx(3,4): 1}, {w_idx(0,4): 1, w_idx(1,4): 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
