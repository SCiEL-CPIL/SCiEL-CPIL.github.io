import numpy as np

def generate_strassen_decomp(n):
    """
    Generates U, V, W for n x n x n matrix multiplication using Strassen.
    Returns lists of lists for U, V, W.
    """
    if n == 2:
        # Base case: 2x2x2 Strassen
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 += M1, C11 += M1 ? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices: 0:00, 1:01, 2:10, 3:11
        # V indices: 0:00, 1:01, 2:10, 3:11
        # W indices: 0:00, 1:01, 2:10, 3:11
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = -1; W[2][0] = 1; W[3][0] = 1
        
        # Term 2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # Term 3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # Term 6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    else:
        # Recursive case
        # Split n into n1, n2. For Strassen, usually n1=n2=n/2.
        # This works for powers of 2.
        half = n // 2
        U_sub, V_sub, W_sub = generate_strassen_decomp(half)
        rank_sub = len(U_sub[0])
        rank = rank_sub * rank_sub
        
        U = [[0]*rank for _ in range(n*n)]
        V = [[0]*rank for _ in range(n*n)]
        W = [[0]*rank for _ in range(n*n)]
        
        # We need to map the 7 block operations to the scalar operations.
        # The 7 block operations are defined by the Strassen formulas.
        # For each block operation s (0..6), we have a linear combination of A blocks and B blocks.
        # Then we apply the sub-Strassen (7 scalar ops) to the resulting 2x2 matrices?
        # No, the "matrices" in the block operation are half-size matrices.
        # So for each of the 7 block terms, we instantiate the sub-decomposition.
        
        # Define the 7 Strassen block terms for n x n
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        
        # Term 1: (A00+A11)(B00+B11)
        # Term 2: (A10+A11)B00
        # Term 3: A00(B01-B11)
        # Term 4: A11(B10-B00)
        # Term 5: (A00+A01)B11
        # Term 6: (A10-A00)(B00+B01)
        # Term 7: (A01-A11)(B10+B11)
        
        # For each term, we define the coefficients for A blocks and B blocks.
        # Then for each scalar index in the sub-decomp, we add the contribution.
        
        # This approach is tricky because the sub-decomp U, V, W operate on the "flattened" half-matrices.
        # We need to map the indices of the half-matrices to the full matrix indices.
        
        # Let's use a different strategy:
        # The tensor for n x n x n is the "Strassen tensor" which is a sum of 7 rank-1 tensors? No, rank 7 is for 2x2.
        # For 4x4, rank 49.
        # The decomposition is:
        # T_4 = sum_{s=1}^7 sum_{t=1}^7 u_{s,t} \otimes v_{s,t} \otimes w_{s,t}
        
        # We can construct this by iterating s (block term) and t (scalar term within sub-strassen).
        
        # Helper to get block indices
        def get_block_indices(i, j, n, half):
            # i, j are indices in the n x n matrix
            # returns (block_row, block_col, in_row, in_col)
            br = i // half
            bc = j // half
            ir = i % half
            ic = j % half
            return br, bc, ir, ic

        # We need the structure of the 7 Strassen terms.
        # Term s involves A blocks with coeffs a_s[br][bc] and B blocks with coeffs b_s[br][bc].
        # And the result C blocks get coeffs c_s[br][bc].
        
        # Strassen Coefficients for Blocks:
        # s=0: A: (0,0):1, (1,1):1. B: (0,0):1, (1,1):1. C: (0,0):1, (0,1):-1, (1,0):1, (1,1):1
        # s=1: A: (1,0):1, (1,1):1. B: (0,0):1. C: (1,0):1, (1,1):-1
        # s=2: A: (0,0):1. B: (0,1):1, (1,1):-1. C: (0,1):1, (1,1):1
        # s=3: A: (1,1):1. B: (1,0):1, (0,0):-1. C: (0,0):1, (1,0):1
        # s=4: A: (0,0):1, (0,1):1. B: (1,1):1. C: (0,0):-1, (1,1):1
        # s=5: A: (1,0):1, (0,0):-1. B: (0,0):1, (0,1):1. C: (1,1):1
        # s=6: A: (0,1):1, (1,1):-1. B: (1,0):1, (1,1):1. C: (0,0):1, (1,1):1
        
        a_coeffs = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1}
        ]
        b_coeffs = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1}
        ]
        c_coeffs = [
            {(0,0):1, (0,1):-1, (1,0):1, (1,1):1},
            {(1,0):1, (1,1):-1},
            {(0,1):1, (1,1):1},
            {(0,0):1, (1,0):1},
            {(0,0):-1, (1,1):1},
            {(1,1):1},
            {(0,0):1, (1,1):1}
        ]
        
        for s in range(7):
            for t in range(rank_sub):
                col = s * rank_sub + t
                
                # Process U (A side)
                # U_sub[i_sub][t] gives coeff for A_sub[i_sub] in scalar term t.
                # A_sub is flattened half x half matrix.
                # We need to map A_sub indices to full A indices.
                # But A_sub is formed by linear combination of A blocks.
                # This means for a specific scalar term t, the "A matrix" is X_s.
                # X_s is a half x half matrix.
                # The entries of X_s are sums of entries from A blocks.
                # U_sub[i_sub][t] is the coeff of X_s[i_sub_row, i_sub_col] in the t-th scalar product.
                # X_s[i_sub_row, i_sub_col] = sum_{br,bc} a_coeffs[s][(br,bc)] * A[br*half + i_sub_row, bc*half + i_sub_col]
                
                # So U[full_i][col] += U_sub[i_sub][t] * a_coeffs[s][(br,bc)]
                
                for i_sub in range(half*half):
                    coeff_u_sub = U_sub[i_sub][t]
                    if coeff_u_sub != 0:
                        i_sub_row = i_sub // half
                        i_sub_col = i_sub % half
                        for (br, bc), coeff_a in a_coeffs[s].items():
                            full_i = br * half + i_sub_row
                            # Wait, A is n x n. Index is i*m + j. Here m=n.
                            # So index is full_i * n + (bc * half + i_sub_col)
                            full_idx = full_i * n + (bc * half + i_sub_col)
                            U[full_idx][col] += coeff_u_sub * coeff_a
                            
                # Process V (B side)
                for j_sub in range(half*half):
                    coeff_v_sub = V_sub[j_sub][t]
                    if coeff_v_sub != 0:
                        j_sub_row = j_sub // half
                        j_sub_col = j_sub % half
                        for (br, bc), coeff_b in b_coeffs[s].items():
                            # B index: j * p + k. Here p=n.
                            # B is n x n.
                            # j is row, k is col.
                            # j = br * half + j_sub_row
                            # k = bc * half + j_sub_col
                            full_j = br * half + j_sub_row
                            full_k = bc * half + j_sub_col
                            full_idx = full_j * n + full_k
                            V[full_idx][col] += coeff_v_sub * coeff_b
                            
                # Process W (C side)
                for k_sub in range(half*half):
                    coeff_w_sub = W_sub[k_sub][t]
                    if coeff_w_sub != 0:
                        k_sub_row = k_sub // half
                        k_sub_col = k_sub % half
                        for (br, bc), coeff_c in c_coeffs[s].items():
                            # C index: k * n + i. Here C is n x n.
                            # k is col, i is row.
                            # k = bc * half + k_sub_col
                            # i = br * half + k_sub_row
                            full_k = bc * half + k_sub_col
                            full_i = br * half + k_sub_row
                            full_idx = full_k * n + full_i
                            W[full_idx][col] += coeff_w_sub * coeff_c
                            
        return U, V, W

def construct():
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    U1, V1, W1 = generate_strassen_decomp(4)
    
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    # A is 4x4, B_part is 4x1, C_part is 4x1
    # Rank 16
    U2 = [[0]*16 for _ in range(16)]
    V2 = [[0]*16 for _ in range(20)] # m*p = 4*5 = 20. But we only use last col of B?
    # Wait, V shape is (m*p, rank).
    # For the 4x4x1 part, we multiply A (4x4) by B[:, 4] (4x1).
    # The V matrix corresponds to B.
    # The indices in V for this part should only touch the column k=4 of B.
    # B index is j*p + k. j in 0..3, k=4.
    # So V indices are j*5 + 4.
    
    # Let's just build the full 4x4x5 decomposition directly using the logic above?
    # It's easier to combine the two parts.
    
    rank_total = 49 + 16
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Fill Part 1 (cols 0-48)
    for i in range(n*m):
        for k in range(49):
            U[i][k] = U1[i][k]
    for i in range(m*n): # V1 is 4x4, so 16 rows. But V is 20 rows.
        for k in range(49):
            V[i][k] = V1[i][k]
    for i in range(p*n): # W1 is 4x4, so 16 rows. But W is 20 rows.
        for k in range(49):
            W[i][k] = W1[i][k]
            
    # Fill Part 2 (cols 49-64)
    # Schoolbook for A (4x4) * B_col (4x1)
    # C_col (4x1)
    col_offset = 49
    for i in range(n):
        for j in range(m):
            # A[i,j] * B[j,4] -> C[i,4]
            U[i*m + j][col_offset] = 1
            V[j*p + 4][col_offset] = 1
            W[4*n + i][col_offset] = 1 # k=4, i=i. Index k*n + i.
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
