"""Matrix multiplication 4x5x5 - Linear Algebra Rank Minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Safety margin
    
    # Build schoolbook as baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Convert to list of lists for return, but use numpy for computation
    def to_list(arr):
        return arr.tolist()
        
    def verify(U, V, W, rank):
        # U, V, W are numpy arrays
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: Try to find a decomposition of rank R = 99.
    # We fix V and W to be the schoolbook V and W, but with one column removed (or modified).
    # Actually, a better linear approach:
    # Let's try to express the schoolbook tensor as a sum of 99 rank-1 tensors.
    # This is equivalent to finding U, V, W of width 99 such that the bilinear form matches.
    
    # A known technique for small rank reductions is to check if the tensor lies in the 
    # span of 99 rank-1 tensors. 
    # We can set up a system where we try to "merge" two schoolbook terms into one 
    # or modify the coefficients such that the rank drops.
    
    # However, a more robust approach for exact integer/half-integer solutions:
    # We can try to solve for U given V and W, or vice versa.
    # Let's try to find a rank 99 decomposition by setting V and W to be 
    # "almost" schoolbook but with one column zeroed out, and solving for U.
    # But U must also have rank 99.
    
    # Alternative: Use the fact that the matrix multiplication tensor is 
    # "non-degenerate". 
    # Let's try a specific constructive approach for rank 99.
    # If we can find a linear dependency among the 100 basis tensors, we can reduce rank.
    # The basis tensors are T_{i,j,k} with 1 at (i,j,k) in the U,V,W positions? No.
    # The schoolbook terms are:
    # Term c corresponds to (i_c, j_c, k_c).
    # U_c has 1 at (i_c*m + j_c).
    # V_c has 1 at (j_c*p + k_c).
    # W_c has 1 at (k_c*n + i_c).
    
    # If we can find coefficients a_c such that sum a_c U_c = 0, sum a_c V_c = 0, sum a_c W_c = 0?
    # No, that would mean the sum of tensors is 0. We need the sum of tensors to be the original tensor.
    
    # Let's try a different angle: 
    # Is it possible that rank(4x5x5) < 100?
    # Lower bound for matrix multiplication rank R(n,m,p) is n*m*p / (something)?
    # Actually, for square matrices, R(n) >= n^2. For 4x5x5, R >= 20? No, R >= n*m*p is schoolbook.
    # It is known that R(2,2,2) = 7 < 8.
    # R(3,3,3) = 23 < 27.
    # R(4,4,4) is at most 49? (Strassen).
    # For 4x5x5, we can use Strassen on the 4x4 part?
    # 4x5x5 = 4x4x4 + 4x4x1 + 4x1x4 + 4x1x1?
    # Let's decompose the 5x5 matrix B into blocks:
    # B = [B11 B12; B21 B22] where B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1.
    # A is 4x5: A = [A1 A2] where A1 is 4x4, A2 is 4x1.
    # C = A B = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # C is 4x5.
    
    # Let's compute the rank of each part:
    # Part 1: A1 B11 (4x4 * 4x4). Rank <= R(4,4,4). Best known for 4x4x4 is 49 (Strassen-like).
    # Part 2: A2 B21 (4x1 * 1x4). This is an outer product. Rank 1.
    # Part 3: A1 B12 (4x4 * 4x1). This is a matrix-vector product. Rank 4*1*4 = 16? No, rank is 4*4*1 = 16 schoolbook, but can it be lower?
    #    Actually, multiplying 4x4 by 4x1 is just 4 independent dot products.
    #    Wait, the output is 4x1. The input is 4x4 and 4x1.
    #    This is equivalent to 4 separate 4x1x1 multiplications? No.
    #    C_col = A1 * B12.
    #    This is a 4x4 matrix times a 4x1 vector.
    #    The tensor for this operation has rank 4*4*1 = 16?
    #    Actually, we can compute A1 * B12 by treating B12 as 4 scalars.
    #    C = sum_{j=0}^3 A1[:,j] * B12[j].
    #    This is a linear combination of 4 vectors.
    #    The rank of this specific bilinear form (4x4 * 4x1 -> 4x1) is 4*4 = 16?
    #    Let's check: U (4*4=16 dims), V (4*1=4 dims), W (4*1=4 dims).
    #    Schoolbook rank is 16. Can it be lower?
    #    For m=1, R(n,1,p) = n*p. So R(4,1,4) = 16. It cannot be lower.
    # Part 4: A2 B22 (4x1 * 1x1). Rank 4.
    
    # So if we use Strassen for the 4x4x4 part with rank 49, and schoolbook for the rest:
    # Total Rank = 49 (for 4x4x4) + 1 (for 4x1x4) + 16 (for 4x4x1) + 4 (for 4x1x1)
    # Wait, the dimensions don't align perfectly for simple addition because the indices overlap?
    # No, the matrix multiplication is linear.
    # C = A B.
    # A = [A1 A2], B = [B11 B12; B21 B22].
    # C = A1 B11 + A2 B21  (first 4 cols of C)
    # C = A1 B12 + A2 B22  (last 1 col of C)
    
    # Let's re-evaluate the ranks:
    # 1. A1 B11: 4x4 * 4x4 -> 4x4. Rank R(4,4,4).
    # 2. A2 B21: 4x1 * 1x4 -> 4x4. 
    #    A2 is 4x1, B21 is 1x4. This is an outer product.
    #    C_part = A2 * B21.
    #    This is a rank-1 matrix (outer product of vector A2 and vector B21).
    #    The tensor rank for this specific operation:
    #    Inputs: A2 (4), B21 (4). Output: 4x4 matrix (16 entries).
    #    C_{ik} = A2_i * B21_k.
    #    This is a single rank-1 term in the tensor sense?
    #    U (4 dims), V (4 dims), W (16 dims).
    #    U[i,0]=1, V[k,0]=1, W[i,k,0]=1.
    #    Yes, rank 1.
    # 3. A1 B12: 4x4 * 4x1 -> 4x1.
    #    Inputs: A1 (16), B12 (4). Output: 4 entries.
    #    C_i = sum_j A1_{ij} B12_j.
    #    This is 4 independent dot products of length 4.
    #    Rank is 4*4 = 16. (Since R(n,1,p) = n*p).
    # 4. A2 B22: 4x1 * 1x1 -> 4x1.
    #    Inputs: A2 (4), B22 (1). Output: 4 entries.
    #    C_i = A2_i * B22_0.
    #    This is 4 independent scalar multiplications.
    #    Rank is 4*1*1 = 4.
    
    # Total Rank = R(4,4,4) + 1 + 16 + 4 = R(4,4,4) + 21.
    # If R(4,4,4) is 49, Total = 70.
    # If R(4,4,4) is 50, Total = 71.
    # Even if R(4,4,4) is 40 (unlikely), Total = 61.
    # This is MUCH better than 100.
    
    # So the strategy is:
    # 1. Find a rank-minimal decomposition for 4x4x4.
    # 2. Construct the full 4x5x5 decomposition by combining it with the simple blocks.
    
    # How to get R(4,4,4)?
    # We can use the known 7-term Strassen decomposition for 2x2x2, and pad it.
    # Or use the 49-term decomposition for 4x4x4.
    # Actually, we can just use the schoolbook for 4x4x4 if we don't have a better one, 
    # but 64 + 21 = 85, which is still < 100.
    # Let's try to implement a Strassen-like decomposition for 4x4x4.
    
    # Strassen for 2x2x2:
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
    
    # We can build a 4x4x4 decomposition by splitting 4x4 into 2x2 blocks.
    # A = [A11 A12; A21 A22] (each 2x2)
    # B = [B11 B12; B21 B22] (each 2x2)
    # C = A B = [C11 C12; C21 C22] (each 2x2)
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Each term like A11 B11 is a 2x2x2 multiplication.
    # There are 8 such terms.
    # If we use Strassen for each, that's 8 * 7 = 56 terms.
    # But we can share computations?
    # Strassen's algorithm is usually applied recursively.
    # For 4x4, we can apply Strassen to the 2x2 blocks.
    # The number of multiplications for 4x4 using Strassen is 7^2 = 49.
    
    # So we can achieve rank 49 for 4x4x4.
    # Total rank for 4x5x5 = 49 + 21 = 70.
    
    # Let's implement this.
    
    # First, define the Strassen decomposition for 2x2x2.
    # We need to map the global indices to local indices.
    
    # Let's build the 49-term decomposition for 4x4x4.
    # This is a bit complex to hardcode. 
    # Alternative: Use a known library or algorithm? No, standard library only.
    # We can generate the Strassen decomposition programmatically.
    
    # Let's try a simpler approach first: Just use schoolbook for the 4x4x4 part.
    # Rank = 64 + 21 = 85.
    # This is already an improvement over 100.
    # Let's verify if 85 is valid.
    
    def construct_block_decomposition(rank_444):
        # rank_444 is the rank of the 4x4x4 part.
        # We will construct U, V, W for the full 4x5x5 problem.
        
        # Indices for A: i (0..3), j (0..4)
        # Indices for B: j (0..4), k (0..4)
        # Indices for C: k (0..4), i (0..3)
        
        # Split A into A1 (4x4, cols 0-3) and A2 (4x1, col 4)
        # Split B into B11 (4x4, rows 0-3, cols 0-3), B12 (4x1, rows 0-3, col 4),
        #           B21 (1x4, row 4, cols 0-3), B22 (1x1, row 4, col 4)
        
        # U shape: (20, rank_total)
        # V shape: (25, rank_total)
        # W shape: (20, rank_total)
        
        total_rank = rank_444 + 1 + 16 + 4
        U = np.zeros((20, total_rank))
        V = np.zeros((25, total_rank))
        W = np.zeros((20, total_rank))
        
        col_offset = 0
        
        # Part 1: A1 B11 (4x4 * 4x4)
        # Use schoolbook for this part for simplicity (rank 64)
        # If we had a better decomposition, we'd plug it in here.
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # A1 index: i, j (global j is 0..3)
                    # B11 index: j, k (global j is 0..3, global k is 0..3)
                    # C index: k, i (global k is 0..3, global i is 0..3)
                    
                    # U entry: A1[i,j] -> index i*5 + j
                    # V entry: B11[j,k] -> index j*5 + k
                    # W entry: C[k,i] -> index k*4 + i
                    
                    U[i*5 + j, col_offset] = 1
                    V[j*5 + k, col_offset] = 1
                    W[k*4 + i, col_offset] = 1
                    col_offset += 1
        
        # Part 2: A2 B21 (4x1 * 1x4) -> Outer Product
        # A2 is col 4 of A. Indices i (0..3), j=4.
        # B21 is row 4 of B. Indices j=4, k (0..3).
        # C part is the first 4 cols of C. Indices k (0..3), i (0..3).
        # C_{ki} = A2_i * B21_k.
        # This is a single rank-1 term.
        
        # U entry: A2[i] -> index i*5 + 4
        # V entry: B21[k] -> index 4*5 + k = 20 + k
        # W entry: C[k,i] -> index k*4 + i
        
        for i in range(4):
            for k in range(4):
                # This is not a single term, it's a sum?
                # No, C_{ki} = A2_i * B21_k.
                # The tensor is sum over i,k of (A2_i * B21_k) * E_{ki}.
                # This can be written as a single rank-1 term in the tensor decomposition?
                # Let's check:
                # U has 1 at (i, 4). V has 1 at (4, k). W has 1 at (k, i).
                # If we set U[i*5+4, col] = 1, V[20+k, col] = 1, W[k*4+i, col] = 1,
                # then the product is 1*1*1 = 1 for all i,k.
                # This contributes 1 to C_{ki} for all i,k.
                # But we need C_{ki} = A2_i * B21_k.
                # The inputs A2 and B21 are variables.
                # The decomposition must be linear in A and B.
                # The term is U_a * V_b * W_c.
                # If U depends on A and V depends on B, then U_a * V_b is the product.
                # Here U_a is a linear combination of A entries.
                # V_b is a linear combination of B entries.
                # So U_a * V_b is a bilinear form.
                # For the outer product A2 * B21, we can set:
                # U[i*5+4, col] = 1 (coefficient of A2_i)
                # V[20+k, col] = 1 (coefficient of B21_k)
                # W[k*4+i, col] = 1 (coefficient of C_ki)
                # Then the contribution is A2_i * B21_k * 1.
                # This matches C_{ki}.
                # So yes, it is a single term.
                
                pass # We only need to set the columns once.
        
        # Set the column for Part 2
        for i in range(4):
            U[i*5 + 4, col_offset] = 1
        for k in range(4):
            V[20 + k, col_offset] = 1
        for i in range(4):
            for k in range(4):
                W[k*4 + i, col_offset] = 1
        col_offset += 1
        
        # Part 3: A1 B12 (4x4 * 4x1)
        # A1 is 4x4. B12 is 4x1 (col 4 of B).
        # C part is the last col of C (col 4). Indices k=4, i (0..3).
        # C_i = sum_j A1_ij * B12_j.
        # This is 4 dot products.
        # Each dot product is a 4x1x1 multiplication? No, it's a sum of 4 products.
        # The rank of this part is 16.
        # We can decompose it into 16 schoolbook terms.
        
        for i in range(4):
            for j in range(4):
                # A1_ij * B12_j contributes to C_i (where k=4).
                # U entry: A1_ij -> i*5 + j
                # V entry: B12_j -> j*5 + 4
                # W entry: C_i (k=4) -> 4*4 + i = 16 + i
                
                U[i*5 + j, col_offset] = 1
                V[j*5 + 4, col_offset] = 1
                W[16 + i, col_offset] = 1
                col_offset += 1
        
        # Part 4: A2 B22 (4x1 * 1x1)
        # A2 is 4x1. B22 is 1x1.
        # C part is the last col of C (col 4). Indices k=4, i (0..3).
        # C_i = A2_i * B22.
        # This is 4 scalar multiplications.
        # Rank 4.
        
        for i in range(4):
            # U entry: A2_i -> i*5 + 4
            # V entry: B22 -> 4*5 + 4 = 24
            # W entry: C_i (k=4) -> 16 + i
            
            U[i*5 + 4, col_offset] = 1
            V[24, col_offset] = 1
            W[16 + i, col_offset] = 1
            col_offset += 1
            
        return U, V, W, total_rank

    # Construct with rank 64 for the 4x4x4 part
    U, V, W, rank = construct_block_decomposition(64)
    
    if verify(U, V, W, rank):
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    else:
        # Fallback to schoolbook
        best_rank = schoolbook_rank
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
