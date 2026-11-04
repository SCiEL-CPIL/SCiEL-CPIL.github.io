"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import time

# EVOLVE-BLOCK-START
def _matmul_tensor(n, m, p):
    """Build the flattened matrix multiplication tensor as a vector of length n*m*p.
    Convention: T[i, j, k] = 1 if we are computing (AB)[i,k] = sum_j A[i,j]*B[j,k].
    We flatten in order (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    """
    # The tensor T has entries T[i,j,k] which is 1 for all valid (i,j,k).
    # We need to decompose T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product of 3 vectors)
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The index for U is (i,j) -> i*m+j
    # The index for V is (j,k) -> j*p+k
    # The index for W is (k,i) -> k*n+i   (trace convention W[k*n+i, r])
    #
    # The decomposition condition is:
    # For all (i,j,k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is a bilinear algorithm. We can think of it as:
    # Fix k. Then for each (i,j), we need sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = T[i,j,k] = 1
    #
    # Alternatively, think of it as a 3-tensor decomposition.
    #
    # A standard approach: use block decomposition.
    # For 4x4 * 4x7, we can decompose into blocks.
    #
    # Let's try a block approach:
    # Split the 4x4 matrix into four 2x2 blocks, and the 4x7 into four 2x7 blocks? No.
    # Actually, let's split A (4x4) into 2x2 blocks (4 blocks), B (4x7) into 2x7 blocks (2 blocks).
    # That gives 4*2 = 8 products of 2x2 * 2x7.
    # Each 2x2 * 2x7 product has rank at most 2*2*7 = 28, but we can do better.
    #
    # Actually, let's think about it differently.
    # The total rank of schoolbook is 4*4*7 = 112.
    #
    # Let's try to use the fact that 4x4 can be done with rank 7 (Strassen-like) or similar.
    # Actually, for 2x2, Strassen gives rank 7.
    # For 4x4, we can use 4x4 = 4 blocks of 2x2, each with rank 7, total 28.
    # Then for the 4x7, we need to multiply 4x4 (rank 28 intermediates) by 4x7.
    #
    # Let's try a different approach: use the bilinear structure.
    #
    # Actually, let me try a simple block decomposition:
    # A is 4x4, B is 4x7.
    # Split A into rows: A1 (2x4), A2 (2x4)
    # Split B into columns: B1 (4x3), B2 (4x4) -- not even
    #
    # Let's try:
    # A = [A11 A12; A21 A22] where each Ai is 2x2
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7
    #
    # C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    #
    # Each Ai*Bj is a 2x2 * 2x7 product.
    # The rank of 2x2 * 2x7: schoolbook is 2*2*7=28.
    # Can we do better? 2x2*2x7 can be seen as 7 independent 2x2*2x1 products.
    # 2x2*2x1 has rank 4 (schoolbook) or 3? No, 2x2*1 is just 2x2 matrix times scalar, rank 4.
    # Wait, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is rank 4 (schoolbook). Can it be done in rank 3?
    # The tensor is 2x2x1. The rank of the 2x2x1 tensor... 
    # Actually, for 2x2*2x1, the minimal rank is 4? No.
    # Let's think: we need to compute 4 outputs from 4 inputs (2 from A, 2 from B... wait, A has 4 entries, B has 2 entries).
    # Actually, A has 4 entries, B has 2 entries, total 6 inputs, 4 outputs.
    # The bilinear rank... I think it's 4.
    #
    # So 2x2*2x7 has rank at most 7*4 = 28? No, we can share.
    # Actually, 2x2*2x7: we compute C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=0..6.
    # This is 14 outputs.
    # Schoolbook: 2*2*7 = 28 multiplications.
    # Can we do better? 
    #
    # Let's try a different block structure.
    #
    # Actually, let me just try to use a known result or a simple improvement.
    #
    # For 4x4 * 4x7, one approach:
    # Use the fact that 4x4 matrix multiplication can be done with rank 21 (or similar).
    # Actually, the best known for 4x4 is 21? No, I think it's higher.
    #
    # Let me try a simpler approach: just use schoolbook but with some sharing.
    #
    # Actually, let's try the following:
    # Decompose 4x4 * 4x7 into 7 independent 4x4 * 4x1 problems.
    # Each 4x4 * 4x1 has rank 16 (schoolbook).
    # Total: 7*16 = 112. Same as schoolbook.
    #
    # Can we do 4x4 * 4x1 better?
    # C[i] = sum_j A[i,j]*B[j] for i=0..3.
    # This is 4 outputs, 16 inputs (4 from A... wait, A is 4x4 so 16 entries, B is 4x1 so 4 entries, total 20 inputs).
    # The bilinear rank of 4x4*4x1... 
    #
    # Actually, I think the minimal rank for 4x4*4x1 is 16 (schoolbook is optimal for this case?).
    #
    # Let me try a completely different approach: use random search or a known decomposition.
    #
    # Actually, let's try to use the "border rank" or "tensor rank" concepts.
    #
    # For now, let's try a block decomposition that might give a slight improvement.
    #
    # Split 4x4 into 2x2 blocks (4 blocks), and 4x7 into 2x7 blocks (2 blocks).
    # C = [A11 A12; A21 A22] * [B1; B2] = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    #
    # Each Ai*Bj is 2x2 * 2x7.
    # Let's compute the rank of 2x2 * 2x7.
    # C[i,k] = sum_{j=0}^1 A[i,j]*B[j,k]
    # We can use the following trick: for each k, we compute a 2x2 * 2x1 product.
    # But we can share across k.
    #
    # Actually, let's just compute 2x2*2x7 with schoolbook: rank = 2*2*7 = 28.
    # 4 such products: 4*28 = 112. Same as before.
    #
    # Let me try to find a better decomposition for 2x2*2x7.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # ...
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    #
    # We can factor: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is like a matrix-vector product where A[i,:] is a 2-vector and B is a 2x7 matrix.
    # The rank of this operation: we need to compute 2*7 = 14 outputs.
    # Schoolbook: 2*2*7 = 28.
    #
    # Can we do it in fewer? 
    # Let's think of it as: we have 4 parameters from A (A[0,0], A[0,1], A[1,0], A[1,1]) and 14 from B.
    # We want to compute 14 bilinear forms.
    #
    # I don't see an obvious improvement.
    #
    # Let me try a different strategy: use the known result that the rank of 4x4*4x4 is at most 49 (or similar).
    # Actually, I recall that for n x n, the rank is at most n^3, and for n=2 it's 7 (Strassen).
    # For n=4, using Strassen recursively: 4x4 = 4 blocks of 2x2, each with rank 7, total 28.
    # So 4x4*4x4 has rank at most 28.
    #
    # Then 4x4*4x7: we can think of B as 4x7 = [B1 B2] where B1 is 4x3, B2 is 4x4.
    # C = A*[B1 B2] = [A*B1, A*B2]
    # A*B2 is 4x4*4x4, rank at most 28.
    # A*B1 is 4x4*4x3. 
    # 4x4*4x3: split B1 into 4x1 and 4x2.
    # 4x4*4x1: rank 16 (schoolbook).
    # 4x4*4x2: split into two 4x4*4x1, rank 32.
    # Total: 28 + 16 + 32 = 76? That's worse than 112.
    #
    # Wait, that's not right. Let me recompute.
    # 4x4*4x4: rank 28 (using Strassen on 2x2 blocks).
    # 4x4*4x3: 
    #   Split B into 4x1 and 4x2.
    #   4x4*4x1: rank 16.
    #   4x4*4x2: split into 4x4*4x1 (twice), rank 32.
    #   Total: 16+32 = 48.
    # Total for 4x4*4x7: 28 + 48 = 76.
    #
    # That's better than 112! Let me verify this.
    #
    # Actually, I need to be more careful. When I say "rank 28" for 4x4*4x4, I mean the bilinear rank.
    # The decomposition is: split A and B into 2x2 blocks, use Strassen for each pair.
    # Strassen: 7 multiplications per 2x2*2x2.
    # 4x4 has 4*4 = 16 pairs of 2x2 blocks? No.
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Using Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Each M is a 2x2*2x2 product, which has rank 4 (schoolbook) or 7? No, Strassen reduces the number of 2x2*2x2 products from 4 to 7? No, Strassen uses 7 multiplications of 2x2 matrices.
    # Wait, I'm confusing myself.
    #
    # Strassen's algorithm: to multiply two 2x2 matrices, you need 7 multiplications of 2x2 matrices? No, you need 7 multiplications of scalars if you recurse.
    #
    # Let me restart.
    #
    # For 2x2*2x2: schoolbook rank = 4. Strassen rank = 7? No, Strassen uses 7 multiplications, but each multiplication is of 2x2 matrices, which themselves have rank 4.
    # So the total rank for 2x2*2x2 using Strassen is 7*4 = 28? That's worse than schoolbook's 4.
    #
    # I think I'm making a mistake. Let me clarify.
    #
    # The bilinear rank of the 2x2*2x2 matrix multiplication tensor is 7 (Strassen).
    # This means we can decompose the tensor as a sum of 7 rank-1 tensors.
    # Each rank-1 tensor corresponds to one "multiplication" in the algorithm.
    #
    # So for 4x4*4x4, we can use the block structure:
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C11 = A11*B11 + A12*B21
    # etc.
    #
    # Each term like A11*B11 is a 2x2*2x2 product, which has rank 7.
    # But we have 8 such terms (4 blocks of C, each is a sum of 2 products).
    # Total: 8*7 = 56? That's worse than schoolbook's 64? No, 8*7=56 < 64.
    #
    # Wait, schoolbook for 4x4*4x4 is 4*4*4 = 64.
    # Using Strassen on 2x2 blocks: we have 4 block positions in C, each is a sum of 2 2x2*2x2 products.
    # So 8 2x2*2x2 products, each of rank 7, total 56.
    #
    # But we can do better by using Strassen's algorithm directly on the 4x4 matrices.
    # Strassen's algorithm for 4x4: split into 2x2 blocks, use the 7-multiplication formula.
    # The 7 multiplications are of 2x2 matrices, each of which can be done with rank 4 (schoolbook) or 7 (Strassen).
    # If we use schoolbook for the 2x2 multiplications: 7*4 = 28.
    # If we use Strassen for the 2x2 multiplications: 7*7 = 49.
    #
    # So the best is 28 for 4x4*4x4.
    #
    # Now for 4x4*4x7:
    # B is 4x7. We can split B into B1 (4x4) and B2 (4x3).
    # C = A*B = [A*B1, A*B2]
    # A*B1 is 4x4*4x4, rank 28.
    # A*B2 is 4x4*4x3.
    #
    # For 4x4*4x3:
    # Split B2 into B21 (4x1) and B22 (4x2).
    # A*B21 is 4x4*4x1, rank 16 (schoolbook).
    # A*B22 is 4x4*4x2.
    # Split B22 into B221 (4x1) and B222 (4x1).
    # A*B221 is 4x4*4x1, rank 16.
    # A*B222 is 4x4*4x1, rank 16.
    # Total for A*B22: 32.
    # Total for A*B2: 16+32 = 48.
    # Total for 4x4*4x7: 28+48 = 76.
    #
    # Let me verify this is correct.
    #
    # Actually, I realize that when I say "rank 28" for 4x4*4x4, I need to make sure the decomposition is valid.
    #
    # Let me implement this step by step.
    #
    # First, let's implement Strassen for 2x2*2x2 with rank 7.
    # Then use it to build 4x4*4x4 with rank 28.
    # Then extend to 4x4*4x7.
    #
    # Actually, let me just implement the 4x4*4x7 decomposition directly using the block approach.
    #
    # Here's my plan:
    # 1. Implement a function to get the Strassen decomposition for 2x2*2x2 (rank 7).
    # 2. Use it to build 4x4*4x4 (rank 28).
    # 3. Extend to 4x4*4x7 by splitting B into 4x4 and 4x3.
    #
    # Let's start.
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 with rank 7.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
        """
        # Strassen's 7 multiplications:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # U[i*2+j, r] for A[i,j]
        # V[j*2+k, r] for B[j,k]
        # W[k*2+i, r] for C[i,k]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # M2 = (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10
        
        # M3 = A00*(B01-B11)
        U[0][2] = 1  # A00
        V[1][2] = 1; V[3][2] = -1  # B01, -B11
        W[1][2] = 1  # C01
        
        # M4 = A11*(B10-B00)
        U[3][3] = 1  # A11
        V[2][3] = 1; V[0][3] = -1  # B10, -B00
        W[0][3] = 1; W[2][3] = 1  # C00, C10
        
        # M5 = (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        W[0][4] = -1; W[1][4] = 1  # -C00, C01
        
        # M6 = (A10-A00)*(B00+B01)
        U[2][5] = 1; U[0][5] = -1  # A10, -A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        W[3][5] = 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U[1][6] = 1; U[3][6] = -1  # A01, -A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        W[0][6] = 1  # C00
        
        return U, V, W
    
    def combine(U1, V1, W1, U2, V2, W2, offset):
        """Combine two decompositions by concatenating and offsetting the second."""
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        U = [row[:] + [0]*rank2 for row in U1]
        for r in range(rank2):
            U.append([0]*rank1 + [0]*0)  # placeholder
        # Actually, let me just build it properly.
        
        rows_U = len(U1)
        cols_U = rank
        U_new = [[0]*cols_U for _ in range(rows_U)]
        for i in range(rows_U):
            for r in range(rank1):
                U_new[i][r] = U1[i][r]
            for r in range(rank2):
                U_new[i][rank1+r] = U2[i][r]
        
        rows_V = len(V1)
        cols_V = rank
        V_new = [[0]*cols_V for _ in range(rows_V)]
        for i in range(rows_V):
            for r in range(rank1):
                V_new[i][r] = V1[i][r]
            for r in range(rank2):
                V_new[i][rank1+r] = V2[i][r]
        
        rows_W = len(W1)
        cols_W = rank
        W_new = [[0]*cols_W for _ in range(rows_W)]
        for i in range(rows_W):
            for r in range(rank1):
                W_new[i][r] = W1[i][r]
            for r in range(rank2):
                W_new[i][rank1+r] = W2[i][r]
        
        return U_new, V_new, W_new
    
    # Now, let's build 4x4*4x4 using Strassen.
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # We need 8 2x2*2x2 products.
    # Each has rank 7, total 56.
    #
    # But we can use Strassen's formula to reduce to 7 2x2*2x2 products.
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Each M is a 2x2*2x2 product, rank 7.
    # Total: 7*7 = 49? No, each M is one 2x2*2x2 product, which has rank 7.
    # So 7 such products, total rank 7*7 = 49.
    #
    # Wait, that's not right. Each M is a 2x2*2x2 product, and we're using the schoolbook decomposition for it (rank 4) or Strassen (rank 7)?
    # If we use schoolbook for the inner 2x2*2x2 products, each has rank 4, total 7*4 = 28.
    # If we use Strassen, each has rank 7, total 7*7 = 49.
    #
    # So the best is 28.
    #
    # Let me implement this.
    
    def schoolbook_2x2():
        """Return (U, V, W) for 2x2*2x2 with rank 4 (schoolbook)."""
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*2 + j*1 + k  # not quite
        # Let me just do it properly.
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][r] = 1
                    V[j*2+k][r] = 1
                    W[k*2+i][r] = 1
                    r += 1
        return U, V, W
    
    # Now, for 4x4*4x4, we need to combine 7 2x2*2x2 products.
    # But the indices are more complex because the 2x2 blocks are at different positions.
    #
    # Let me think about this differently.
    #
    # Actually, let me just implement the full 4x4*4x7 decomposition using a simple block approach.
    #
    # I'll use the following strategy:
    # 1. Split A (4x4) into 2x2 blocks: A11, A12, A21, A22.
    # 2. Split B (4x7) into 2x7 blocks: B1, B2.
    # 3. C = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    # 4. Each Ai*Bj is a 2x2*2x7 product.
    # 5. For 2x2*2x7, use schoolbook: rank 28.
    # 6. Total: 4*28 = 112. Same as schoolbook.
    #
    # That's not an improvement.
    #
    # Let me try a different approach: use the fact that 4x4*4x4 can be done in rank 28.
    #
    # Actually, I think the key insight is that we can use the Strassen decomposition for 4x4*4x4 (rank 28) and then extend it to 4x4*4x7.
    #
    # Let me implement the 4x4*4x4 decomposition first.
    
    def matmul_4x4():
        """Return (U, V, W) for 4x4*4x4 with rank 28."""
        # Use Strassen's formula with schoolbook for inner 2x2*2x2.
        U_inner, V_inner, W_inner = schoolbook_2x2()
        
        # The 7 multiplications in Strassen's formula:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For each M, we need to map the 2x2 indices to the 4x4 indices.
        # A11 is at positions (0,0), (0,1), (1,0), (1,1) in the 4x4 matrix.
        # A12 is at positions (0,2), (0,3), (1,2), (1,3).
        # A21 is at positions (2,0), (2,1), (3,0), (3,1).
        # A22 is at positions (2,2), (2,3), (3,2), (3,3).
        
        # U has shape (16, 28), V has shape (16, 28), W has shape (16, 28).
        # U[i*4+j, r] for A[i,j]
        # V[j*4+k, r] for B[j,k]
        # W[k*4+i, r] for C[i,k]
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(16)]
        
        # Helper to add a 2x2*2x2 product at a given offset.
        def add_product(U_offset, V_offset, W_offset, sign=1):
            """Add a 2x2*2x2 product at the given offsets."""
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = 0  # we'll handle the rank offset externally
            # This is getting complicated. Let me just build it directly.
        
        # Let me just build the 4x4*4x4 decomposition directly.
        #
        # Actually, let me try a different approach: just use schoolbook for now and return rank 112.
        # The task says "rank-minimal", but maybe 112 is the best we can do quickly.
        #
        # Actually, the current program already returns rank 112.
        # Let me try to improve it.
        
        # Let me try the following: use the fact that 4x4*4x1 has rank 16, and there are 7 such products.
        # Total: 7*16 = 112. Same.
        
        # Let me try to find a better decomposition.
        
        # Actually, let me just return the schoolbook decomposition for now.
        n, m, p = 4, 4, 7
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
        return U, V, W
    
    # Let me try to implement a better decomposition.
    #
    # Idea: use the block structure to reduce the rank.
    #
    # Actually, let me try the following approach:
    # 1. Use Strassen for 4x4*4x4 (rank 28).
    # 2. For the remaining 3 columns of B, use schoolbook (rank 4*4*3 = 48).
    # 3. Total: 28 + 48 = 76.
    
    def matmul_4x4_strassen():
        """Return (U, V, W) for 4x4*4x4 with rank 28."""
        n = 4
        rank = 28
        U = [[0]*rank for _ in range(n*n)]
        V = [[0]*rank for _ in range(n*n)]
        W = [[0]*rank for _ in range(n*n)]
        
        # Strassen's formula for 4x4*4x4:
        # Split A and B into 2x2 blocks.
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        # The 7 multiplications:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # Each M is a 2x2*2x2 product, done with schoolbook (rank 4).
        
        # Helper to get the 2x2 block indices.
        def block_indices(block):
            """Return (i_offset, j_offset) for the block."""
            if block == '11':
                return (0, 0)
            elif block == '12':
                return (0, 2)
            elif block == '21':
                return (2, 0)
            elif block == '22':
                return (2, 2)
        
        # For each M, we need to add 4 rank-1 terms (schoolbook for 2x2*2x2).
        # M1 = (A11+A22)*(B11+B22)
        # This is a 2x2*2x2 product where:
        # A' = A11+A22, B' = B11+B22
        # A'[i,j] = A[i,j] + A[i+2,j+2]
        # B'[i,j] = B[i,j] + B[i+2,j+2]
        # C'[i,j] = C[i,j] + C[i+2,j+2] (but we need to distribute to C11 and C22)
        
        # Actually, M1 contributes to C11 and C22.
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        
        # Let me just build the decomposition directly.
        
        # For M1 = (A11+A22)*(B11+B22):
        # A' = A11+A22, so A'[i,j] = A[i,j] + A[i+2,j+2] for i,j in {0,1}
        # B' = B11+B22, so B'[i,j] = B[i,j] + B[i+2,j+2] for i,j in {0,1}
        # C' = A'*B', which is a 2x2 matrix.
        # C' contributes to C11 (top-left 2x2) and C22 (bottom-right 2x2).
        
        # The rank-1 terms for M1:
        # For i,j,k in {0,1}:
        #   U[i*4+j, r] += 1 (for A[i,j])
        #   U[(i+2)*4+(j+2), r] += 1 (for A[i+2,j+2])
        #   V[j*4+k, r] += 1 (for B[j,k])
        #   V[(j+2)*4+(k+2), r] += 1 (for B[j+2,k+2])
        #   W[k*4+i, r] += 1 (for C[i,k] in C11)
        #   W[(k+2)*4+(i+2), r] += 1 (for C[i+2,k+2] in C22)
        
        # Wait, that's not right. The product (A11+A22)*(B11+B22) is a 2x2 matrix, and it contributes to both C11 and C22.
        # But in the decomposition, each rank-1 term should correspond to a single "multiplication".
        
        # Let me think about this more carefully.
        #
        # Actually, I think the issue is that Strassen's formula reduces the number of 2x2*2x2 multiplications from 8 to 7, but each 2x2*2x2 multiplication still has rank 4 (schoolbook).
        # So the total rank is 7*4 = 28.
        
        # Let me just implement it directly.
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(16)]
        
        # M1 = (A11+A22)*(B11+B22)
        # This is a 2x2*2x2 product.
        # The "A" matrix is A' where A'[i,j] = A[i,j] + A[i+2,j+2]
        # The "B" matrix is B' where B'[i,j] = B[i,j] + B[i+2,j+2]
        # The "C" matrix is C' where C'[i,j] = C[i,j] + C[i+2,j+2] (but we need to split it)
        
        # Actually, C' = A'*B' is a 2x2 matrix, and it contributes to C11 and C22.
        # C11 = C' (top-left 2x2 of C)
        # C22 = C' (bottom-right 2x2 of C)
        
        # So for each rank-1 term in the schoolbook decomposition of 2x2*2x2:
        # U[i*4+j, r] += 1 (for A[i,j] in A11)
        # U[(i+2)*4+(j+2), r] += 1 (for A[i+2,j+2] in A22)
        # V[j*4+k, r] += 1 (for B[j,k] in B11)
        # V[(j+2)*4+(k+2), r] += 1 (for B[j+2,k+2] in B22)
        # W[k*4+i, r] += 1 (for C[i,k] in C11)
        # W[(k+2)*4+(i+2), r] += 1 (for C[i+2,k+2] in C22)
        
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*4+j][r] = 1
                    U[(i+2)*4+(j+2)][r] = 1
                    V[j*4+k][r] = 1
                    V[(j+2)*4+(k+2)][r] = 1
                    W[k*4+i][r] = 1
                    W[(k+2)*4+(i+2)][r] = 1
                    r += 1
        
        # M2 = (A21+A22)*B11
        # A' = A21+A22, so A'[i,j] = A[i+2,j] + A[i+2,j+2]
        # B' = B11, so B'[i,j] = B[i,j]
        # C' = A'*B', which contributes to C21 (bottom-left 2x2 of C)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[(i+2)*4+j][r] = 1
                    U[(i+2)*4+(j+2)][r] = 1
                    V[j*4+k][r] = 1
                    W[(k)*4+(i+2)][r] = 1
                    r += 1
        
        # M3 = A11*(B12-B22)
        # A' = A11, so A'[i,j] = A[i,j]
        # B' = B12-B22, so B'[i,j] = B[i,j+2] - B[i+2,j+2]
        # C' = A'*B', which contributes to C12 (top-right 2x2 of C)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*4+j][r] = 1
                    V[j*4+(k+2)][r] = 1
                    V[(j+2)*4+(k+2)][r] = -1
                    W[(k+2)*4+i][r] = 1
                    r += 1
        
        # M4 = A22*(B21-B11)
        # A' = A22, so A'[i,j] = A[i+2,j+2]
        # B' = B21-B11, so B'[i,j] = B[i+2,j] - B[i,j]
        # C' = A'*B', which contributes to C11 and C21
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[(i+2)*4+(j+2)][r] = 1
                    V[(j+2)*4+k][r] = 1
                    V[j*4+k][r] = -1
                    W[k*4+i][r] = 1
                    W[k*4+(i+2)][r] = 1
                    r += 1
        
        # M5 = (A11+A12)*B22
        # A' = A11+A12, so A'[i,j] = A[i,j] + A[i,j+2]
        # B' = B22, so B'[i,j] = B[i+2,j+2]
        # C' = A'*B', which contributes to C11 and C12
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*4+j][r] = 1
                    U[i*4+(j+2)][r] = 1
                    V[(j+2)*4+(k+2)][r] = 1
                    W[k*4+i][r] = -1
                    W[(k+2)*4+i][r] = 1
                    r += 1
        
        # M6 = (A21-A11)*(B11+B12)
        # A' = A21-A11, so A'[i,j] = A[i+2,j] - A[i,j]
        # B' = B11+B12, so B'[i,j] = B[i,j] + B[i,j+2]
        # C' = A'*B', which contributes to C22
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[(i+2)*4+j][r] = 1
                    U[i*4+j][r] = -1
                    V[j*4+k][r] = 1
                    V[j*4+(k+2)][r] = 1
                    W[(k+2)*4+(i+2)][r] = 1
                    r += 1
        
        # M7 = (A12-A22)*(B21+B22)
        # A' = A12-A22, so A'[i,j] = A[i,j+2] - A[i+2,j+2]
        # B' = B21+B22, so B'[i,j] = B[i+2,j] + B[i+2,j+2]
        # C' = A'*B', which contributes to C11
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*4+(j+2)][r] = 1
                    U[(i+2)*4+(j+2)][r] = -1
                    V[(j+2)*4+k][r] = 1
                    V[(j+2)*4+(k+2)][r] = 1
                    W[k*4+i][r] = 1
                    r += 1
        
        return U, V, W
    
    # Now, for 4x4*4x7, we can use:
    # C = A*B = [A*B1, A*B2] where B1 is 4x4, B2 is 4x3.
    # A*B1 is 4x4*4x4, rank 28.
    # A*B2 is 4x4*4x3, rank 48 (schoolbook).
    # Total: 28 + 48 = 76.
    
    def matmul_4x4x7():
        """Return (U, V, W) for 4x4*4x7 with rank 76."""
        U1, V1, W1 = matmul_4x4_strassen()
        
        # For A*B2 (4x4*4x3), use schoolbook.
        n, m, p = 4, 4, 3
        rank2 = n * m * p
        U2 = [[0]*rank2 for _ in range(n*m)]
        V2 = [[0]*rank2 for _ in range(m*p)]
        W2 = [[0]*rank2 for _ in range(p*n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][r] = 1
                    V2[j*p+k][r] = 1
                    W2[k*n+i][r] = 1
                    r += 1
        
        # Now combine U1, V1, W1 and U2, V2, W2.
        # U1 has shape (16, 28), V1 has shape (16, 28), W1 has shape (16, 28).
        # U2 has shape (16, 48), V2 has shape (12, 48), W2 has shape (12, 48).
        
        rank = 28 + 48
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(28)]  # m*p = 4*7 = 28
        W = [[0]*rank for _ in range(28)]  # p*n = 7*4 = 28
        
        # Copy U1, V1, W1.
        for i in range(16):
            for r in range(28):
                U[i][r] = U1[i][r]
        
        for i in range(16):
            for r in range(28):
                V[i][r] = V1[i][r]
        
        for i in range(16):
            for r in range(28):
                W[i][r] = W1[i][r]
        
        # Copy U2, V2, W2 with offset 28.
        for i in range(16):
            for r in range(48):
                U[i][28+r] = U2[i][r]
        
        for i in range(12):
            for r in range(48):
                V[i][28+r] = V2[i][r]
        
        for i in range(12):
            for r in range(48):
                W[i][28+r] = W2[i][r]
        
        # But wait, V2 has shape (12, 48) because m*p = 4*3 = 12.
        # But V should have shape (28, 76) because m*p = 4*7 = 28.
        # The first 4 columns of B are handled by U1, V1, W1.
        # The last 3 columns of B are handled by U2, V2, W2.
        # So V2 should be placed at rows 4*0+0 to 4*3+0? No.
        #
        # Actually, V[j*4+k, r] for B[j,k].
        # For the first 4 columns (k=0..3), V is handled by V1.
        # For the last 3 columns (k=4..6), V is handled by V2.
        #
        # V2[j*3+k, r] for B[j, k+4] (k=0..2).
        # So V[j*4+(k+4), r] = V2[j*3+k, r].
        
        # Let me redo this.
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(28)]
        W = [[0]*rank for _ in range(28)]
        
        for i in range(16):
            for r in range(28):
                U[i][r] = U1[i][r]
        
        for i in range(16):
            for r in range(28):
                V[i][r] = V1[i][r]
        
        for i in range(16):
            for r in range(28):
                W[i][r] = W1[i][r]
        
        for i in range(16):
            for r in range(48):
                U[i][28+r] = U2[i][r]
        
        for j in range(4):
            for k in range(3):
                for r in range(48):
                    V[j*4+(k+4)][28+r] = V2[j*3+k][r]
        
        for k in range(3):
            for i in range(4):
                for r in range(48):
                    W[(k+4)*4+i][28+r] = W2[k*4+i][r]
        
        return U, V, W
    
    U, V, W = matmul_4x4x7()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
