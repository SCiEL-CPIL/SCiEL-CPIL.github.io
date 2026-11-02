import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # 1. Construct Schoolbook Decomposition (Rank 60)
    # This is the baseline. The tensor for matrix multiplication C = AB
    # where A is n x m and B is m x p is T_{i,j,k} = 1 if (i,j,k) corresponds
    # to C[i,k] += A[i,j] * B[j,k].
    # The standard decomposition has rank n*m*p.
    # It is known that for 2x2, rank is 7 (Strassen) < 8.
    # For 3x4 x 4x5, is rank < 60 possible?
    # The "border rank" or "tensor rank" might be lower, but we need exact decomposition.
    # The problem asks for "rank-minimal exact tensor decomposition".
    # The schoolbook rank is 60.
    # Can we do better?
    # For small matrices, Strassen-like algorithms exist.
    # However, implementing a general rank-reduction search for 3x4x5 is complex and time-consuming.
    # Given the 150s limit and the need for a guaranteed valid result,
    # and the fact that the seed program (score 1.0) returns rank 60,
    # we should first ensure we return a valid rank 60 solution quickly.
    
    # Then, we can attempt to search for a lower rank if time permits.
    # But finding a lower rank exact decomposition is a hard NP-hard problem in general.
    # For specific small sizes, known results might exist.
    # For 3x3 x 3x3, rank is 23 (Lader).
    # For 2x2 x 2x2, rank is 7.
    # For 3x4 x 4x5, the exact minimal rank is not widely known or easily computable in 100s.
    # The "score" is the rank. Lower is better.
    # If we can't find a lower rank, we return 60.
    # The seed program returns 60.
    # To beat 1.0 (seed), we need rank < 60.
    # This is a very difficult task to achieve reliably in 100s without prior knowledge of a specific algorithm.
    
    # Let's try a randomized search for a lower rank?
    # Or try to apply Strassen's algorithm recursively?
    # Strassen works for 2x2.
    # We can pad 3x4 and 4x5 to 4x4?
    # If we pad to 4x4, the rank of 4x4 x 4x4 is 49 (Strassen recursion: 7^2=49).
    # 49 < 60.
    # So, if we can embed the 3x4 x 4x5 multiplication into a 4x4 x 4x4 multiplication,
    # and then extract the result, we might achieve a rank of 49 (plus some overhead for embedding/extraction?).
    # Actually, the tensor decomposition rank is for the specific multiplication.
    # If we use a 4x4 algorithm, the rank is 49 for the 4x4 part.
    # But we need to handle the specific dimensions 3,4,5.
    # We can pad A to 4x4 (add 1 row, 0 cols? No, A is 3x4. Pad to 4x4 by adding a zero row).
    # Pad B to 4x4 (B is 4x5. Pad to 4x4 by dropping the 5th column? No, we need the result C to be 3x5.
    # If we truncate B to 4x4, we lose the 5th column of C.
    # So simple padding to 4x4 doesn't directly work for 3x4 x 4x5 -> 3x5.
    
    # Let's try padding to 5x5?
    # Rank of 5x5 is not known to be better than schoolbook 125? No, Strassen gives 7^log2(5) ~ 7^2.32 ~ 52.
    # But 5x5 rank is likely around 50-60.
    # 3x4 x 4x5 is "smaller" than 5x5 x 5x5.
    
    # Let's reconsider the 4x4 idea.
    # A is 3x4. B is 4x5.
    # We can split B into B1 (4x4) and B2 (4x1).
    # C = A * [B1, B2] = [A*B1, A*B2].
    # A*B1 is 3x4. A*B2 is 3x1.
    # We can compute A*B1 using a 4x4 algorithm (if we pad A to 4x4).
    # Pad A to 4x4: A' = [[A], [0 0 0 0]].
    # Then A'*B1 is 4x4. The top 3 rows are A*B1.
    # Rank of A'*B1 part is rank of 4x4 mult = 49.
    # A*B2 is 3x4 * 4x1.
    # This is a vector-matrix product (or matrix-vector if transposed).
    # Rank of 3x4 x 4x1 multiplication?
    # This is equivalent to computing 3 vectors of length 4 multiplied by a matrix 4x1.
    # This is 3 separate dot products of length 4.
    # Rank of dot product of length 4 is 4 (schoolbook) or 3? (Rank of 1x4 x 4x1 is 4? No, 1x1 result. T_{i,j,k} with i,k fixed. Sum_j A_{i,j} B_{j,k}. If i,k are single indices, it's just a dot product. Rank of dot product of length m is m. So rank 4.)
    # So total rank would be 49 (for 3x4 part) + 4 (for 3x1 part) = 53?
    # But wait, the decomposition must be a single tensor decomposition for the whole operation.
    # The tensor is T_{i,j,k} for i in 0..2, j in 0..3, k in 0..4.
    # If we use a block decomposition, we are effectively using a larger rank for the whole thing?
    # No, if we decompose the operation into independent sub-operations, the rank is the sum of the ranks of the sub-operations, provided they are independent.
    # Are they independent?
    # The sub-tensors for A*B1 and A*B2 have disjoint support in the k-index (k=0..3 vs k=4).
    # So yes, they are independent.
    # So we can compute rank(A*B1) + rank(A*B2).
    # rank(A*B1) where A is 3x4, B1 is 4x4.
    # Pad A to 4x4. The multiplication is 4x4 x 4x4.
    # But we only need the top 3 rows of the result.
    # Does the rank reduce if we only need a subset of outputs?
    # Generally, no. The rank of the tensor is determined by the full structure.
    # However, if the extra row of A is zero, it doesn't contribute.
    # So the tensor for 3x4 x 4x4 is a sub-tensor of 4x4 x 4x4.
    # The rank of a sub-tensor is <= rank of the full tensor.
    # So rank(3x4 x 4x4) <= 49.
    # Is it strictly less?
    # The zero row in A means that for i=3, all T_{3,j,k} = 0.
    # This might allow a lower rank.
    # But we can just use the 49-rank decomposition for 4x4 and ignore the 4th row of the result.
    # So we use 49 factors.
    # For A*B2 (3x4 x 4x1), rank is 4 (dot product).
    # Total rank = 49 + 4 = 53.
    # 53 < 60.
    # This is a valid improvement!
    
    # Let's verify this logic.
    # We need to construct the U, V, W matrices for this hybrid decomposition.
    # Part 1: 3x4 x 4x4 multiplication using Strassen (Rank 49).
    # Part 2: 3x4 x 4x1 multiplication using Schoolbook (Rank 4).
    
    # Strassen for 4x4:
    # A' is 4x4 (top 3 rows from A, bottom row 0).
    # B1 is 4x4.
    # We need U1, V1, W1 for 4x4 x 4x4 -> 4x4.
    # Then we extract the top 3 rows of the result.
    # The U, V, W matrices for the 4x4 case have dimensions:
    # U: (4*4, 49) = (16, 49)
    # V: (4*4, 49) = (16, 49)
    # W: (4*4, 49) = (16, 49)
    
    # But our A is 3x4. The indices for A are i=0..2, j=0..3.
    # In the 4x4 case, indices are i=0..3, j=0..3.
    # We can map the 3x4 indices to the 4x4 indices (i,j) -> (i,j).
    # The 4th row (i=3) of A' is zero, so it doesn't contribute to the product.
    # The result C1 is 4x4. We need C1[0..2, 0..3].
    # The W matrix for 4x4 has indices k*n + i = k*4 + i.
    # We need to map these to our output indices.
    # Our output C is 3x5. Indices i=0..2, k=0..4.
    # For Part 1 (k=0..3), the output index is i (0..2) and k (0..3).
    # The W index in 4x4 is k*4 + i.
    # We need to select the rows of W1 corresponding to i=0..2.
    # So we take W1[k*4 + i, :] for i=0..2, k=0..3.
    # This gives a matrix of size (3*4, 49) = (12, 49).
    # We need to map these to the global W matrix indices.
    # Global W index for Part 1 is k*n + i = k*3 + i.
    # So we need to permute the rows of the selected W1.
    
    # Let's construct the 4x4 Strassen decomposition.
    # Strassen's algorithm for 2x2 has rank 7.
    # For 4x4, we can use 4x4 = 2x(2x2).
    # Rank is 7^2 = 49.
    # We need the explicit U, V, W matrices for 4x4.
    # This is tedious to derive manually.
    # Alternative: Use the known 2x2 Strassen and recursively apply?
    # Or just use a library? No, we must generate it.
    # Maybe we can find a simpler decomposition?
    # What if we just use schoolbook for the 3x4 x 4x4 part? Rank = 3*4*4 = 48.
    # And schoolbook for 3x4 x 4x1 part? Rank = 3*4*1 = 12.
    # Total = 48 + 12 = 60. No improvement.
    
    # So we MUST use Strassen for the 4x4 part to get rank 49.
    # 49 + 4 = 53.
    
    # Let's implement the 4x4 Strassen decomposition.
    # Strassen 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we split A, B into 2x2 blocks of 2x2 matrices.
    # This leads to a rank 49 decomposition.
    # Constructing the explicit 16x49 matrices is complex.
    # Is there a simpler way?
    # Maybe we can use a rank 50 decomposition?
    # Or maybe the rank of 3x4 x 4x4 is actually lower than 49?
    # Since one row of A is zero, the tensor is "degenerate".
    # The rank of a tensor with a zero slice might be lower.
    # But we don't know the exact rank.
    # Given the time constraint, implementing a full 4x4 Strassen decomposition from scratch is risky and time-consuming.
    
    # Let's look for a different split.
    # Split B into 4x2 and 4x3?
    # 3x4 x 4x2 (Rank 24 schoolbook) + 3x4 x 4x3 (Rank 36 schoolbook) = 60.
    # No gain.
    
    # What if we use Strassen on the 4x4 part but don't implement the full 49-rank matrix?
    # We can try to search for a low-rank decomposition of the 3x4 x 4x4 tensor.
    # But that's hard.
    
    # Let's stick to the safe baseline of 60 for now, unless we can quickly implement a known lower rank.
    # Actually, the prompt says "score is the rank".
    # If we return 60, score is 1.0.
    # If we return 53, score is 60/53 > 1.0.
    # The challenge is generating the 53-rank decomposition.
    
    # Let's try to implement the 4x4 Strassen decomposition.
    # We can generate it programmatically using the recursive definition.
    
    def strassen_2x2():
        # Returns U, V, W for 2x2 x 2x2 -> 2x2
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: i*2+j for U, i*2+j for V, i*2+j for W (but W is k*2+i in our convention? No, W[k*n+i] for n=2. So k*2+i).
        # Wait, the convention is W[k*n+i, r].
        # For 2x2, n=2. W index is k*2 + i.
        # Strassen M1..M7.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U matrix (for A):
        # U[0,0]=1 (A00), U[1,0]=1 (A01), U[2,0]=0 (A10), U[3,0]=1 (A11) -> M1
        # U[0,1]=0, U[1,1]=0, U[2,1]=1, U[3,1]=1 -> M2
        # U[0,2]=1, U[1,2]=0, U[2,2]=0, U[3,2]=0 -> M3
        # U[0,3]=0, U[1,3]=0, U[2,3]=0, U[3,3]=1 -> M4
        # U[0,4]=1, U[1,4]=1, U[2,4]=0, U[3,4]=0 -> M5
        # U[0,5]=-1, U[1,5]=0, U[2,5]=1, U[3,5]=0 -> M6
        # U[0,6]=0, U[1,6]=1, U[2,6]=0, U[3,6]=-1 -> M7
        
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        
        # V matrix (for B):
        # V[0,0]=1 (B00), V[1,0]=0 (B01), V[2,0]=1 (B10), V[3,0]=1 (B11) -> M1
        # V[0,1]=1, V[1,1]=0, V[2,1]=0, V[3,1]=0 -> M2
        # V[0,2]=0, V[1,2]=0, V[2,2]=1, V[3,2]=-1 -> M3
        # V[0,3]=-1, V[1,3]=1, V[2,3]=0, V[3,3]=0 -> M4
        # V[0,4]=0, V[1,4]=0, V[2,4]=0, V[3,4]=1 -> M5
        # V[0,5]=1, V[1,5]=1, V[2,5]=0, V[3,5]=0 -> M6
        # V[0,6]=0, V[1,6]=0, V[2,6]=1, V[3,6]=1 -> M7
        
        V2 = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        
        # W matrix (for C):
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W index k*2+i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],   # C00
            [0, 0, 1, 0, 1, 0, 0],    # C01
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ])
        
        return U2, V2, W2

    # Now construct 4x4 using 2x2 Strassen recursively.
    # A4 is 4x4. Split into 4 blocks of 2x2: A00, A01, A10, A11.
    # B4 is 4x4. Split into 4 blocks: B00, B01, B10, B11.
    # C4 = A4 * B4.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # This is just 8 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 8 * 7 = 56.
    # 56 < 60.
    # This is simpler than full Strassen 4x4 (which is 49).
    # 56 + 4 (for the 3x4 x 4x1 part) = 60.
    # Wait. 56 + 4 = 60. No improvement.
    # We need the 49-rank Strassen for 4x4.
    
    # Let's implement the 4x4 Strassen (rank 49).
    # It uses 7 multiplications of 2x2 blocks, but the blocks are sums/differences.
    # This requires a more complex U, V, W construction.
    
    # Given the complexity and time limit, I will return the schoolbook rank 60.
    # It is safe and correct.
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
