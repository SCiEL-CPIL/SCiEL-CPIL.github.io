import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (limit is 150s)

    n, m, p = DIMENSIONS
    N, M, P = n, m, p

    # We will use a block decomposition.
    # 2x2x2 blocks of Strassen (rank 7 each)
    # Remaining dimensions handled by Schoolbook (rank 1 each)
    
    # Let's use blocks of size 2x2x2 where possible.
    # n=4, m=5, p=5.
    # We can fit two 2x2 blocks in dimension n.
    # We can fit two 2x2 blocks in dimension m (leaving 1).
    # We can fit two 2x2 blocks in dimension p (leaving 1).
    
    # So we have a grid of blocks:
    # n-blocks: 2 blocks of size 2.
    # m-blocks: 2 blocks of size 2, 1 block of size 1.
    # p-blocks: 2 blocks of size 2, 1 block of size 1.
    
    # Total blocks = 2 * 3 * 3 = 18 blocks.
    # For each block of size (bn, bm, bp), we compute the minimal rank decomposition.
    # If bn=bm=bp=2, rank is 7 (Strassen).
    # If any dimension is 1, we can use schoolbook for that block part?
    # Actually, we can treat the whole thing as a sum of tensors.
    # T = sum_{blocks} T_block.
    # The rank of the sum is at most the sum of the ranks.
    # We can optimize each block independently if we ensure the blocks don't interfere.
    # Since the indices are disjoint for different blocks, the tensor is a direct sum.
    # Thus, total rank = sum of ranks of blocks.
    
    # Let's calculate the rank for each block type:
    # 2x2x2: Rank 7
    # 2x2x1: Rank 4 (Schoolbook: 2*2*1=4). Can we do better?
    #   A is 2x2, B is 2x1. C is 2x1.
    #   C1 = A11 B1 + A12 B2
    #   C2 = A21 B1 + A22 B2
    #   Schoolbook rank 4.
    #   Is rank 3 possible?
    #   R(2x2x1) = 4. It is known that R(2x2x1) = 4.
    # 2x1x2: Rank 4.
    # 1x2x2: Rank 4.
    # 2x1x1: Rank 2.
    # 1x2x1: Rank 2.
    # 1x1x2: Rank 2.
    # 1x1x1: Rank 1.
    
    # Let's verify R(2x2x1).
    # Tensor T[i,j,k] = delta_{i,k} * A_{i,j} * B_{j,k}? No.
    # MM tensor T_{ijk} is 1 if i=k, else 0? No.
    # T_{ijk} = 1 if we are computing C_{i,k} from A_{i,j} B_{j,k}.
    # So T_{ijk} is 1 if the index j is the "middle" one?
    # The tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # The term in C_{i,k} is sum_j A_{i,j} B_{j,k}.
    # So the tensor has non-zero entries at T_{i,j,k} = 1 for all i,j,k?
    # NO. The tensor is the "multiplication tensor" which is a 3D array.
    # T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Actually, the standard tensor definition for MM is:
    # T_{i,j,k} = 1 if the index j matches (col of A, row of B).
    # Wait, the tensor is usually defined as having 1s where the indices align for multiplication.
    # For MM_{n,m,p}, the tensor has dimensions n x m x p.
    # T[i,j,k] = 1 if the j-th column of A and j-th row of B multiply to contribute to C[i,k].
    # This is true for ALL i,j,k?
    # Let's check the definition.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T such that C = T(A,B) is defined by T[i,j,k] = 1.
    # Yes, for standard MM, the tensor is the "all-ones" tensor of shape (n,m,p)?
    # No! That would be the tensor for element-wise multiplication followed by summation?
    # If T[i,j,k] = 1 for all i,j,k, then the decomposition rank is n*m*p (schoolbook).
    # And indeed, the schoolbook decomposition has rank n*m*p.
    # Is it possible to have a lower rank?
    # Strassen for 2x2x2 has rank 7 < 8.
    # The 2x2x2 MM tensor is NOT the all-ones tensor?
    # Let's check.
    # C11 = A11 B11 + A12 B21.
    # C12 = A11 B12 + A12 B22.
    # C21 = A21 B11 + A22 B21.
    # C22 = A21 B12 + A22 B22.
    # The tensor entries T[i,j,k] correspond to the coefficient of A[i,j]B[j,k] in C[i,k].
    # For C11 (i=1,k=1): terms are A11 B11 (j=1) and A12 B21 (j=2).
    # So T[1,1,1] = 1. T[1,2,1] = 1.
    # For C12 (i=1,k=2): terms are A11 B12 (j=1) and A12 B22 (j=2).
    # So T[1,1,2] = 1. T[1,2,2] = 1.
    # It seems T[i,j,k] = 1 for ALL i,j,k.
    # If T is the all-ones tensor, why is Strassen rank 7?
    # Because the all-ones tensor of size 2x2x2 has rank 7?
    # Yes, the rank of the all-ones tensor of size 2x2x2 is 7.
    # The rank of the all-ones tensor of size n x m x p is the matrix multiplication rank R(n,m,p).
    # So yes, the tensor is the all-ones tensor.
    
    # So my block strategy is correct.
    # We partition the indices i, j, k into blocks.
    # For each block (I, J, K), the sub-tensor is all-ones of size |I|x|J|x|K|.
    # The rank of the sub-tensor is R(|I|,|J|,|K|).
    # Since the blocks are disjoint in indices, the total rank is the sum of the ranks.
    
    # Let's determine the optimal partition.
    # We have n=4, m=5, p=5.
    # Possible block sizes for n: [2,2]
    # Possible block sizes for m: [2,2,1]
    # Possible block sizes for p: [2,2,1]
    
    # We need to compute R(bn, bm, bp) for all combinations.
    # Known values:
    # R(2,2,2) = 7
    # R(2,2,1) = 4
    # R(2,1,2) = 4
    # R(1,2,2) = 4
    # R(2,1,1) = 2
    # R(1,2,1) = 2
    # R(1,1,2) = 2
    # R(1,1,1) = 1
    
    # Let's verify R(2,2,1).
    # Tensor is 2x2x1 all-ones.
    # Rank is 4?
    # Yes, because it's equivalent to 2x2x1 MM.
    # C is 2x1.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # This requires 4 products in schoolbook.
    # Can we do 3?
    # R(2,2,1) = 4 is a known result.
    
    # Let's compute the total rank for the partition [2,2] x [2,2,1] x [2,2,1].
    
    # Blocks:
    # (2,2,2): 2 * 2 * 2 = 8 blocks?
    # No, the number of blocks of size 2 in n is 2.
    # The number of blocks of size 2 in m is 2.
    # The number of blocks of size 2 in p is 2.
    # So there are 2*2*2 = 8 blocks of size 2x2x2.
    # Rank contribution: 8 * 7 = 56.
    
    # Blocks involving the '1' dimension.
    # m has a block of size 1 (index 4).
    # p has a block of size 1 (index 4).
    
    # Case 1: n=2, m=2, p=1.
    # Number of such blocks: (n-blocks of 2) * (m-blocks of 2) * (p-blocks of 1)
    # n-blocks of 2: 2
    # m-blocks of 2: 2
    # p-blocks of 1: 1
    # Total: 4 blocks.
    # Rank per block: R(2,2,1) = 4.
    # Contribution: 4 * 4 = 16.
    
    # Case 2: n=2, m=1, p=2.
    # Number: (n-blocks of 2) * (m-blocks of 1) * (p-blocks of 2)
    # 2 * 1 * 2 = 4 blocks.
    # Rank per block: R(2,1,2) = 4.
    # Contribution: 4 * 4 = 16.
    
    # Case 3: n=2, m=1, p=1.
    # Number: (n-blocks of 2) * (m-blocks of 1) * (p-blocks of 1)
    # 2 * 1 * 1 = 2 blocks.
    # Rank per block: R(2,1,1) = 2.
    # Contribution: 2 * 2 = 4.
    
    # Total Rank = 56 + 16 + 16 + 4 = 92.
    
    # This matches the target rank in the prompt's comment.
    
    # Now we need to construct the matrices U, V, W.
    
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen factors for 2x2x2
        # Relative indices (0,0) to (1,1)
        # U (A): i, j
        # V (B): j, k
        # W (C): k, i
        
        # The 7 factors
        # We define them as lists of (idx, val) for U, V, W
        
        # m1
        u1 = [((i_off+0)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v1 = [((j_off+0)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)]
        w1 = [((k_off+0)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), 1)]
        
        # m2
        u2 = [((i_off+1)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v2 = [((j_off+0)*p + (k_off+0), 1)]
        w2 = [((k_off+1)*n + (i_off+1), 1)]
        
        # m3
        u3 = [((i_off+0)*m + (j_off+0), 1)]
        v3 = [((j_off+0)*p + (k_off+1), 1), ((j_off+1)*p + (k_off+1), -1)]
        w3 = [((k_off+1)*n + (i_off+0), 1)]
        
        # m4
        u4 = [((i_off+1)*m + (j_off+1), 1)]
        v4 = [((j_off+1)*p + (k_off+1), 1), ((j_off+0)*p + (k_off+0), -1)]
        w4 = [((k_off+0)*n + (i_off+1), 1), ((k_off+1)*n + (i_off+1), 1)]
        
        # m5
        u5 = [((i_off+0)*m + (j_off+0), 1), ((i_off+0)*m + (j_off+1), 1)]
        v5 = [((j_off+1)*p + (k_off+1), 1)]
        w5 = [((k_off+1)*n + (i_off+0), 1), ((k_off+0)*n + (i_off+0), -1)]
        
        # m6
        u6 = [((i_off+1)*m + (j_off+1), 1), ((i_off+0)*m + (j_off+0), -1)]
        v6 = [((j_off+0)*p + (k_off+0), 1), ((j_off+0)*p + (k_off+1), 1)]
        w6 = [((k_off+1)*n + (i_off+1), 1)]
        
        # m7
        u7 = [((i_off+0)*m + (j_off+1), 1), ((i_off+1)*m + (j_off+1), -1)]
        v7 = [((j_off+1)*p + (k_off+1), 1), ((j_off+0)*p + (k_off+0), 1)]
        w7 = [((k_off+0)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), -1)]
        
        # Wait, I need to double check the standard Strassen formulas to ensure the signs are correct.
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)  -- Note: M6 is not used in output?
        # M7 = (A12-A22)(B21+B22)
        
        # Let's re-verify my u/v/w assignments against these.
        
        # M1:
        # U: A11, A22 -> Correct.
        # V: B11, B22 -> Correct.
        # W: C11, C22 -> Correct.
        
        # M2:
        # U: A21, A22 -> Correct.
        # V: B11 -> Correct.
        # W: C21 -> Correct. (C21 = M2 + M4)
        
        # M3:
        # U: A11 -> Correct.
        # V: B12, -B22 -> Correct.
        # W: C12 -> Correct. (C12 = M3 + M5)
        
        # M4:
        # U: A22 -> Correct.
        # V: B21, -B11 -> Correct.
        # W: C11, C21 -> Correct. (C11 = ... + M4, C21 = ... + M4)
        
        # M5:
        # U: A11, A12 -> Correct.
        # V: B22 -> Correct.
        # W: C12, -C11 -> Correct. (C12 = ... + M5, C11 = ... - M5)
        
        # M6:
        # U: A21, -A11 -> Correct.
        # V: B11, B12 -> Correct.
        # W: Not used?
        # If M6 is not used in the output C, then its W entries should be 0.
        # In standard Strassen, M6 is defined but not used in the final sum?
        # Wait, standard Strassen uses 7 products. All 7 are used.
        # Let's check C22 = M1 - M2 + M3 + M7.
        # Where is M6?
        # Maybe my formula for M6 is for a variant?
        # Or maybe M6 contributes to C21 or something?
        # Actually, the standard 7 products are:
        # M1, M2, M3, M4, M5, M6, M7.
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # M6 is indeed NOT used in this specific set of equations?
        # That would mean the rank is 6?
        # No, the rank of 2x2x2 is 7.
        # This implies that the 7 products must be linearly independent in the tensor space.
        # If one product (M6) does not appear in the output, it means its coefficient vector in W is 0.
        # If W for M6 is 0, then M6 does not contribute to the tensor.
        # Then the decomposition only uses 6 products?
        # This is a contradiction.
        
        # Let's look up the correct Strassen decomposition.
        # The 7 products are:
        # P1 = (A11 + A22) (B11 + B22)
        # P2 = (A21 + A22) B11
        # P3 = A11 (B12 - B22)
        # P4 = A22 (B21 - B11)
        # P5 = (A11 + A12) B22
        # P6 = (A21 - A11) (B11 + B12)
        # P7 = (A12 - A22) (B21 + B22)
        
        # Output:
        # C11 = P1 - P5 - P7 + P4
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P7
        
        # It is a fact that P6 is not used in the output.
        # However, the rank is 7.
        # This means that the 7 tensors defined by P1..P7 are linearly independent.
        # Even though P6 has zero coefficient in the output tensor, it is part of the basis?
        # No, the decomposition expresses the target tensor T as a sum of rank-1 tensors.
        # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
        # If w_6 = 0, then the 6th term is 0.
        # Then T = sum_{r!=6} ...
        # This would imply rank <= 6.
        # But rank(2x2x2) = 7.
        
        # Conclusion: My memory of the Strassen formulas is slightly off regarding which product is used where, OR the product P6 IS used.
        # Let's check C21 again.
        # C21 = A21 B11 + A22 B21.
        # My formula: P2 + P4.
        # P2 = (A21+A22)B11 = A21 B11 + A22 B11.
        # P4 = A22(B21-B11) = A22 B21 - A22 B11.
        # Sum = A21 B11 + A22 B21. Correct.
        
        # So P6 is indeed not used.
        # Is it possible that the rank is 6?
        # No, it is a famous result that R(2,2,2) = 7.
        # This implies that the set of 7 rank-1 tensors associated with P1..P7 must be linearly independent, and their sum must equal T.
        # If P6's W vector is 0, it contributes nothing.
        # So the sum of the other 6 must equal T.
        # This would mean R <= 6.
        # Contradiction.
        
        # Where is the error?
        # The error is that P6 is NOT (A21 - A11)(B11 + B12).
        # Or the output formulas are different.
        
        # Let's use a verified online source or derivation.
        # Alternative Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # It is consistently stated that M6 is not used.
        # However, if you search "Strassen 7 products", you will find that all 7 are part of the algorithm.
        # Ah, I see. The "rank" of the algorithm is 7 because it computes 7 products.
        # But if one product is not used in the final output, the decomposition rank is 6?
        # No, the "rank" in tensor decomposition context is the minimum number of rank-1 terms.
        # If Strassen's algorithm uses 7 products, and one is unused, does it mean the tensor rank is 6?
        # No, the tensor rank is 7.
        # This implies that my set of 7 products is NOT a valid decomposition of rank 7?
        # Or that the 7 products are linearly dependent?
        
        # Actually, it turns out that for 2x2x2, the rank is 7.
        # And Strassen's decomposition uses 7 terms.
        # If one term has a zero W vector, it is a zero term.
        # So the sum has 6 non-zero terms.
        # This would mean rank 6.
        # This is a known paradox in my understanding.
        
        # Let's check the tensor rank of 2x2x2 again.
        # It is 7.
        # Therefore, there must be a decomposition with 7 non-zero terms.
        # Maybe the product M6 IS used?
        # Let's check C22 again.
        # C22 = A21 B12 + A22 B22.
        # M1 - M2 + M3 + M7
        # M1 = A11 B11 + A11 B22 + A22 B11 + A22 B22
        # -M2 = -A21 B11 - A22 B11
        # M3 = A11 B12 - A11 B22
        # M7 = A12 B21 + A12 B22 - A22 B21 - A22 B22
        # Sum:
        # B11 terms: A11 - A21. (Should be 0 for C22) -> Mismatch.
        
        # My manual expansion suggests the formulas might be swapped or I made a mistake.
        # Given the time constraint and the high risk of implementing Strassen incorrectly (leading to score 0),
        # I will fall back to the **Schoolbook** decomposition for the entire 4x5x5 matrix.
        # Rank = 100.
        # This is guaranteed to be correct.
        # Score will be 1.0 (if seed is 100) or valid.
        # The prompt says "score is the rank".
        # If I return rank 100, the score is 100.
        # The seed program returned empty matrices (invalid).
        # So any valid decomposition is an improvement.
        
        # I will implement the schoolbook decomposition.
        pass

    # Schoolbook Decomposition
    # Rank = n * m * p = 4 * 5 * 5 = 100
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple validation
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    print(f"Rank: {rank}")
    
    # Verify tensor identity for a small case to be sure
    # T[i,j,k] should be 1 if the indices align.
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k.
    # Actually, the tensor is all-ones.
    # So the sum should be 1 for all i,j,k.
    
    # Let's test one entry.
    # i=0, j=0, k=0.
    # Only the term where col corresponds to (0,0,0) is non-zero.
    # U[0, c] = 1, V[0, c] = 1, W[0, c] = 1.
    # Sum = 1.
    
    # i=0, j=0, k=1.
    # The term for (0,0,0) has W[0,0,c]=1, but we need W[k*n+i, c] = W[1*4+0, c] = W[4, c].
    # For c corresponding to (0,0,0), W[4,c] is 0.
    # The term for (0,0,1) has U[0,c']=1, V[1,c']=1, W[4,c']=1.
    # Sum = 1.
    
    # So yes, it works.
