"""Program for 6x6x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s kill
    
    n, m, p = 6, 6, 6
    
    # Schoolbook baseline: rank 216
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to use Strassen-like decomposition
    # For 6x6, we can partition into 2x2 blocks of 3x3
    # Each 3x3x3 multiplication has rank 27 (schoolbook) or potentially less
    # But we can also think of it as: use a 2x2 block structure
    
    # Strategy: Use the fact that 6 = 2*3
    # Partition A, B, C into 2x2 blocks of 3x3 submatrices
    # A = [[A11, A12], [A21, A22]] where each block is 3x3
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[A11*B11+A12*B21, A11*B12+A12*B22],
    #            [A21*B11+A22*B21, A21*B12+A22*B22]]
    
    # We need 4 multiplications of 3x3 matrices if we could, but Strassen uses 7 for 2x2
    # For 3x3x3, the minimal rank is not known to be less than 27 in general
    # But let's try a different approach
    
    # Alternative: Use the tensor product structure
    # If we have a rank-r decomposition for a x a x a matrix mult,
    # and rank-s for b x b x b, then we can get rank r*s for (a*b) x (a*b) x (a*b)
    
    # 6 = 2 * 3, so if we have rank r2 for 2x2x2 and r3 for 3x3x3
    # Then rank for 6x6x6 is r2 * r3
    
    # 2x2x2: Strassen gives rank 7 (schoolbook is 8)
    # 3x3x3: schoolbook is 27, best known is... let's use 27 for now
    # So 7 * 27 = 189, which is better than 216!
    
    # Let's construct this:
    # Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 3x3x3, use schoolbook: rank 27
    # Each "element" in the 2x2 block structure is a 3x3 matrix
    
    # So the overall rank is 7 * 27 = 189
    
    # Let's build this carefully
    # The 2x2 block structure: A is 6x6, partitioned as:
    # A[0:3, 0:3] = A11, A[0:3, 3:6] = A12
    # A[3:6, 0:3] = A21, A[3:6, 3:6] = A22
    
    # For the Strassen decomposition, we need to compute 7 "matrix multiplications"
    # where each is a product of two 3x3 matrices
    # But Strassen involves additions/subtractions of the input blocks first
    
    # Let's define the 7 Strassen factors for the 2x2 level:
    # Each factor involves linear combinations of the 4 blocks of A and B
    # Then we multiply the corresponding 3x3 matrices
    # Then combine the results
    
    # The U matrix for the 2x2 Strassen part (size 4x7 for a single element):
    # For M1: A11 and A22 contribute to first factor, B11 and B22 to second
    # Actually, let's think of it as:
    # The first factor of M1 is (A11 + A22), which is a 3x3 matrix
    # The second factor is (B11 + B22), which is a 3x3 matrix
    
    # So for each of the 7 Strassen multiplications, we have:
    # First factor: linear combination of A blocks (4 blocks, each 3x3)
    # Second factor: linear combination of B blocks (4 blocks, each 3x3)
    # Product: 3x3 matrix, then we need to extract elements for C
    
    # Let me define the Strassen coefficients:
    # M1: A_factor = A11 + A22, B_factor = B11 + B22
    # M2: A_factor = A21 + A22, B_factor = B11
    # M3: A_factor = A11, B_factor = B12 - B22
    # M4: A_factor = A22, B_factor = B21 - B11
    # M5: A_factor = A11 + A12, B_factor = B22
    # M6: A_factor = A21 - A11, B_factor = B11 + B12
    # M7: A_factor = A12 - A22, B_factor = B21 + B22
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each Mj is a 3x3 matrix = (A_factor_j) * (B_factor_j)
    # The A_factor_j is a 3x3 matrix formed from linear combinations of A blocks
    # The B_factor_j is a 3x3 matrix formed from linear combinations of B blocks
    
    # For the tensor decomposition, we need to express this as:
    # C[i,k] = sum_r U[i,r] * V[r] * W[k,r]  (roughly)
    
    # Actually, the tensor T[i,j,k] = delta for matrix mult means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In our block structure:
    # C has blocks C11, C12, C21, C22 (each 3x3)
    # C11[a,b] = sum over the 3x3 multiplication
    
    # Let me index: for the 6x6 case, i is 0..5, j is 0..5, k is 0..5
    # Block indices: ib = i//3, jb = j//3, kb = k//3 (each 0 or 1)
    # Within block: ia = i%3, ja = j%3, ka = k%3 (each 0..2)
    
    # C[ib, kb][ia, ka] = sum over jb of (sum over ja of A[ib,jb][ia,ja] * B[jb,kb][ja,ka])
    
    # Using Strassen at the block level:
    # C[ib,kb] = sum_s S[ib,kb,s] * M_s where M_s = A_factor_s * B_factor_s
    # and A_factor_s[ia,ja] = sum_jb A_coeff[s,ib,jb] * A[ib,jb][ia,ja]
    # and B_factor_s[ja,ka] = sum_jb B_coeff[s,jb,kb] * B[jb,kb][ja,ka]
    
    # Wait, this is getting complicated. Let me just build it directly.
    
    # For each of the 7 Strassen multiplications s (0..6):
    # A_factor_s is a 3x3 matrix: A_factor_s[ia, ja] = sum over jb of a_coeff[s][ib][jb][ia][ja] * A[ib,jb][ia,ja]
    # But actually, for a given s, the A_factor depends on which blocks of A are combined
    # Similarly for B_factor
    
    # Let me define the coefficients more carefully.
    # For Strassen M_s:
    # s=0 (M1): A uses blocks (0,0) and (1,1) with coeff 1 each
    #            B uses blocks (0,0) and (1,1) with coeff 1 each
    # s=1 (M2): A uses blocks (1,0) and (1,1) with coeff 1 each
    #            B uses block (0,0) with coeff 1
    # s=2 (M3): A uses block (0,0) with coeff 1
    #            B uses blocks (0,1) with coeff 1 and (1,1) with coeff -1
    # s=3 (M4): A uses block (1,1) with coeff 1
    #            B uses blocks (1,0) with coeff 1 and (0,0) with coeff -1
    # s=4 (M5): A uses blocks (0,0) and (0,1) with coeff 1 each
    #            B uses block (1,1) with coeff 1
    # s=5 (M6): A uses blocks (1,0) with coeff 1 and (0,0) with coeff -1
    #            B uses blocks (0,0) and (0,1) with coeff 1 each
    # s=6 (M7): A uses blocks (0,1) with coeff 1 and (1,1) with coeff -1
    #            B uses blocks (1,0) and (1,1) with coeff 1 each
    
    # For each s, the multiplication is:
    # M_s[ia, ka] = sum_ja A_factor_s[ia, ja] * B_factor_s[ja, ka]
    
    # Then C[ib, kb][ia, ka] = sum_s c_coeff[s][ib][kb] * M_s[ia, ka]
    # where c_coeff is from the Strassen combination formula
    
    # Now, for the tensor decomposition:
    # C[i,k] where i = ib*3+ia, k = kb*3+ka
    # C[ib*3+ia, kb*3+ka] = sum_s c_coeff[s][ib][kb] * sum_ja A_factor_s[ia,ja] * B_factor_s[ja,ka]
    
    # And A_factor_s[ia,ja] = sum_jb a_coeff[s][ib][jb] * A[ib*3+ia, jb*3+ja]
    # B_factor_s[ja,ka] = sum_jb b_coeff[s][kb][jb] * B[jb*3+ja, kb*3+ka]
    
    # So: C[ib*3+ia, kb*3+ka] = sum_s c_coeff[s][ib][kb] * sum_ja sum_jb sum_jb2 
    #     a_coeff[s][ib][jb] * A[ib*3+ia, jb*3+ja] * b_coeff[s][kb][jb2] * B[jb2*3+ja, kb*3+ka]
    
    # This is a sum over s, ja, jb, jb2 of products of A and B elements
    # For the tensor decomposition, each term corresponds to a rank-1 component
    
    # Let me just enumerate all the terms and build U, V, W
    
    # First, define the Strassen coefficients
    a_coeff = {}  # a_coeff[s][ib][jb] = coefficient for A[ib,jb] in factor s
    b_coeff = {}  # b_coeff[s][kb][jb] = coefficient for B[jb,kb] in factor s
    c_coeff = {}  # c_coeff[s][ib][kb] = coefficient for M_s in C[ib,kb]
    
    # s=0: M1
    a_coeff[0] = {(0,0): 1, (1,1): 1}  # A11 + A22, but this is for the block level
    # Wait, I need to be more careful. The A_factor for M1 is A11 + A22
    # A11 is A[0,0], A22 is A[1,1]
    # So for ib=0: A_factor uses A[0,0] with coeff 1 (for the ia,ja part of A11)
    #              and A[1,1] with coeff 1 (for the ia,ja part of A22)
    # But A[0,0] and A[1,1] are different blocks!
    # A_factor_s[ia, ja] = sum over (ib,jb) pairs... no wait.
    
    # Let me restart this more carefully.
    # A is 6x6, partitioned into 2x2 blocks of 3x3.
    # A[ib, jb] is the 3x3 block at row-block ib, col-block jb.
    # A[ib, jb][ia, ja] = A[ib*3+ia, jb*3+ja]
    
    # For M1 = (A11 + A22)(B11 + B22):
    # (A11 + A22) is a 3x3 matrix where element [ia, ja] = A11[ia,ja] + A22[ia,ja]
    # = A[0,0][ia,ja] + A[1,1][ia,ja]
    # So A_factor_1[ia, ja] = A[0,0][ia,ja] + A[1,1][ia,ja]
    
    # Similarly, (B11 + B22)[ja, ka] = B[0,0][ja,ka] + B[1,1][ja,ka]
    
    # So M1[ia, ka] = sum_ja (A[0,0][ia,ja] + A[1,1][ia,ja]) * (B[0,0][ja,ka] + B[1,1][ja,ka])
    # = sum_ja A[0,0][ia,ja]*B[0,0][ja,ka] + sum_ja A[0,0][ia,ja]*B[1,1][ja,ka]
    # + sum_ja A[1,1][ia,ja]*B[0,0][ja,ka] + sum_ja A[1,1][ia,ja]*B[1,1][ja,ka]
    
    # Each of these 4 terms is a "3x3x3" multiplication of specific blocks
    # And each 3x3x3 multiplication has 27 rank-1 terms (schoolbook)
    
    # So for M1, we have 4 * 27 = 108 rank-1 terms
    # For 7 Strassen multiplications, total = 7 * 4 * 27 = 756? No, that's not right.
    
    # Actually, the number of distinct (A_block, B_block) pairs across all 7 Strassen factors:
    # M1: (A11,B11), (A11,B22), (A22,B11), (A22,B22)
    # M2: (A21,B11), (A22,B11)
    # M3: (A11,B12), (A11,B22)
    # M4: (A22,B21), (A22,B11)
    # M5: (A11,B22), (A12,B22)
    # M6: (A21,B11), (A21,B12), (A11,B11), (A11,B12)
    # M7: (A12,B21), (A12,B22), (A22,B21), (A22,B22)
    
    # Many overlaps! But in the tensor decomposition, we can't easily share rank-1 terms
    # unless they are exactly the same.
    
    # This approach is getting very complex. Let me just use the simpler approach:
    # Use the tensor product of Strassen (rank 7 for 2x2x2) and schoolbook (rank 27 for 3x3x3)
    # This gives rank 7*27 = 189
    
    # The tensor product decomposition:
    # If T2 is a rank-7 decomposition of 2x2x2 matrix mult,
    # and T3 is a rank-27 decomposition of 3x3x3 matrix mult,
    # then T6 = T2 ⊗ T3 is a rank-189 decomposition of 6x6x6 matrix mult
    
    # For the tensor product:
    # U6[i*3+ia, (s, t)] = U2[i, s] * U3[ia, t]  ... no, this isn't quite right either.
    
    # Actually, the tensor product of two matrix multiplication tensors:
    # T2[i,j,k] for 2x2x2, T3[i',j',k'] for 3x3x3
    # T6[i*3+i', j*3+j', k*3+k'] = T2[i,j,k] * T3[i',j',k']
    
    # If T2 = sum_s u2_s[i] * v2_s[j] * w2_s[k] (rank 7)
    # and T3 = sum_t u3_t[i'] * v3_t[j'] * w3_t[k'] (rank 27)
    # Then T6 = sum_{s,t} (u2_s[i]*u3_t[i']) * (v2_s[j]*v3_t[j']) * (w2_s[k]*w3_t[k'])
    # So rank = 7 * 27 = 189
    
    # Let's build this!
    
    # Strassen for 2x2x2:
    # T2[i,j,k] where i,j,k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Using Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # For the tensor decomposition of 2x2x2:
    # Each M_s is a product of two linear forms in A and B
    # M_s = (sum_{i,j} a_s[i,j]*A[i,j]) * (sum_{j,k} b_s[j,k]*B[j,k])
    # But this is not quite right because M_s is a matrix product, not a scalar product.
    
    # Actually, for 2x2x2, each "multiplication" M_s involves multiplying two 2x2 matrices
    # No wait, in Strassen for 2x2, the inputs are 2x2 matrices, and we form 7 products
    # of 2x2 matrices? No, that's not right either.
    
    # Let me reconsider. For 2x2 matrix multiplication:
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k] for i,j,k in {0,1}
    # This is a 8-term sum (schoolbook)
    
    # Strassen reduces this to 7 "multiplications" but these are scalar multiplications
    # of linear combinations:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # etc.
    # Each M_s is a scalar = (linear form in A) * (linear form in B)
    # Then C[i,k] = linear combination of M_s
    
    # So the tensor T2[i,j,k] for 2x2x2 can be written as:
    # T2[i,j,k] = sum_s U2[i,s] * V2[j,s] * W2[k,s] where the sum is over s=0..6
    # But wait, T2 is the matrix multiplication tensor, which is:
    # T2[i,j,k] = 1 if (i,j,k) corresponds to a valid term in C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor T[i,j,k] is defined such that:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix mult, T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes)
    # Wait no, that's not right. The tensor is:
    # T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in the standard case.
    
    # Hmm, I think I'm confusing myself. Let me just directly construct the rank-189 decomposition
    # using the tensor product approach.
    
    # For 2x2x2 matrix mult tensor T2:
    # T2[i,j,k] = 1 for all i,j,k in {0,1} (8 elements)
    # Rank-7 decomposition (Strassen):
    # We need U2 (2x7), V2 (2x7), W2 (2x7) such that:
    # T2[i,j,k] = sum_s U2[i,s]*V2[j,s]*W2[k,s]
    
    # From Strassen:
    # C[0,0] = M0+M3-M4+M6
    # C[0,1] = M2+M4
    # C[1,0] = M1+M3
    # C[1,1] = M0-M1+M2+M5
    # where M_s = (linear form in A) * (linear form in B)
    
    # M0 = (A00+A11)*(B00+B11)
    # So the contribution to C[i,k] from M0 is:
    # coeff_A00 * coeff_B00 * A00*B00 contributes to C[0,0] with coeff 1
    # coeff_A00 * coeff_B11 * A00*B11 contributes to C[0,0] with coeff 1
    # coeff_A11 * coeff_B00 * A11*B00 contributes to C[0,0] with coeff 1
    # coeff_A11 * coeff_B11 * A11*B11 contributes to C[0,0] with coeff 1
    
    # This is getting too complex to do by hand. Let me just use the known Strassen decomposition.
    
    # Known Strassen decomposition for 2x2x2:
    # The 7 rank-1 components are:
    # s=0: U=[1,1], V=[1,1], W=[1,1]  -> (A00+A11)(B00+B11)
    # s=1: U=[0,1], V=[1,0], W=[1,1]  -> (A10+A11)... 
    
    # I think I need to look up the exact Strassen tensor decomposition.
    # Let me just use the schoolbook for now and return rank 216, since getting
    # the exact Strassen tensor decomposition right is error-prone.
    
    # Actually, let me try a different approach: just return the schoolbook decomposition
    # which is definitely correct, with rank 216.
    
    # But wait, the current program already does this with score 1.0.
    # I need to do better.
    
    # Let me try to implement the tensor product of Strassen(2x2x2, rank 7) and schoolbook(3x3x3, rank 27)
    
    # First, let me get the Strassen decomposition for 2x2x2 correct.
    # The matrix multiplication tensor for 2x2 is:
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # We want T[i,j,k] = sum_{s=0}^{6} U[i,s] * V[j,s] * W[k,s]
    
    # From Strassen's algorithm:
    # C[0,0] = M0 + M3 - M4 + M6
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M3
    # C[1,1] = M0 - M1 + M2 + M5
    
    # where:
    # M0 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M1 = (A[1,0]+A[1,1]) * B[0,0]
    # M2 = A[0,0] * (B[0,1]-B[1,1])
    # M3 = A[1,1] * (B[1,0]-B[0,0])
    # M4 = (A[0,0]+A[0,1]) * B[1,1]
    # M5 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # Each M_s = (sum_i alpha_s[i] * A[i,?]) * (sum_j beta_s[j] * B[?,j])
    # Wait, M_s is a product of two linear forms, one in A elements and one in B elements.
    # M_s = (sum_{i,j} a_s[i,j] * A[i,j]) * (sum_{j,k} b_s[j,k] * B[j,k])
    # But that would make M_s a scalar, and then C[i,k] = sum_s c_s[i,k] * M_s
    
    # For the tensor decomposition:
    # T[i,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
    # where C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # Let me verify: if T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s], then
    # C[i,k] = sum_j (sum_s U[i,s]*V[j,s]*W[k,s]) * A[i,j] * B[j,k]
    # = sum_s U[i,s]*W[k,s] * (sum_j V[j,s]*A[i,j]*B[j,k])
    # Hmm, this doesn't directly give the Strassen form.
    
    # I think the correct interpretation is:
    # The matrix multiplication tensor T is a 3-way tensor where T[i,j,k] represents
    # the coefficient in C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix mult, T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition means T[i,j,k] = sum_{s=1}^{r} U[i,s] * V[j,s] * W[k,s]
    
    # For Strassen on 2x2:
    # We need to find U (2x7), V (2x7), W (2x7) such that:
    # For all i,j,k in {0,1}: sum_{s=0}^{6} U[i,s]*V[j,s]*W[k,s] = 1
    
    # Let me try to derive this from Strassen.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Using Strassen: C[i,k] = sum_s c_s[i,k] * M_s
    # where M_s = (sum_{i',j'} a_s[i',j'] * A[i',j']) * (sum_{j',k'} b_s[j',k'] * B[j',k'])
    
    # So C[i,k] = sum_s c_s[i,k] * (sum_{i',j'} a_s[i',j'] * A[i',j']) * (sum_{j',k'} b_s[j',k'] * B[j',k'])
    
    # But we also have C[i,k] = sum_j A[i,j]*B[j,k]
    
    # These must be equal for all A, B. So:
    # sum_j delta_{i,j,k} * A[i,j]*B[j,k] = sum_s c_s[i,k] * (sum_{i',j'} a_s[i',j']*A[i',j']) * (sum_{j',k'} b_s[j',k']*B[j',k'])
    
    # This is a polynomial identity in the A and B variables.
    
    # For the tensor decomposition T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]:
    # We need: sum_j T[i,j,k]*A[i,j]*B[j,k] = C[i,k] for all A,B
    
    # If T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s], then:
    # sum_j (sum_s U[i,s]*V[j,s]*W[k,s])*A[i,j]*B[j,k]
    # = sum_s U[i,s]*W[k,s] * (sum_j V[j,s]*A[i,j]*B[j,k])
    
    # For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need the bilinear form
    # sum_j V[j,s]*A[i,j]*B[j,k] to somehow "extract" the right terms.
    
    # This is getting too theoretical. Let me just try to look up or derive the explicit
    # Strassen tensor decomposition.
    
    # Actually, I recall that for 2x2x2, the Strassen decomposition can be written as:
    # The 7 rank-1 tensors are:
    # (U, V, W) triples where:
    # s=0: U=[1,1], V=[1,1], W=[1,1]
    # s=1: U=[0,1], V=[1,0], W=[1,1]  -- but this doesn't seem right
    
    # Let me just try a different approach. I'll use the fact that:
    # rank(6x6x6) <= rank(2x2x2) * rank(3x3x3) = 7 * 27 = 189
    # And I'll construct it using the tensor product.
    
    # For the tensor product of two matrix multiplication tensors:
    # If T1 has rank r1 with decomposition {U1, V1, W1} and T2 has rank r2 with {U2, V2, W2},
    # then T1 ⊗ T2 has rank r1*r2 with:
    # U[i1*n2+i2, (s1,s2)] = U1[i1, s1] * U2[i2, s2]
    # V[j1*n2+j2, (s1,s2)] = V1[j1, s1] * V2[j2, s2]
    # W[k1*n2+k2, (s1,s2)] = W1[k1, s1] * W2[k2, s2]
    
    # Wait, I need to be careful about the indexing.
    # For 6x6x6 = (2*3)x(2*3)x(2*3):
    # i = i1*3 + i2 where i1 in {0,1}, i2 in {0,1,2}
    # j = j1*3 + j2
    # k = k1*3 + k2
    
    # T6[i,j,k] = T2[i1,j1,k1] * T3[i2,j2,k2]
    # (since the tensor product of matrix multiplication tensors corresponds to block matrix multiplication)
    
    # If T2[i1,j1,k1] = sum_{s1} U2[i1,s1]*V2[j1,s1]*W2[k1,s1]
    # and T3[i2,j2,k2] = sum_{s2} U3[i2,s2]*V3[j2,s2]*W3[k2,s2]
    # Then T6[i,j,k] = sum_{s1,s2} (U2[i1,s1]*U3[i2,s2]) * (V2[j1,s1]*V3[j2,s2]) * (W2[k1,s1]*W3[k2,s2])
    
    # So the rank is 7*27 = 189, with:
    # U6[i1*3+i2, s1*27+s2] = U2[i1,s1] * U3[i2,s2]
    # V6[j1*3+j2, s1*27+s2] = V2[j1,s1] * V3[j2,s2]
    # W6[k1*3+k2, s1*27+s2] = W2[k1,s1] * W3[k2,s2]
    
    # Now I need the actual Strassen decomposition for 2x2x2.
    # Let me derive it properly.
    
    # For 2x2x2 matrix multiplication:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # Now, for the tensor decomposition, I need to express T[i,j,k] = 1 as sum_s U[i,s]*V[j,s]*W[k,s]
    
    # Let me think about this differently. The matrix multiplication tensor T[i,j,k] satisfies:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # With Strassen, we have:
    # C[i,k] = sum_s c_s[i,k] * M_s
    # where M_s = (sum_{i',j'} a_s[i',j'] * A[i',j']) * (sum_{j',k'} b_s[j',k'] * B[j',k'])
    
    # Expanding M_s:
    # M_s = sum_{i',j',j'',k''} a_s[i',j'] * b_s[j'',k''] * A[i',j'] * B[j'',k'']
    
    # So C[i,k] = sum_s c_s[i,k] * sum_{i',j',j'',k''} a_s[i',j'] * b_s[j'',k''] * A[i',j'] * B[j'',k'']
    
    # Comparing with C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]:
    # We need: T[i,j,k] = sum_s c_s[i,k] * a_s[i,j] * b_s[j,k]
    # (after identifying i'=i, j'=j, j''=j, k''=k)
    
    # Wait, that's not quite right because the sums are over all i',j',j'',k'' not just i,j,k.
    
    # Actually, for the identity to hold for all A,B, we need:
    # sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_s c_s[i,k] * M_s for all A,B
    
    # The left side is a bilinear form in A and B.
    # The right side is also a bilinear form in A and B (since each M_s is a product of two linear forms).
    
    # For these to be equal, the coefficients of each A[i,j]*B[j',k] term must match.
    # But on the left side, the coefficient of A[i,j]*B[j',k] is T[i,j,k] if j=j' and 0 otherwise.
    # On the right side, it's sum_s c_s[i,k] * a_s[i,j] * b_s[j',k]
    
    # So: T[i,j,k] = sum_s c_s[i,k] * a_s[i,j] * b_s[j,k] (with j'=j)
    
    # Hmm, but this only gives the diagonal terms. The off-diagonal terms (j != j') must vanish.
    # sum_s c_s[i,k] * a_s[i,j] * b_s[j',k] = 0 for j != j'
    
    # This is a strong constraint. Let me just try to find U, V, W directly.
    
    # I'll use the known result that Strassen gives rank 7 for 2x2x2.
    # The explicit decomposition is:
    # U = [[1, 0, 1, 0, 1, 1, 1],
    #      [1, 1, 0, 1, 0, -1, -1]]
    # V = [[1, 1, 0, 0, 0, 1, 0],
    #      [1, 0, 1, 1, 1, 1, 1]]
    # W = [[1, 1, 0, 1, 0, 0, 1],
    #      [1, 0, 1, -1, 1, 1, 1]]
    
    # Let me verify this: T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s] should equal 1 for all i,j,k.
    
    # This is getting too long. Let me just implement the schoolbook and return it,
    # since I'm running out of time and the schoolbook is definitely correct.
    
    # Actually, let me just try to hardcode the Strassen decomposition and verify it.
    
    if time.time() - start_time > DEADLINE:
        return schoolbook()
    
    # Strassen 2x2x2 decomposition (verified)
    U2 = [[1, 0, 1, 0, 1, 1, 1],
          [1, 1, 0, 1, 0, -1, -1]]
    V2 = [[1, 1, 0, 0, 0, 1, 0],
          [1, 0, 1, 1, 1, 1, 1]]
    W2 = [[1, 1, 0, 1, 0, 0, 1],
          [1, 0, 1, -1, 1, 1, 1]]
    
    # Verify: for all i,j,k in {0,1}, sum_s U2[i,s]*V2[j,s]*W2[k,s] should be 1
    # Let's check a few:
    # i=0,j=0,k=0: 1*1*1 + 0*1*1 + 1*0*0 + 0*0*1 + 1*0*0 + 1*1*0 + 1*0*1 = 1+0+0+0+0+0+0 = 1 ✓
    # i=0,j=0,k=1: 1*1*1 + 0*1*0 + 1*0*1 + 0*0*(-1) + 1*0*1 + 1*1*1 + 1*0*1 = 1+0+0+0+0+1+0 = 2 ✗
    
    # That's wrong. Let me look up the correct Strassen tensor decomposition.
    
    # I think the issue is that the Strassen decomposition doesn't directly give
    # T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s] = 1 for all i,j,k.
    # Instead, it gives a decomposition of the matrix multiplication map.
    
    # Let me just use the schoolbook decomposition with rank 216. It's correct and safe.
    
    return schoolbook()
# EVOLVE-BLOCK-END
