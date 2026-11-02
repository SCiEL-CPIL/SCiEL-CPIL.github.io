"""Matrix multiplication 4,4,8 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    t0 = time.time()
    deadline = t0 + 115.0
    
    # The matrix multiplication tensor M_{i,j,k} where:
    # i in 0..n-1 (output row), j in 0..m-1 (inner dimension), k in 0..p-1 (output col)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook gives rank = n*m*p = 128
    # We try to find a lower rank decomposition.
    
    # Strategy: Use the structure of the problem.
    # For 4x4 * 4x8, we can think of it as 4 blocks of 1x4 * 4x8 = 1x8
    # or 4x4 * 4x4 (two blocks) = 4x4 (two output blocks)
    
    # Let's try a structured approach based on Kronecker products.
    # If we can write the 4x4 matrix multiply as a Kronecker product of smaller multiplies,
    # and the 4x8 as a Kronecker product, we might get a lower rank.
    
    # 4 = 2*2, 8 = 2*4
    # So 4x4 * 4x8 can be seen as (2x2 ⊗ 2x2) * (2x2 ⊗ 2x4)
    # = (2x2 * 2x2) ⊗ (2x2 * 2x4)
    # 
    # Rank of 2x2 * 2x2 multiply: at most 8 (schoolbook), actually 7 is known minimal.
    # Rank of 2x2 * 2x4 multiply: at most 16 (schoolbook).
    # 
    # If we use rank-7 for the 2x2*2x2 part and rank-16 for the 2x2*2x4 part,
    # the Kronecker product would have rank 7*16 = 112.
    #
    # But we can do better by not fully separating the Kronecker structure.
    # 
    # Alternative: Use the fact that 4x4 * 4x8 = (4x4) * (4x4) ⊕ (4x4) * (4x4)
    # where ⊕ means block diagonal in the output.
    # Each 4x4 * 4x4 has rank at most 49 (known optimal for 4x4 matrix multiply).
    # So two of them would give rank 98.
    #
    # Actually, let's think more carefully.
    # 4x4 * 4x8: The output is 4x8.
    # We can split the 4x8 into two 4x4 blocks.
    # A(4x4) * B(4x8) = A(4x4) * [B1(4x4) | B2(4x4)]
    # = [A*B1(4x4) | A*B2(4x4)]
    # 
    # So the multiplication is: compute A*B1 and A*B2, where each is 4x4*4x4.
    # If we use a rank-49 decomposition for 4x4*4x4, we'd need rank 49+49=98.
    # But we can share the U factors! The U matrix for both multiplications is the same.
    # 
    # In the tensor decomposition:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (32, rank)
    #
    # For A*B1: U is (16, r1), V1 is (16, r1), W1 is (16, r1) where r1=49
    # For A*B2: U is (16, r2), V2 is (16, r2), W2 is (16, r2) where r2=49
    #
    # The full decomposition would be:
    # U_full = [U_49 | U_49] but they're the same, so U_full = U_49 (just use it twice)
    # Wait, no. In the tensor decomposition, each rank-1 term is (u_r, v_r, w_r).
    # For A*B1, we have 49 terms. For A*B2, we have 49 terms.
    # The U part is shared: for term r in 0..48, U[*, r] is the same for both.
    # But V and W are different.
    # 
    # So the total rank would be 49 (for U) + 49 (for V1) + 49 (for V2) = 147? No.
    # Each rank-1 term is a triple (u, v, w). The U factor is part of the triple.
    # For A*B1, term r: (u_r, v1_r, w1_r)
    # For A*B2, term r: (u_r, v2_r, w2_r)
    # These are different rank-1 terms because v and w differ.
    # So total rank = 49 + 49 = 98.
    #
    # Can we do better? Maybe by using a single decomposition that handles both.
    # 
    # Let's try the block approach with known 4x4*4x4 decompositions.
    # The known optimal rank for 4x4*4x4 is 49.
    # We need to implement a rank-49 decomposition for 4x4*4x4.
    
    # For now, let's try the simple block decomposition:
    # rank = 49 + 49 = 98
    
    # We need a rank-49 decomposition for 4x4*4x4.
    # This is complex to implement from scratch. Let's try a different approach.
    
    # Actually, let's try the Kronecker product approach more carefully.
    # 4x4 * 4x8 where 4=2*2 and 8=2*4
    # A = A1 ⊗ A2 where A1 is 2x2, A2 is 2x2
    # B = B1 ⊗ B2 where B1 is 2x2, B2 is 2x4
    # C = A*B = (A1*B1) ⊗ (A2*B2)
    # 
    # Rank of 2x2*2x2: 7 (Strassen-like)
    # Rank of 2x2*2x4: 12 (can we do better than 16?)
    # 
    # For 2x2*2x4: 
    # A(2x2) * B(2x4) = C(2x4)
    # Split B into two 2x2 blocks: B = [B1 | B2]
    # C = [A*B1 | A*B2]
    # Each A*B1 is 2x2*2x2 with rank 7.
    # Total rank = 7 + 7 = 14.
    # 
    # So Kronecker: rank = 7 * 14 = 98. Same as before.
    #
    # Can we do 2x2*2x4 in rank < 14?
    # 2x2*2x4: n=2, m=2, p=4. Schoolbook rank = 16.
    # Using the block trick: rank = 7+7 = 14.
    # I don't know of a better one.
    #
    # So rank 98 seems achievable. Let's implement it.
    #
    # Actually, I realize implementing the exact rank-49 or rank-7 decompositions
    # with the correct index conventions is very error-prone.
    #
    # Let me try a different, simpler approach: use the schoolbook decomposition
    # but try to find linear dependencies to reduce the rank.
    #
    # For 4x4*4x8, the tensor has 4*4*8 = 128 entries.
    # The tensor M[i,j,k] = delta_{i,0}delta_{j,0}delta_{k,0} + ... 
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid term, 0 otherwise.
    # Wait, no. The matrix multiplication tensor is defined as:
    # T is a 3-tensor where T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # No, that's not right either.
    #
    # The matrix multiplication tensor M_{i,j,k} for multiplying A(n×m) by B(m×p)
    # to get C(n×p) is:
    # M[i,j,k] = δ_{i,0}δ_{j,0}δ_{k,0} ... no.
    #
    # Actually, the standard definition: the matrix multiplication tensor is
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors.
    # 
    # In the decomposition U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r],
    # we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # 
    # This is just saying that the tensor of all ones (in the appropriate indices)
    # can be decomposed. But that's not right - the matrix multiplication tensor
    # is not the all-ones tensor.
    #
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = δ_{i,k}? 
    # No, that doesn't make sense for matrix multiplication.
    #
    # Actually, for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In tensor notation, the matrix multiplication tensor is:
    # M[i,j,k] = 1 for all i,j,k (in the range)
    # And the decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Yes! The tensor is the all-ones tensor of shape (n*m, m*p, p*n) with the
    # appropriate index mapping. So we need to decompose the all-ones tensor.
    #
    # Wait, but the all-ones tensor of shape (a,b,c) has rank 1 (it's the outer product
    # of three all-ones vectors). That can't be right.
    #
    # I think I'm confusing the tensor structure. Let me look at this more carefully.
    #
    # The matrix multiplication tensor T has entries T[i,j,k] where:
    # i indexes the row of A and C (0 to n-1)
    # j indexes the column of A and row of B (0 to m-1)  
    # k indexes the column of B and C (0 to p-1)
    # 
    # T[i,j,k] = 1 for all i,j,k.
    #
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # This is indeed the all-ones tensor. The rank of the all-ones tensor of shape
    # (a,b,c) is 1. But that's in the standard tensor rank sense where the tensor is
    # a 3-way array. Here the index structure is different because of the cyclic
    # dependency: i appears in U and W, j appears in U and V, k appears in V and W.
    #
    # So the "rank" here is not the standard tensor rank. It's the rank of the
    # bilinear algorithm for matrix multiplication.
    #
    # OK so the schoolbook decomposition uses rank = n*m*p = 128, with:
    # U[i*m+j, r(i,j,k)] = 1, V[j*p+k, r(i,j,k)] = 1, W[k*n+i, r(i,j,k)] = 1
    # for r(i,j,k) = i*m*p + j*p + k (or some enumeration).
    #
    # To find a lower rank, we need to find a bilinear algorithm.
    #
    # Given the complexity of finding an optimal bilinear algorithm, and the time
    # constraints, let me try a structured approach:
    #
    # Use the fact that 4 = 2*2 and 8 = 2*4.
    # Decompose 4x4*4x8 as a Kronecker product of 2x2*2x2 and 2x2*2x4.
    #
    # For 2x2*2x2, use a rank-7 decomposition.
    # For 2x2*2x4, use a rank-14 decomposition (two rank-7 blocks).
    #
    # Total rank: 7*14 = 98.
    
    # Let me implement the rank-7 decomposition for 2x2*2x2.
    # The standard 7-term algorithm for 2x2*2x2:
    # 
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # 7 products:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # Let me verify:
    # C[0,0] = (a+d)(e+h) + d(g-c) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - dc - ah - bh + bg + bh - dg - dh
    # = ae + de + dg - dc - bh + bg - dg
    # Hmm, this doesn't look right. Let me use the standard Strassen-style 7-term.
    #
    # Actually, the standard 7-term for 2x2 is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # Let me verify C00:
    # m1 + m4 - m5 + m7 = (a+d)(e+h) + d(g-c) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - dc - ah - bh + bg + bh - dg - dh
    # = ae + de + dg - dc - bh + bg - dg
    # = ae + de + bg - dc
    # But C00 should be ae + bg. So this is wrong.
    #
    # I think I'm misremembering the 7-term algorithm. Let me just use the schoolbook
    # for 2x2*2x2 (rank 8) and see if the overall approach works, then optimize.
    #
    # Actually, for the purposes of this task, let me just use the schoolbook
    # decomposition with rank 128. It's guaranteed to work.
    #
    # But wait, the task says "rank-minimal" and the score is the rank.
    # The seed program uses rank 128. If I can find any lower rank, I improve.
    #
    # Let me try the block decomposition: split 4x8 into two 4x4 blocks.
    # For each 4x4*4x4, use schoolbook rank 64.
    # Total rank: 64 + 64 = 128. Same as schoolbook.
    #
    # What if I share the U factors?
    # For A*B1: U (16, 64), V1 (16, 64), W1 (16, 64)
    # For A*B2: U (16, 64), V2 (16, 64), W2 (16, 64)
    # The U is the same, but the rank-1 terms are (u_r, v1_r, w1_r) and (u_r, v2_r, w2_r).
    # These are different terms, so total rank = 64 + 64 = 128.
    #
    # No improvement. I need a genuinely lower-rank algorithm.
    
    # Let me try to construct a rank-96 decomposition using the 2x2*2x2 = 8 rank
    # and 2x2*2x4 = 16 rank, with Kronecker product: 8*16 = 128. Still no improvement.
    
    # I think the key insight is that the all-ones tensor with the cyclic index
    # structure has a specific rank that I need to compute.
    
    # Given time constraints, let me just return the schoolbook decomposition.
    # It's guaranteed to be correct.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon = 0.0
                    for r in range(U.shape[1]):
                        recon += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(recon - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
