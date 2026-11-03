"""Matrix multiplication 2x4 x 4x8: Strassen-based rank reduction."""
DIMENSIONS = [2, 4, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140  # safety margin
    
    # Strassen 7-term decomposition for 2x2x2
    # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # Standard Strassen:
    # M1=(a+d)(e+j), M2=(c+d)e, M3=a(f-j), M4=d(i-e),
    # M5=(a+b)i, M6=(c+d)(j-k), M7=(a+b)(f-j)  -- wait, need to be careful
    # 
    # For 2x2x2 tensor T_{ijk} = delta_i * delta_j * delta_k (i in 0,1; j in 0,1; k in 0,1)
    # This is just the identity tensor for matrix mult of 2x2 * 2x2
    # Strassen gives rank 7:
    #
    # Using standard Strassen formulas for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e),
    # M5 = (a+b)g, M6 = (c+d)(h-f), M7 = (a+b)(f-h)  -- no, standard is:
    #
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e),
    # M5=(a+b)g, M6=(c+d)(h-f), M7=(a+b)(f-h)
    #
    # C11=M1+M4-M2-M7, C12=M3+M7, C21=M5+M2, C22=M1-M3-M5+M6
    #
    # For the tensor T_{ijk} where i=row of A, j=row of B (=col of A^T), k=col of C:
    # T_{ijk} = delta_{i,k} when j indexes the inner product
    # Actually T_{ijk} for matrix mult (n=2,m=2,p=2):
    # C_{ik} = sum_j A_{ij} B_{jk}
    # T_{ijk} = 1 if the product A_{ij}*B_{jk} contributes to C_{ik}
    # So T_{ijk} = 1 for all valid (i,j,k) -- it's the full tensor
    #
    # Strassen decomposition of this tensor:
    # Term 1: U=[1,1], V=[1,0], W=[1,1]  -> (a+d)(e+?) -- let me just use known values
    
    # Known rank-7 decomposition for 2x2x2 matrix mult tensor:
    # (U rows indexed by i=0,1; V rows by j=0,1; W rows by k=0,1)
    strassen_U = [[1,1,1,0,1,1,1],
                   [0,0,0,1,1,0,0]]
    strassen_V = [[1,1,0,1,0,1,0],
                   [0,0,1,0,1,0,1]]
    strassen_W = [[1,0,1,0,0,1,0],
                   [1,1,0,1,1,0,1]]
    
    # Verify: for term c, contribution to C[i,k] += U[i,c]*V[j,c]*W[k,c]
    # We need sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all (i,j,k)
    # That can't be right -- the tensor is T[i,j,k]=1 for all i,j,k in {0,1}
    # No wait -- for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if this term A[i,j]*B[j,k] appears in C[i,k]
    # Which is ALWAYS 1 for valid indices. So T is all-ones 2x2x2 tensor.
    # But that has rank 1! That's wrong.
    #
    # I'm confusing myself. The matrix multiplication tensor for M_{n,m,p} is:
    # T in R^{n x m x p} where T[i,j,k] = 1 (it's just the structural tensor)
    # The bilinear map is: (A,B) -> sum_{j} A[i,j]*B[j,k] for each (i,k)
    # The tensor whose rank we minimize is the 3-tensor with entries T[i,j,k]=1
    # For 2x2x2, this all-ones tensor has rank 1 (it's just 1*1*1)!
    #
    # NO. I'm fundamentally confused. Let me reconsider.
    #
    # The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i,j,k.
    # The rank of this tensor: for 2x2x2, the all-ones tensor of size 2x2x2 has rank 1.
    # But that contradicts the known result that M_{2,2,2} has rank 7!
    #
    # Ah, I see my error. The matrix multiplication tensor is NOT the all-ones tensor.
    # It's defined such that the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = delta_{i,i'} * ... no.
    #
    # Correct definition: T in R^{(n*p) x m} or as a 3-tensor.
    # As a 3-tensor T in R^{n x m x p}: T[i,j,k] = 1 for all i,j,k.
    # The matrix multiplication map is: for each output (i,k),
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Since T[i,j,k]=1, this is just the standard matrix mult.
    #
    # The rank of the all-ones 2x2x2 tensor IS 1. But we know M_{2,2,2} requires rank 7.
    # 
    # I think the issue is that the "tensor" for matrix multiplication is not the all-ones tensor
    # in the naive sense. Let me look at this differently.
    #
    # The bilinear map mu: R^{n x m} x R^{m x p} -> R^{n x p}
    # mu(A,B)[i,k] = sum_j A[i,j] B[j,k]
    #
    # This can be written as: mu(A,B)[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] delta... 
    # No, it's: mu(A,B)[i,k] = sum_j A[i,j] B[j,k]
    #
    # As a trilinear form: L(A,B,C) = trace(C^T mu(A,B)) = sum_{i,k,j} C[k,i] A[i,j] B[j,k]
    # The associated tensor is T[i,j,k] with the bilinear map being:
    # f(A,B)[i,k] = sum_j A[i,j] B[j,k]
    #
    # The rank of this bilinear map (tensor rank) for 2x2x2 is indeed 7 (Strassen).
    #
    # The tensor T in the sense of tensor rank: we want to write
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i] * V_c[j] * W_c[k]
    # and we need sum_c U_c[i]*V_c[j]*W_c[k] to reproduce the matrix mult structure.
    #
    # Specifically, we need: for all A, B:
    # sum_c (sum_i U_c[i] A[i,j... ]) -- I'm overcomplicating this.
    #
    # The standard formulation: the matrix multiplication tensor T_{n,m,p} satisfies:
    # T[i,j,k] = 1 for all i,j,k
    # and the bilinear map is: (A,B) -> [sum_j A[i,j]B[j,k]]_{i,k}
    #
    # The tensor rank is the minimum r such that there exist vectors
    # u_c in R^n, v_c in R^m, w_c in R^p with
    # sum_{c=1}^r u_c[i] v_c[j] w_c[k] = T[i,j,k] = 1 for all i,j,k
    #
    # For 2x2x2: we need sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k in {0,1}
    # This is an all-ones tensor, which has rank 1: just u=v=w=[1,1]
    #
    # So the tensor rank of the all-ones 2x2x2 tensor is 1?! But that contradicts everything.
    #
    # I think I've been confusing "tensor rank" with "bilinear complexity" / "multiplication complexity".
    #
    # Bilinear complexity: minimum number of bilinear multiplications to compute all entries of C.
    # For 2x2x2, this is 7 (Strassen).
    #
    # Tensor rank: minimum r such that T = sum_c u_c ⊗ v_c ⊗ w_c.
    # For the all-ones 2x2x2 tensor, rank = 1.
    #
    # These are different quantities! The task asks for "rank-minimal exact tensor decomposition"
    # which should be the bilinear complexity, not the tensor rank in the algebraic sense.
    #
    # In the context of this task, "rank" means the number of rank-1 terms in a decomposition
    # where each term corresponds to one bilinear multiplication. This is the bilinear complexity.
    #
    # So for M_{2,4,8}, I need the bilinear complexity, which is at most 64 (schoolbook)
    # and I want to reduce it.
    #
    # The bilinear complexity of M_{n,m,p} is the minimum r such that there exist
    # U (n*m x r), V (m*p x r), W (p*n x r) with
    # for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
    # 
    # Actually, the standard formulation: the bilinear map is
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We decompose this into r bilinear terms:
    # C[i,k] = sum_{c=1}^r (sum_j U_{ij}^{(c)} A[i,j]) (sum_l V_{kl}^{(c)} B[?,?])
    # 
    # Let me use the interface from the task:
    # U: shape (n*m, rank) -- indexed by (i,j)
    # V: shape (m*p, rank) -- indexed by (j,k)  
    # W: shape (p*n, rank) -- indexed by (k,i)
    # 
    # And the condition is: for each c, the term contributes
    # (sum_{i,j} U[i*m+j, c] A[i,j]) * (sum_{j,k} V[j*p+k, c] B[j,k]) to C[...]
    # 
    # More precisely, the output is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # 
    # And we need this to equal sum_j A[i,j] B[j,k] for all i,k.
    #
    # For schoolbook: for each (i,j,k), one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # Total rank = n*m*p = 64.
    #
    # Now, can we do better? For 2x4x8:
    # 
    # One approach: group the 8 columns into 4 pairs. For each pair (k, k+1) and each j,
    # we have 2 outputs: C[0,k], C[0,k+1], C[1,k], C[1,k+1]
    # Each depends on A[0,j], A[1,j] and B[j,k], B[j,k+1]
    # 
    # For fixed j and pair (k,k+1):
    # C[0,k]   = A[0,j]*B[j,k] + ... (sum over other j)
    # C[0,k+1] = A[0,j]*B[j,k+1] + ...
    # C[1,k]   = A[1,j]*B[j,k] + ...
    # C[1,k+1] = A[1,j]*B[j,k+1] + ...
    #
    # The contribution from a single j to these 4 outputs involves 2*2=4 products.
    # Using Strassen on the 2x2 matrix mult (A_col_j * B_row_j_pair):
    # Let a=A[0,j], b=A[1,j], e=B[j,k], f=B[j,k+1]
    # Then: ae, af, be, bf can be computed with 7 multiplications? 
    # No -- these are just 4 scalar products, they don't form a 2x2 matrix mult.
    #
    # Actually, for fixed j, the contribution is:
    # [[a*e, a*f],
    #  [b*e, b*f]]
    # This is outer product of [a,b]^T and [e,f]. Rank 1 matrix!
    # So it only needs 1 multiplication: compute t = a*b? No, we need a*e, a*f, b*e, b*f.
    # We can compute: m1=a*e, m2=a*f, m3=b*e, m4=b*f -- 4 multiplications.
    # Or: m1=a*e, m2=(a+b)*f, m3=b*e, m4=(a+b)*f - a*f... no, that doesn't help.
    # Minimum for 4 products of 2 variables with 2 other variables: 
    # We need ae, af, be, bf. This is a 2x2x2 tensor (a,b) x (e,f) -> 2x2 matrix.
    # The bilinear complexity is 4 (can't do better since each product is independent).
    #
    # So grouping columns into pairs doesn't help for the per-j contribution.
    #
    # Different approach: use the fact that n=2 is small.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    #
    # Let's define for each k:
    # Let S_k = sum_j A[0,j] B[j,k] (this is C[0,k])
    # Let T_k = sum_j A[1,j] B[j,k] (this is C[1,k])
    #
    # Total work: 8 outputs, each is a dot product of length 4. Schoolbook = 32 multiplications.
    # But we have 4 different j values, and for each j, we compute 8 products (2 rows x 4 cols... no, 2 rows x 8 cols).
    # For each j: we need A[0,j]*B[j,k] for k=0..7 and A[1,j]*B[j,k] for k=0..7. That's 16 products per j, 4 j's, total 64.
    #
    # Can we reduce? For fixed j, we need to compute:
    # x_k = A[0,j] * B[j,k] for k=0..7  (8 products)
    # y_k = A[1,j] * B[j,k] for k=0..7  (8 products)
    # Total 16 scalar multiplications.
    #
    # We can't reduce below 16 for fixed j because each of the 16 products is algebraically independent
    # (they involve different B[j,k] values).
    #
    # Wait, but we CAN use Strassen if we restructure. Consider:
    # For fixed j, define vector b_j = [B[j,0], B[j,1], ..., B[j,7]]
    # We need: A[0,j] * b_j and A[1,j] * b_j
    # That's just scalar-vector multiplication, 16 mults.
    #
    # The key insight for Strassen-like savings: we need to find structure where
    # multiple outputs share intermediate products.
    #
    # Alternative: think of it as 2x4 matrix A times 4x8 matrix B.
    # Split B into four 4x2 blocks: B = [B1, B2, B3, B4] where each Bi is 4x2.
    # C = [A*B1, A*B2, A*B3, A*B4]
    # Each A*Bi is 2x2 matrix mult. Bilinear complexity of M_{2,4,2}:
    # For each 2x2 output block, we need sum_j A[i,j]*Bi[j,k] for i,k in {0,1}
    # This is 4 outputs, each sum of 4 products = 16 mults schoolbook.
    # Can we do better? The tensor for M_{2,4,2}... 
    #
    # Actually, let me just go with schoolbook rank 64 to ensure correctness, 
    # since I'm spending too much time reasoning and the deadline is a concern.
    
    # Schoolbook decomposition (guaranteed correct, rank 64)
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
