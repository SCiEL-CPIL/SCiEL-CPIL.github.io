"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product decomposition:
    # Decompose 2x3x6 multiplication as tensor product of (2x1x3) and (1x3x2).
    #
    # The 2x3x6 multiplication tensor T_{i,j,k} (i in 0..1, j in 0..2, k in 0..5)
    # can be viewed as: T_{i,j,k} = A_{i,j,l} * B_{l,j',k'} where we split
    # the middle dimension and output dimension appropriately.
    #
    # Actually, let's think more carefully. We have a 2x3 matrix A and 3x6 matrix B.
    # The product C = A*B is 2x6.
    #
    # We want to decompose the bilinear map (A, B) -> C.
    #
    # Approach: Write the 3x6 matrix B as a 2x2 block matrix where each block is 3x3?
    # No, 3x6 = 3 rows, 6 cols. We can split the 6 columns into two groups of 3.
    # So B = [B1 | B2] where B1, B2 are 3x3.
    #
    # Then C = A*B = [A*B1 | A*B2].
    #
    # Now A is 2x3, B1 and B2 are 3x3.
    #
    # The multiplication of a 2x3 matrix by a 3x3 matrix: rank of this tensor?
    # The 2x3x3 multiplication tensor. Its rank is at most 2*3*3=18, but can be lower.
    #
    # Hmm, let me reconsider the Kronecker product approach mentioned in the directive.
    #
    # The directive says: "Model the 2x3x6 multiplication tensor as a Kronecker product 
    # of a 2x1x3 tensor (which has rank 3) and a 1x3x2 tensor (which has rank 3)"
    #
    # Let me think about this differently. 
    #
    # Consider the index structure. We have:
    # - Input 1: A[i,j] with i in {0,1}, j in {0,1,2}  (2x3)
    # - Input 2: B[j,k] with j in {0,1,2}, k in {0,...,5}  (3x6)
    # - Output: C[i,k] with i in {0,1}, k in {0,...,5}  (2x6)
    #
    # The tensor is T[i,j,k] = 1 if the multiplication contributes, i.e., for each (i,j,k),
    # C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # So the tensor has entries T[i,j,k] = 1 for all valid (i,j,k).
    #
    # Now, the Kronecker product idea: can we write this as a tensor product?
    #
    # Let's try to factor the dimensions:
    # - i: 2
    # - j: 3  
    # - k: 6
    #
    # We can write 3 = 1*3 and 6 = 3*2.
    #
    # Let j = j1 * 1 + j2 where j1 in {0}, j2 in {0,1,2}? That's just j = j2.
    # Let k = k1 * 2 + k2 where k1 in {0,1,2}, k2 in {0,1}? So k = 2*k1 + k2.
    #
    # Hmm, this doesn't quite work as a clean Kronecker product.
    #
    # Let me try another approach. The directive mentions:
    # - 2x1x3 tensor with rank 3
    # - 1x3x2 tensor with rank 3
    # - Their Kronecker product gives rank 3*3 = 9
    #
    # For a 2x1x3 multiplication: multiply 2x1 by 1x3 to get 2x3.
    # The tensor T[i,j,k] = A[i,j] * B[j,k] where A is 2x1, B is 1x3, result is 2x3.
    # Here j is the only index in {0}, so T[i,0,k] = A[i,0]*B[0,k].
    # This is just an outer product structure. The rank is 2*3 = 6? No wait.
    #
    # Actually for 2x1x3: C[i,k] = A[i,0] * B[0,k].
    # This is a rank-1 tensor in the sense that it's a sum over j of A[i,j]*B[j,k],
    # but j only takes one value. So C[i,k] = A[i,0]*B[0,k].
    # This is just the outer product of the vector A[:,0] and B[0,:].
    # The bilinear map has rank... well, it's just multiplying two vectors to get a matrix.
    # The number of independent products is 2*3 = 6, so the rank is 6? 
    #
    # Wait, no. The "rank" of a tensor in the context of matrix multiplication refers to
    # the minimum number of rank-1 tensors needed to express it. A rank-1 tensor in this
    # context is of the form u_i * v_j * w_k for some vectors u, v, w.
    #
    # For the 2x1x3 case: T[i,j,k] = delta_{j,0} * (something). Actually T[i,j,k] = 1 if 
    # the term A[i,j]*B[j,k] contributes. Since j must be 0, T[i,0,k] = 1 for all i,k,
    # and T[i,j,k] = 0 otherwise.
    #
    # So T = sum over i in {0,1}, k in {0,1,2} of e_i * e_0 * e_k.
    # That's 6 rank-1 tensors. So the rank is 6, not 3.
    #
    # Hmm, the directive says rank 3. Let me reconsider.
    #
    # Maybe I'm misunderstanding the tensor structure. Let me re-read:
    # "a 2x1x3 tensor (which has rank 3)"
    #
    # Perhaps they mean the multiplication of a 2x1 matrix by a 1x3 matrix, but the
    # "rank" refers to something else? Or maybe it's a different kind of tensor.
    #
    # Actually, wait. Maybe the decomposition is not about standard matrix multiplication
    # tensors but about a structured decomposition.
    #
    # Let me try a different interpretation. What if we decompose the 3x6 matrix B into
    # a product of a 3x3 matrix and a 3x2 matrix? No, that changes the problem.
    #
    # Let me just try to construct a rank-9 decomposition directly.
    #
    # We need: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the convention is W[k*n+i, r]. Let me re-read the interface.
    #
    # U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    # V has shape (m*p, rank) = (3*6, rank) = (18, rank)
    # W has shape (p*n, rank) = (6*2, rank) = (12, rank)
    #
    # The decomposition is: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, that's not right either. The standard decomposition is:
    # C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c]
    # and W is used differently.
    #
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    #
    # I think the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # And T[i,j,k] = 1 for the matrix multiplication tensor.
    #
    # So we need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    #
    # With rank 9, let's try to construct this.
    #
    # Let me try a simple construction. Use 9 columns.
    #
    # For the schoolbook, we have 2*3*6 = 36 terms. We want to reduce to 9.
    #
    # Let's try: for each (i, j, k), we need the sum to be 1.
    #
    # Idea: Use the fact that 6 = 3*2. Write k = 2*k1 + k2 where k1 in {0,1,2}, k2 in {0,1}.
    #
    # Let me try a construction where we group the k indices.
    #
    # Actually, let me just try a direct construction with rank 9 and see if I can make it work.
    #
    # Let's use 9 columns. Label them by (a, b) where a in {0,1,2}, b in {0,1,2}.
    #
    # U[i*m+j, (a,b)] = 1 if i == a % 2 and j == b, else 0? Let's see...
    #
    # This is getting complicated. Let me just try the Kronecker product approach more carefully.
    #
    # The 2x3x6 tensor can be seen as a tensor product of smaller tensors if we can factor
    # the dimensions. 2 = 2*1, 3 = 3*1, 6 = 3*2.
    #
    # So T_{i,j,k} where i in {0,1}, j in {0,1,2}, k in {0,...,5}.
    # Write k = 2*k1 + k2, k1 in {0,1,2}, k2 in {0,1}.
    #
    # Then T_{i,j,2*k1+k2} = 1 for all i, j, k1, k2.
    #
    # This is the tensor product of:
    # - T1_{i,j} = 1 for i in {0,1}, j in {0,1,2}  (2x3 tensor, all ones)
    # - T2_{k1,k2} = 1 for k1 in {0,1,2}, k2 in {0,1}  (3x2 tensor, all ones)
    #
    # The rank of an all-ones tensor of size a x b is a*b (since it's the outer product of
    # two all-ones vectors, which is rank 1... wait, no).
    #
    # Actually, the all-ones 2x3 tensor as a "multiplication tensor" doesn't quite make sense.
    #
    # I think I'm overcomplicating this. Let me just try to find a rank-9 decomposition
    # by solving the system of equations.
    #
    # We have 2*3*6 = 36 equations (one for each (i,j,k)) and 9*6 + 9*18 + 9*12 = 54 + 162 + 108 = 324 unknowns.
    # So it's underdetermined, and a solution likely exists.
    #
    # Let me try a simple structured solution.
    #
    # Let rank = 9.
    #
    # For column c in 0..8, let c = 3*a + b where a in {0,1,2}, b in {0,1,2}.
    #
    # U[i*m+j, c] = U[3*i+j, 3*a+b]
    # V[j*p+k, c] = V[6*j+k, 3*a+b]
    # W[k*n+i, c] = W[2*k+i, 3*a+b]
    #
    # Let's try:
    # U[3*i+j, 3*a+b] = 1 if i==a and j==b, else 0
    # V[6*j+k, 3*a+b] = 1 if j==b, else 0  (independent of k? No, that won't work)
    #
    # Hmm, let me try:
    # U[3*i+j, 3*a+b] = 1 if i==a%2 and j==b, else 0
    # V[6*j+k, 3*a+b] = 1 if j==b, else 0
    # W[2*k+i, 3*a+b] = 1 if k==a, else 0
    #
    # Then for a given (i,j,k), the sum over c of U*V*W is:
    # sum over a,b of [i==a%2 and j==b] * [j==b] * [k==a]
    # = sum over a,b of [i==a%2] * [j==b] * [j==b] * [k==a]
    # = [i==k%2] * [j==j] * 1  (if k is in 0..2, but k is in 0..5)
    #
    # This doesn't work because k ranges over 0..5, not 0..2.
    #
    # Let me try a different approach. Let me just use the schoolbook but with some
    # clever grouping to reduce the rank.
    #
    # Actually, let me just try rank 9 with a simple pattern and verify.
    
    n, m, p = DIMENSIONS
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Let's try a construction based on the Kronecker product idea.
    # Decompose k = 3*k1 + k2 where k1 in {0,1}, k2 in {0,1,2}.
    # So k = 3*k1 + k2, with k1 in {0,1}, k2 in {0,1,2}.
    
    # Use 9 columns indexed by (i_val, k2_val) where i_val in {0,1}, k2_val in {0,1,2}.
    # That's 2*3 = 6 columns. Not enough.
    
    # Let me use (a, b) where a in {0,1,2}, b in {0,1,2}.
    
    # Try:
    # U[3*i+j, 3*a+b] = 1 if i == a % 2 and j == b, else 0
    # V[6*j+k, 3*a+b] = 1 if j == b and k == 3*a + (k % 3), else 0  -- this is getting messy
    
    # Let me just try a brute-force construction and verify it works.
    
    # For each (i, j, k), we need sum_c U[3*i+j, c] * V[6*j+k, c] * W[2*k+i, c] = 1
    
    # Let's try:
    # U[3*i+j, c] = 1 if c == 3*i + j, else 0
    # V[6*j+k, c] = 1 if c == 3*i + j, else 0  -- but i is not known in V
    #
    # This won't work because V doesn't depend on i.
    
    # OK let me try yet another approach. Let me use the fact that the problem is small
    # and just construct a valid rank-9 decomposition by solving a system of linear equations
    # or by using a known construction.
    
    # Actually, let me just try the schoolbook with rank 36 first to make sure the interface
    # is correct, then try to reduce.
    
    # For now, let me return a valid rank-36 decomposition to ensure correctness,
    # then I'll try to optimize.
    
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
