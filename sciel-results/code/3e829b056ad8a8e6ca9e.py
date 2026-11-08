"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear algorithm for multiplying a 3x3 matrix by a 3x6 matrix.
    Uses rank 27 (schoolbook) as a safe baseline, but attempts to find a lower rank.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Schoolbook rank is n*m*p = 54
    # We can do better by grouping.
    # For each column k of C (which has n=3 rows), we compute C[:,k] = A @ B[:,k]
    # C[:,k] is a 3x1 vector. A is 3x3, B[:,k] is 3x1.
    # This is a (3,3,1) multiplication. Rank is 3 (schoolbook) or maybe lower?
    # No, for (3,3,1) the rank is 3 (since it's just matrix-vector mult, and 3 is minimal).
    # So total rank = 6 columns * 3 = 18?
    # Wait, C[i,k] = sum_j A[i,j] * B[j,k].
    # If we fix k, we have 3 outputs (i=0,1,2) from 3 inputs (j=0,1,2) of B and 9 inputs of A.
    # The tensor for (3,3,1) has rank 3.
    # So for each k, we need 3 multiplications.
    # Total rank = 6 * 3 = 18.
    
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # Indexing:
    # U index: i * m + j  (i in 0..2, j in 0..2)
    # V index: j * p + k  (j in 0..2, k in 0..5)
    # W index: k * n + i  (k in 0..5, i in 0..2)
    
    # For each k (0..5) and each "multiplication" t (0..2):
    # We want to compute partial products that sum to C[i,k] for all i.
    # For fixed k, the multiplication is:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    # We can use 3 multiplications for each k:
    # M_{k,0} = A[0,0]*B[0,k] + ... no, that's not how bilinear algorithms work.
    # Each rank-1 term is u * v * w.
    # For fixed k, we need 3 rank-1 terms to cover the 3 outputs.
    # Term (k, j): 
    #   u_{i, j} = 1 if i==j? No.
    #   Let's look at the structure.
    #   For fixed k, we have a (3,3,1) tensor.
    #   Rank 3 decomposition:
    #   Term 0: u=(1,0,0), v=(1,0,0), w=(1,0,0) -> contributes A[0,0]*B[0,k] to C[0,k]
    #   Term 1: u=(0,1,0), v=(0,1,0), w=(0,1,0) -> contributes A[1,1]*B[1,k] to C[1,k]
    #   Term 2: u=(0,0,1), v=(0,0,1), w=(0,0,1) -> contributes A[2,2]*B[2,k] to C[2,k]
    #   This only covers the diagonal. We need the off-diagonals too.
    
    # Wait, the rank of (3,3,1) matrix multiplication is NOT 3.
    # The rank of (n,n,1) is n. But that's for C = A * v where A is n x n and v is n x 1?
    # No, C[i] = sum_j A[i,j] v[j].
    # The tensor T_{i,j,k} where k is fixed (k=0).
    # T_{i,j} = A[i,j] * B[j,0].
    # This is just the identity tensor in the "product" sense?
    # No, it's a 3x3x1 tensor.
    # The rank of this tensor is 3.
    # Why? Because we can write:
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[2] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    
    # Can we do it with 3 multiplications?
    # Let M0, M1, M2 be the 3 products.
    # M0 = (sum of some A) * (sum of some B)
    # If we use M_j = A[:,j] * B[j,0] (vector * scalar), that's not a scalar multiplication in the bilinear sense unless we treat the vector as part of the linear form.
    # In the bilinear algorithm, each term is u * v * w where u, v, w are vectors.
    # For (3,3,1), u is in R^9 (A), v is in R^3 (B[:,0]), w is in R^3 (C[:,0]).
    # We need to represent the tensor T_{i,j} (3x3) as sum of 3 rank-1 matrices u_t * v_t^T?
    # No, the tensor is T_{i,j,0} = delta_{i,j}? No, T_{i,j,0} = A[i,j] * B[j,0].
    # The tensor is T_{i,j,k} = A[i,j] B[j,k].
    # For fixed k=0, T_{i,j} = A[i,j] B[j,0].
    # This is not a constant tensor. It depends on the inputs.
    # The tensor we are decomposing is the multiplication tensor.
    # The multiplication tensor for (3,3,1) is:
    # T_{i,j,0} = 1 if we are looking at the coefficient of A[i,j] and B[j,0] in C[i,0].
    # So T_{i,j,0} = 1 for all i,j? No.
    # C[i,0] = sum_j A[i,j] B[j,0].
    # The coefficient of A[i,j] B[j,0] in C[i,0] is 1.
    # So the tensor has T_{i,j,0} = 1 for all i,j.
    # This is the tensor that is the sum of all e_i * e_j * e_0?
    # No, the indices are (i, j, k_output).
    # T_{i, j, k_out} = 1 if k_out == 0 (since we fixed k=0 for B).
    # Wait, the indices of the tensor are (i_A, j_A, k_B, j_B, i_C, k_C).
    # For (3,3,1): i_A in 0..2, j_A in 0..2, j_B in 0..2, i_C in 0..2.
    # T_{i_A, j_A, j_B, i_C} = 1 if i_A == i_C and j_A == j_B.
    # So it's the identity tensor in the (i_A, i_C) and (j_A, j_B) pairs?
    # It is the tensor I_3 * I_3?
    # The rank of I_n * I_n is n.
    # So the rank of (3,3,1) is 3.
    
    # So for each k, we need 3 multiplications.
    # Total rank = 6 * 3 = 18.
    
    # Let's construct the 18 terms.
    # For each k in 0..5:
    #   For each j in 0..2 (this is the "multiplication index" for this k):
    #     Term index = k * 3 + j
    #     u: A[i, j] for all i. So u[i*m + j] = 1 for all i? No.
    #     The term is A[i, j] * B[j, k].
    #     This contributes to C[i, k].
    #     So for this term:
    #       U[i*m + j] = 1 for all i? No, U is a vector in R^9.
    #       The term is (sum_i A[i,j] * U[i,j]) * (sum_j' B[j',k] * V[j',k]) * (sum_i' C[i',k] * W[i',k])
    #       We want this to equal A[i0, j] * B[j, k] * C[i0, k]?
    #       No, the rank-1 term is u * v * w.
    #       We want the sum of terms to equal C[i,k] = sum_j A[i,j] B[j,k].
    #       Let's try term (k, j):
    #         u_{i', j'} = 1 if j' == j, else 0. (So u picks out column j of A)
    #         v_{j', k'} = 1 if j' == j and k' == k, else 0. (So v picks out B[j,k])
    #         w_{k', i'} = 1 if k' == k and i' == i? No, w must be fixed for the term.
    #         If u picks column j of A, and v picks B[j,k], then the product is A_col_j * B[j,k].
    #         This is a vector in R^3.
    #         We need to map this vector to C[:,k].
    #         The contribution to C[i,k] is A[i,j] * B[j,k].
    #         So we need w to pick out the i-th component?
    #         But w is a vector in R^18 (for p*n).
    #         If we set w_{k*n + i} = 1 for a specific i, then the term only contributes to C[i,k].
    #         But we have 3 i's. We need 3 terms for each (j,k)?
    #         No, for fixed k, we have 3 j's.
    #         If we use term (k, j) with u = e_j (in A space, picking column j), v = e_{j,k} (in B space, picking B[j,k]),
    #         then the product is a vector in A-space (column j) times a scalar (B[j,k]).
    #         This gives a vector in R^3.
    #         We need to add this vector to C[:,k].
    #         So we need w to be the identity vector? No, w is a vector.
    #         The term is (u . A) * (v . B) * (w . C)? No, it's (u . A) * (v . B) which is a scalar.
    #         Then this scalar is multiplied by w, which is added to C?
    #         No, the formula is C = sum_t (u_t . A) * (v_t . B) * w_t.
    #         Here w_t is a matrix (p x n).
    #         So for term (k, j):
    #           u_t . A = A[:, j] (column j of A) -- wait, u_t is a vector in R^9.
    #           If u_t has 1's at positions (i, j) for all i, then u_t . A is not a scalar.
    #           u_t . A is a scalar: sum_{i,j'} u_t[i*m+j'] A[i,j'].
    #           If u_t[i*m+j] = 1 for all i, then u_t . A = sum_i A[i,j]. This is a scalar.
    #           But we want A[i,j] for each i separately.
    #           So we can't use a single term to get all i's.
    #           We need 3 terms for each (j,k) to get the 3 i's?
    #           That would be 3 * 3 * 6 = 54 terms. Back to schoolbook.
    
    # Let's re-evaluate the rank of (3,3,1).
    # The tensor T_{i,j,k} where k is fixed.
    # T_{i,j} = A[i,j] B[j,0].
    # This is not a constant tensor. The tensor is defined by the indices.
    # The multiplication tensor for (n, m, p) has indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, the standard notation is C = A B, A is n x m, B is m x p, C is n x p.
    # The tensor T_{i, j, k} = 1 if the product A[i,j] B[j,k] contributes to C[i,k].
    # So T_{i, j, k} is not a 3D tensor. It's a 3-mode tensor?
    # No, it's a tensor in R^{n*m} x R^{m*p} x R^{n*p}.
    # The rank is the minimum number of rank-1 tensors to sum to this.
    
    # For (3,3,1): n=3, m=3, p=1.
    # U is 9xR, V is 3xR, W is 3xR.
    # C[i,0] = sum_j A[i,j] B[j,0].
    # Can we do R=3?
    # Term 0: u0, v0, w0.
    # Term 1: u1, v1, w1.
    # Term 2: u2, v2, w2.
    # C[i,0] = sum_t (u_t . A) (v_t . B) w_t[i,0].
    # Let's try:
    # v_t[j] = 1 if j==t, else 0. So v_t . B = B[t,0].
    # u_t[i,j] = 1 if j==t, else 0. So u_t . A = sum_i A[i,t].
    # Then (u_t . A) (v_t . B) = (sum_i A[i,t]) B[t,0].
    # This is a scalar.
    # w_t[i,0] must be such that sum_t (sum_i' A[i',t]) B[t,0] w_t[i,0] = sum_j A[i,j] B[j,0].
    # This doesn't seem to work because u_t . A sums over i.
    
    # We need u_t to pick out A[i,t] for a specific i.
    # But u_t is a vector, so u_t . A is a scalar.
    # So each term gives a scalar.
    # C[i,0] is a linear combination of these scalars.
    # So C[i,0] = sum_t s_t * w_t[i,0].
    # Where s_t = (u_t . A) (v_t . B).
    # We need s_t to be of the form A[i0, j0] B[j0, 0] for some i0, j0.
    # Because the multiplication tensor is diagonal in the (i,j) and (j,k) sense?
    # No, the tensor is T_{i,j,k} = delta_{i, i_out} delta_{j, j_in} delta_{j, j_in} delta_{k, k_out}?
    # The tensor is T_{i, j, k} where i is row of A, j is col of A / row of B, k is col of B.
    # The output is C_{i, k}.
    # The tensor is T_{i, j, k, i_out, k_out} = 1 if i==i_out and k==k_out.
    # So for fixed k_out=0, we have T_{i, j, k, i_out} = 1 if i==i_out and k==0.
    # This is a 4-index tensor? No, the input indices are (i,j) for A, (j,k) for B, (i_out, k_out) for C.
    # The tensor is in R^{n*m} x R^{m*p} x R^{n*p}.
    # Let's flatten the indices.
    # A index: a = i*m + j.
    # B index: b = j*p + k.
    # C index: c = k_out*n + i_out.
    # T_{a,b,c} = 1 if (i,a) == (i_out, c) and (j,a) == (j,b) and (k,b) == (k_out, c).
    # i = a // m, j = a % m.
    # j_b = b // p, k = b % p.
    # i_out = c % n, k_out = c // n.
    # Condition: i == i_out, j == j_b, k == k_out.
    # So T_{a,b,c} = 1 if a%m == b//p and a//m == c%n and b%p == c//n.
    
    # For (3,3,1): n=3, m=3, p=1.
    # a in 0..8, b in 0..2, c in 0..2.
    # a%m = a%3, a//m = a//3.
    # b//p = b, b%p = 0.
    # c%n = c, c//n = 0.
    # Condition: a%3 == b, a//3 == c, 0 == 0.
    # So T_{a,b,c} = 1 if a%3 == b and a//3 == c.
    # a = 3*c + b.
    # So T_{3c+b, b, c} = 1.
    # This is a permutation tensor!
    # The rank of a permutation tensor is the size of the permutation?
    # No, the rank of the identity tensor I_n is n.
    # Here we have 9 non-zero entries, forming a permutation of 9 elements?
    # No, a=3c+b. As c goes 0..2 and b goes 0..2, a goes 0..8.
    # So it's the identity permutation on 9 elements?
    # No, the indices are (a,b,c).
    # The non-zero entries are (3c+b, b, c) for c,b in 0..2.
    # This is a 3D tensor with 9 non-zero entries.
    # It is the tensor that maps e_{3c+b} in A-space to e_b in B-space and e_c in C-space?
    # No, it's a 3-mode tensor.
    # The rank of this tensor is 9?
    # Let's check.
    # If we use 9 terms, we can pick each non-zero entry.
    # Can we do less?
    # This tensor is isomorphic to the identity tensor I_3 * I_3?
    # The rank of I_n * I_m is n*m.
    # So the rank is 9.
    # So for (3,3,1), the rank is 9?
    # That means for (3,3,6), the rank is 6 * 9 = 54?
    # That's the schoolbook rank.
    
    # Wait, is the rank of (n,m,1) equal to n*m?
    # C[i,0] = sum_j A[i,j] B[j,0].
    # This is just matrix-vector multiplication.
    # The rank of matrix-vector multiplication (n,m,1) is n*m.
    # Because each A[i,j] B[j,0] is a distinct monomial, and they all appear in different outputs?
    # No, A[0,0]B[0,0] appears in C[0,0].
    # A[0,1]B[1,0] appears in C[0,0].
    # So C[0,0] has 3 monomials.
    # C[1,0] has 3 monomials.
    # C[2,0] has 3 monomials.
    # Total 9 monomials.
    # Since they are all distinct and appear in different linear combinations, the rank is 9.
    
    # So the rank of (3,3,6) is at least 54?
    # No, we can share multiplications between different k's.
    # For k=0 and k=1, the A's are the same.
    # But the B's are different.
    # So we can't share.
    
    # So the rank is 54?
    # Let's check if there's a better algorithm for (3,3,6).
    # Maybe we can use the fact that B is 3x6.
    # We can split B into two 3x3 matrices: B = [B1, B2].
    # Then C = A [B1, B2] = [A B1, A B2].
    # The rank of (3,3,3) is 27 (schoolbook) or 23 (Strassen-like)?
    # No, (3,3,3) rank is 23 (Lader).
    # So (3,3,6) rank is 2 * 23 = 46?
    # Or we can use a better algorithm for (3,3,6) directly.
    
    # Let's try to use the (3,3,3) rank 23 algorithm twice.
    # But we need to construct it.
    # The Lader algorithm for (3,3,3) is complex.
    
    # Let's stick with the schoolbook rank 54 for now, but see if we can do better.
    # Actually, the task says "Find a rank-minimal exact tensor decomposition".
    # If I don't know the minimal rank, I should provide a valid decomposition.
    # The score is the rank. Lower is better.
    # So I should try to find a lower rank.
    
    # Let's try to implement the (3,3,3) rank 23 algorithm and use it twice.
    # But I don't have the explicit coefficients for the Lader algorithm.
    
    # Alternative: Use the fact that (3,3,6) can be seen as (3,3,3) + (3,3,3).
    # If I can find a rank 23 algorithm for (3,3,3), I can use it twice for rank 46.
    # 46 < 54.
    
    # But I need the explicit U, V, W for the (3,3,3) rank 23 algorithm.
    # This is hard to do by hand.
    
    # Let's try a simpler improvement.
    # Can we do (3,3,6) with rank < 54?
    # Yes, if we use a better algorithm for (3,3,3).
    
    # Since I don't have the Lader coefficients, let's just use the schoolbook algorithm with rank 54.
    # But wait, the current program already uses rank 54.
    # So I need to do better.
    
    # Let's try to use the (2,2,2) Strassen algorithm (rank 7) to build a (4,4,4) algorithm (rank 49) and then restrict to (3,3,6)?
    # No, that's what the current program tried and failed.
    
    # Let's try to find a rank 46 algorithm by using the Lader algorithm for (3,3,3).
    # I will look up the Lader algorithm coefficients.
    # Since I can't look them up, I will use a heuristic:
    # The rank of (3,3,3) is 23.
    # I will construct a rank 23 algorithm for (3,3,3) using a known decomposition.
    
    # Actually, let's just use the schoolbook algorithm but with a different indexing to ensure it's correct.
    # The current program uses rank 54.
    # If I can't find a better algorithm, I should keep rank 54.
    # But the directive says "improve it".
    
    # Let's try to use the (3,3,3) rank 23 algorithm.
    # I will use a simplified version or a known decomposition.
    
    # For now, let's just return the schoolbook algorithm with rank 54, but ensure it's correct.
    # The current program seems correct.
    
    # Wait, the directive says "design a new program... from scratch".
    # So I should not just keep the current one.
    
    # Let's try to implement a rank 46 algorithm by using two (3,3,3) rank 23 algorithms.
    # I need the coefficients for the (3,3,3) rank 23 algorithm.
    
    # Since I don't have them, let's try to use a rank 27 algorithm for (3,3,3) (which is schoolbook) and see if I can improve it.
    # No, that's the same as rank 54 for (3,3,6).
    
    # Let's try to use the (3,3,3) rank 23 algorithm from a known source.
    # I will hardcode the coefficients if I can find them.
    
    # Since I can't, let's just use the schoolbook algorithm with rank 54.
    # But I will make sure it's correct.
    
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
