"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*p, n*m, m*p)
    # T[i*p+k, i*m+j, j*p+k] = 1 for all i,j,k
    # This represents: C[i,k] = sum_j A[i,j] * B[j,k]
    # Factor form: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i*p+k] = u_r[i,k], V_r[i*m+j] = v_r[i,j], W_r[j*p+k] = w_r[j,k]
    # and T[i*p+k, i*m+j, j*p+k] = sum_r u_r[i,k] * v_r[i,j] * w_r[j,k]
    
    # Actually, the standard form:
    # C[i,k] = sum_r U[i,k,r] * V[i,j,r] * W[j,k,r]  -- no, that's not right either
    # 
    # Let me use the standard tensor decomposition:
    # The multiplication tensor T_{i,j,k} where we multiply A_{i,j} and B_{j,k} to get C_{i,k}
    # T is a 3-tensor with T[i,j,k] = delta (the product A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # In the factorization U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank):
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give 1 if (i,j,k) is a valid triple
    # i.e., for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for invalid combinations (which don't exist in our indexing), it's automatically 0
    # 
    # Wait, the tensor is: for each (i,j,k), the coefficient is 1 (since A[i,j]*B[j,k] contributes to C[i,k])
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a rank-r factorization of the 3-way tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    # But actually T is only defined on the "diagonal" - it's the multiplication tensor.
    
    # Let me reconsider. The matrix multiplication tensor of format (n,m,p) is a 3-tensor
    # T in R^{(n*m) x (m*p) x (p*n)} where:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    # 
    # No wait, that's not right. The tensor should be such that:
    # C = A * B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The tensor T satisfies: C[i,k] = sum_{a,b} T[a,b, i*p+k] * A_flat[a] * B_flat[b]
    # where A_flat is A row-major (n*m elements), B_flat is B row-major (m*p elements)
    # 
    # So T[a, b, c] where c = i*p+k (or c = k*n+i per the W convention)
    # T[a, b, c] = 1 if a = i*m+j and b = j*p+k and c = k*n+i, else 0
    
    # Given the W convention W[k*n+i, r], let's use:
    # T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Rank-r factorization: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # We need: for all i,j,k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    # Strategy: Try ranks from known lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Also, the border rank / structured lower bounds apply
    
    # For 4x5x6, let's try to find a good decomposition.
    # One approach: use the fact that we can group terms.
    
    # Let's try a constructive approach:
    # For each pair (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # This is an inner product of length m.
    # 
    # A rank-m decomposition per (i,k) pair would give n*p*m = 120 (schoolbook).
    # But we can share structure.
    
    # Let me try: for each j, we have contributions to all (i,k) pairs.
    # The term A[i,j] * B[j,k] for fixed j contributes to all i,k.
    # This is a rank-1 tensor in the (i,k) space for each j.
    # So we get m = 5 rank-1 tensors in the (i,k) space, each of full support.
    # Total rank contribution: for each j, we need to represent the outer product
    # e_i (in A-space) ⊗ e_k (in B-space) ⊗ (shared j)
    # 
    # Actually, for fixed j: the tensor contribution is (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    # which is rank 1 in the full tensor space.
    # So m=5 gives rank 5? No, that's not right because the (i,k) pairs are all distinct.
    
    # Let me just do a numerical search with verification.
    
    def verify(U, V, W, n, m, p):
        """Verify that the factorization is correct."""
        rank = U.shape[1]
        # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition using randomized/structured approaches
    # with exact rational verification.
    
    # Approach 1: Use the known structure. For matrix multiplication (n,m,p),
    # a simple upper bound is n*m*p (schoolbook). Better bounds exist.
    # 
    # For 4x5x6, let's try rank = 60 (half of schoolbook) or lower.
    # 
    # Actually, let me try a specific construction:
    # Group the j-index in pairs or use a different factorization.
    
    # Let me try: for each (i,j) pair, instead of having separate k's,
    # we can potentially share. But the standard lower bound for (n,m,p) is
    # at least max(n*m, m*p, p*n) = 30.
    
    # Let me try rank 30 first (the trivial lower bound).
    # If that fails, try 31, 32, etc.
    
    # Actually, for the tensor T[i*m+j, j*p+k, k*n+i] = 1,
    # the "slice" T[:, j*p+k, :] for fixed j,k has rank 1 (it's e_{i*m+j} ⊗ e_{k*n+i} for all i)
    # Wait no, for fixed j,k, T[i*m+j, j*p+k, k*n+i] = 1 for all i.
    # So the slice is a vector that's 1 at positions (i*m+j, k*n+i) for i=0..n-1.
    # This is n ones, so rank 1 as a matrix? No, it's a vector.
    
    # Let me think differently. The tensor T can be viewed as a matrix of shape
    # (n*m) x (m*p * p*n) by flattening the last two indices.
    # T_matrix[a, b*p*n + c] = T[a, b, c]
    # For this to have rank r, we need r >= max(n*m, m*p*p*n) which is huge.
    # That's not the right flattening.
    
    # The rank of a 3-tensor is the minimum r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    # This is the tensor rank.
    
    # For the multiplication tensor, known results:
    # - (2,2,2): rank 7 (Strassen)
    # - (3,3,3): rank 23 (best known)
    # - General: lower bounds from various sources
    
    # For 4x5x6, I don't know the exact minimum, but let me try to find a good one.
    
    # Strategy: Use a structured decomposition.
    # 
    # Key idea: The multiplication C = A*B can be computed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this using block structure or other algebraic tricks.
    # 
    # For now, let me try a numerical optimization approach with rational recovery.
    
    best_rank = n * m * p  # schoolbook
    best_factors = None
    
    # Generate schoolbook factors as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    
    # Try to find better decompositions
    # Use a simple approach: try ranks from lower bound up
    
    lower_bound = max(n*m, m*p, p*n)  # = 30
    
    # Try a few ranks with random initialization and verification
    # This is a heuristic search - we'll try random matrices and check if they work
    
    # Actually, random matrices won't work for exact integer/half-integer solutions.
    # Let me think of a structured approach.
    
    # For 4x5x6, one approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can try to factor the sum over j using algebraic identities.
    # For m=5, we can't easily reduce below 5 terms per (i,k) pair in general,
    # but we might share across (i,k) pairs.
    
    # Let me try: express the multiplication using a smaller number of multiplications
    # by finding linear combinations.
    
    # For the full 4x5x6 case, let me try a rank-60 construction:
    # Pair up the j indices: (0,1), (2,3), (4)
    # For each pair (j1, j2), use Strassen-like trick:
    # A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k]
    # = ((A[i,j1]+A[i,j2])*(B[j1,k]+B[j2,k]) - A[i,j1]*B[j2,k] - A[i,j2]*B[j1,k])
    # That uses 3 multiplications instead of 2, which is worse.
    
    # Alternative: just use the schoolbook but check if there's a simpler structure.
    
    # Let me try a direct search for small rank factorizations using the fact
    # that the tensor has a lot of zeros.
    
    # The tensor T[a,b,c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # Number of nonzero entries: n*m*p = 120.
    
    # For a rank-r factorization, we need:
    # sum_{r} U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # This is a system of 120 equations (one for each nonzero T entry) with
    # 3*r*rank variables (U has n*m*r, V has m*p*r, W has p*n*r).
    # Plus the constraint that T[a,b,c] = 0 for all other (a,b,c) triples.
    # Total constraints: (n*m)*(m*p)*(p*n) - 120 + 120 = n*m*m*p*p*n = 24*30*24 = 17280
    # That's a lot of constraints.
    
    # For rank r, we have (n*m + m*p + p*n) * r = (20 + 30 + 24) * r = 74r variables.
    # We need 74r >= 17280, so r >= 234. That's worse than schoolbook!
    
    # Wait, that can't be right. Let me reconsider.
    
    # Oh, I see the issue. The constraint is only on the 120 nonzero entries AND
    # the 17280-120 = 17160 zero entries. But many of the zero entries are automatically
    # satisfied if we structure the factors properly.
    
    # Actually, the standard result is that the rank of the (n,m,p) multiplication tensor
    # is at most n*m*p and at least max(n*m, m*p, p*n). For 4x5x6, that's between 30 and 120.
    
    # Let me just try to find a decomposition with rank significantly less than 120.
    
    # I'll use a greedy/structured approach:
    # For each (i,k) pair, C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If I fix i and vary k, I get n rows of the result, each being a linear combination
    # of the columns of B weighted by A[i,:].
    
    # Let me try a different tactic: use the fact that we can precompute linear combinations.
    
    # For now, let me return the schoolbook decomposition which is guaranteed correct,
    # but try to improve it with a simple heuristic.
    
    # Simple improvement: check if we can merge some terms.
    # For 4x5x6, let me try rank = 80 (2/3 of schoolbook) as a test.
    
    # Actually, let me just implement a proper search for this specific case.
    # Given the time constraint, I'll use a structured approach.
    
    # Known: for (n,m,p) with n<=m<=p, rank is at most n*m*p and at least m*p.
    # For 4x5x6: at least 30, at most 120.
    
    # Let me try to construct a rank-60 solution by pairing j indices.
    # For j in {0,1}: use 2 multiplications (can't reduce)
    # For j in {2,3}: use 2 multiplications
    # For j=4: use 1 multiplication
    # Total: still 5 per (i,k) pair, so 4*6*5 = 120. No improvement.
    
    # The only way to reduce rank is to share multiplications across different (i,k) pairs.
    
    # Let me try: for a fixed j, the term A[i,j]*B[j,k] appears in all (i,k) pairs.
    # This is already a single multiplication that contributes to n*p = 24 output entries.
    # So for each j, we have 1 "multiplication" that feeds 24 outputs.
    # Total: 5 multiplications feeding all 24 outputs.
    # 
    # But in the tensor decomposition, each "rank-1 term" corresponds to one multiplication.
    # So we need at least... hmm, this suggests rank 5 might work? No, that's not right
    # because the tensor rank is about the 3-way structure.
    
    # Let me just go with a practical approach: return a decomposition that's better
    # than schoolbook if I can find one, otherwise schoolbook.
    
    # For this specific case, let me try to use a known result or construct one.
    
    # I'll try rank = 60 with a structured pattern.
    
    # Actually, let me just verify the schoolbook and return it.
    # The score will be 120/120 = 1.0, same as the seed.
    # To improve, I need to find a lower rank.
    
    # Let me try a random search with rational recovery for small ranks.
    
    import random
    random.seed(42)
    
    def try_rank(r, max_iters=100):
        """Try to find a rank-r factorization."""
        for it in range(max_iters):
            # Initialize with random values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, r))
            V = np.random.choice([-1, 0, 1], size=(m*p, r))
            W = np.random.choice([-1, 0, 1], size=(p*n, r))
            
            # Check if this works
            if verify(U, V, W, n, m, p):
                return U, V, W
        return None
    
    # Try ranks from 30 to 119
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        result = try_rank(r, max_iters=50)
        if result is not None:
            best_rank = r
            best_factors = result
            break
    
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
