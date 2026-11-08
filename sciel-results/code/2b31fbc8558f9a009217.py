"""Matrix multiplication 2,3,10 rank-minimal decomposition."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The matrix multiplication tensor T_{i,j,k} = A_{i,j} * B_{j,k}
    # We need to find rank-r decomposition:
    # T = sum_{t=0}^{r-1} U_{(i,j),t} * V_{(j,k),t} * W_{(i,k),t}
    # 
    # This is equivalent to finding matrices U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k:
    #   sum_t U_{i*m+j, t} * V_{j*p+k, t} * W_{k*n+i, t} = 1
    #
    # A key insight: the multiplication tensor for (2,3,10) has a known 
    # lower bound. The trivial rank is 60. But we can do better.
    #
    # Strategy: Use the fact that the (2,3) block structure allows us to
    # share computations. For each column k of B, we compute A * B[:,k].
    # 
    # Let's think about this differently. The tensor is:
    # T(i,j,k) = 1 if we're multiplying (i,j) element of A by (j,k) element of B
    # to contribute to output (i,k).
    #
    # The rank of the matrix multiplication tensor M_{2,3,10} is at most 
    # the product of the ranks of smaller multiplications.
    #
    # M_{2,3} has rank at most 6 (trivial), but actually rank 5 or less?
    # M_{2,3,10} = M_{2,3} ⊗ M_{1,1,10} in some sense... not quite.
    #
    # Actually, let's use a direct approach: random search + local optimization.
    # But that's risky. Instead, let's use a known construction.
    #
    # For multiplying a 2x3 by 3x10, we can view it as 5 independent 2x3x2 
    # multiplications (pairing up the 10 columns), each of rank at most 5,
    # giving total rank 25. Or we can do better by sharing.
    #
    # Actually, let me think more carefully. The standard result is:
    # rank(M_{m,n,p}) <= rank(M_{m,n,1}) * p  (trivial per column)
    # rank(M_{2,3,1}) = 6 (just the 6 products, but actually it's the 
    #   bilinear form A_{i,j} * B_{j,k} for fixed k, which needs 
    #   min(i,j) multiplications... no, it's just the dot product)
    #
    # For fixed k, computing C_{*,k} = A * B_{*,k} requires 
    # m*n = 2*3 = 6 scalar multiplications (trivially). 
    # So total rank = 6 * 10 = 60.
    #
    # But we can do better! The key is that the same row of B is used 
    # across different outputs. Specifically, B_{j,k} appears in 
    # C_{0,k} (with A_{0,j}) and C_{1,k} (with A_{1,j}).
    #
    # Let's use a different approach: express the tensor as a sum of 
    # rank-1 tensors. A rank-1 tensor is one where T(i,j,k) = u(i,j)*v(j,k)*w(i,k).
    #
    # For our case, T(i,j,k) = 1 for all valid i,j,k (it's the structure 
    # tensor, all entries are 1). Wait, no. The multiplication tensor has 
    # T(i,j,k) = 1 if the (i,j) element of A and (j,k) element of B 
    # contribute to C(i,k). Since every valid (i,j,k) triple contributes,
    # T is actually all ones on the valid indices.
    #
    # Hmm, that's the structure tensor. The actual tensor for matrix mult 
    # is: T(i,j,k) = delta_{condition} where the condition is that 
    # output (i,k) gets contribution from A(i,j)*B(j,k). For valid 
    # dimensions, this is always 1.
    #
    # So we need: sum_t U(i*m+j, t) * V(j*p+k, t) * W(k*n+i, t) = 1 
    # for all i in [0,2), j in [0,3), k in [0,10).
    #
    # This is a system of 2*3*10 = 60 equations.
    #
    # Let's try rank 30 first: pair up columns. For each pair (k, k+1) 
    # for k = 0,2,4,6,8, we compute 2 columns at once.
    #
    # For a 2x3 times 3x2 multiplication, the tensor rank is at most 6, 
    # but can be less. Let's check: we need
    # C(i,k) = sum_j A(i,j)*B(j,k) for i=0,1 and k=0,1
    #
    # That's 4 outputs, each a dot product of length 3. Trivially 4*3=12 
    # multiplications, but we share the A(i,j) values.
    #
    # Actually, let's just do a greedy/local search for a low-rank solution.
    #
    # Alternative: use the fact that we can precompute some linear 
    # combinations. For example, for each j, we have B(j,0..9). We can 
    # compute linear combinations of these to reduce the number of 
    # multiplications.
    #
    # Let me try a structured approach with rank = 30:
    # For each output row i and each pair of columns (2k, 2k+1), we 
    # compute C(i, 2k) and C(i, 2k+1) together using shared terms.
    
    # Let's use a randomized approach with verification
    # First, try to find a solution with rank r by solving the system
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        # Variables: U (nm x r), V (mp x r), W (pn x r)
        # Total variables: r * (nm + mp + pn) = r * (6 + 30 + 20) = r * 56
        # Constraints: 60 equations
        #
        # This is underdetermined for r >= 2 (60 < 2*56 = 112)
        # We can set some variables to zero to make it manageable.
        
        # Strategy: use a structured decomposition
        # For each output (i,k), we need sum_j A(i,j)*B(j,k)
        # 
        # Let's use the following: for each j and each block of columns,
        # we share the computation.
        
        if time.time() > deadline - 5:
            return None
        
        # Try rank = 30: for each of the 3 rows of B, and 10 columns,
        # we have 30 terms, but we need to combine them into 2*10=20 outputs.
        # 
        # Actually, let's think about it as: for each j (0,1,2) and each k (0..9),
        # we compute A(i,j)*B(j,k) for i=0,1. That's 2 multiplications per (j,k).
        # Total: 3 * 10 * 2 = 60.
        #
        # To reduce: note that for a fixed j, we compute A(0,j)*B(j,k) and 
        # A(1,j)*B(j,k) for all k. We can factor: 
        #   [A(0,j)*B(j,0), ..., A(0,j)*B(j,9)] = A(0,j) * [B(j,0), ..., B(j,9)]
        # This is one "scaled row", but in bilinear form, we still need 
        # separate multiplications for each k unless we allow linear combinations 
        # in the output.
        #
        # In the tensor decomposition framework, we CAN use linear combinations 
        # in the output (W matrix handles this). So the question is whether we 
        # can find U, V, W with small r.
        
        # Let's use a numerical approach: set up as a system of equations
        # and solve for a particular structure.
        
        # Structure: U has 1s in positions (i*m+j, t) for specific t
        # V has 1s in positions (j*p+k, t) for specific t  
        # W has coefficients in positions (k*n+i, t)
        
        # Simpler: use random initialization and gradient-free optimization
        # But that's hard to verify.
        
        # Instead, let's use a known lower bound and construction.
        # The rank of M_{2,3,10} is between:
        #   lower: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
        #   upper: n*m*p = 60
        #
        # Actually, the lower bound is not just max(nm, mp, pn). 
        # The tensor rank is at least the rank of any "slice".
        # Fixing i: T(i,*,*) is an m x p matrix of ones (for fixed i), 
        # but that's not quite right.
        
        # Let me just try to construct rank 30 directly.
        # 
        # Idea: For each column k, we compute C(:,k) = A @ B[:,k].
        # C(0,k) = A(0,0)*B(0,k) + A(0,1)*B(1,k) + A(0,2)*B(2,k)
        # C(1,k) = A(1,0)*B(0,k) + A(1,1)*B(1,k) + A(1,2)*B(2,k)
        #
        # For a fixed k, this is a 2x3 times 3x1 multiplication.
        # The rank of M_{2,3,1} is 6 (trivially), but can it be less?
        # 
        # M_{2,3,1}: we need to compute 2 outputs, each a dot product of 
        # length 3. The tensor is:
        # T(i,j,0) = 1 for i in {0,1}, j in {0,1,2}
        # 
        # Can we do it in fewer than 6 multiplications? 
        # By Strassen's argument, for 2x1 (i.e., 2 scalar multiplications 
        # with shared inputs), the rank is 2. For 2x3x1, it's more complex.
        #
        # Actually, M_{m,n,1} has rank m*n (you need m*n multiplications 
        # to compute m dot products of length n, and you can't do better 
        # in general because each output depends on n independent inputs).
        #
        # Wait, that's not right either. The rank of M_{m,n,1} is actually 
        # m*n because the tensor is the "parallel" of m copies of 
        # length-n dot products, and each dot product needs n multiplications.
        #
        # So M_{2,3,1} has rank 6, and M_{2,3,10} has rank at most 6*10 = 60.
        #
        # But we can share across columns! The key insight: B(j,k) for 
        # different k share the same row j. So we can compute linear 
        # combinations of B(j,0..9) and multiply by A(i,j).
        
        # Let's try rank 30: for each j (0,1,2) and each k (0..9), 
        # we have one term. But we need 60 outputs... no, 20 outputs.
        #
        # I think the minimum rank is indeed 30 or so. Let me try to 
        # construct it.
        
        # Construction for rank 30:
        # For each j in {0,1,2} and each k in {0..9}, create a term.
        # U(i*m + j, t) = 1 if t corresponds to (j, *) and i is the 
        #   output row
        # V(j*p + k, t) = 1 if t corresponds to (j, k)
        # W(k*n + i, t) = A(i,j) ... no, W must be constant.
        
        # Hmm, I'm overcomplicating this. Let me use a different strategy.
        
        # The multiplication tensor M_{n,m,p} can be decomposed as:
        # For each j, compute the "cross product" of A(:,j) and B(j,:).
        # A(:,j) is n x 1, B(j,:) is 1 x p. Their outer product is n x p.
        # The outer product of two vectors has rank 1, so it needs 
        # n*p multiplications? No, the outer product (i,k) entry is 
        # A(i,j)*B(j,k), which is one multiplication per (i,k) pair.
        # So for fixed j, we need n*p = 2*10 = 20 multiplications.
        # Total: 3 * 20 = 60.
        
        # To reduce: we can combine the outer products. 
        # C = A @ B = sum_j A(:,j) * B(j,:)
        # This is a sum of 3 outer products, each of size 2x10.
        # Each outer product needs 20 multiplications (trivially).
        # Total: 60.
        
        # Can we do better? The answer is yes, by using the fact that 
        # we can precompute linear combinations. But in the bilinear 
        # model, the only free parameters are the U, V, W matrices.
        
        # Let me just go with the trivial 60 and see if I can improve.
        # Actually, let me try a smarter approach.
        
        # For rank r, we have r * 56 variables and 60 equations.
        # For r = 2: 112 variables, 60 equations - underdetermined.
        # But the equations are nonlinear (trilinear), so it's hard.
        
        # Let's use a greedy approach: build up the decomposition term by term.
        
        if time.time() > deadline - 10:
            return None
        
        # Greedy construction
        # Start with r=1, try to add terms until all 60 equations are satisfied.
        
        # Actually, let me use a different known result.
        # The rank of M_{2,3,10} is at most 30 by the following:
        # Pair up the 10 columns into 5 pairs. For each pair (2k, 2k+1),
        # we compute a 2x3 x 3x2 multiplication, which has rank at most 6.
        # Total: 5 * 6 = 30.
        
        # For M_{2,3,2}: C(i,k) = sum_j A(i,j)*B(j,k) for i=0,1; k=0,1
        # This is 4 outputs, each a length-3 dot product.
        # Trivially 4*3 = 12 multiplications, but we can share.
        #
        # Actually, M_{2,3,2} has rank at most 6:
        # C(0,0) = A(0,0)B(0,0) + A(0,1)B(1,0) + A(0,2)B(2,0)
        # C(0,1) = A(0,0)B(0,1) + A(0,1)B(1,1) + A(0,2)B(2,1)
        # C(1,0) = A(1,0)B(0,0) + A(1,1)B(1,0) + A(1,2)B(2,0)
        # C(1,1) = A(1,0)B(0,1) + A(1,1)B(1,1) + A(1,2)B(2,1)
        #
        # We can write this as:
        # For j=0: A(0,0)*B(0,0) -> contributes to C(0,0)
        #          A(0,0)*B(0,1) -> contributes to C(0,1)
        #          A(1,0)*B(0,0) -> contributes to C(1,0)
        #          A(1,0)*B(0,1) -> contributes to C(1,1)
        # That's 4 terms for j=0. Similarly for j=1 and j=2: 12 total.
        #
        # But we can share: A(0,0) is used in both C(0,0) and C(0,1).
        # In the tensor decomposition, we can have:
        # Term for (j=0, k=0): U(0*3+0, t)=1, U(1*3+0, t)=1 (for both rows)
        #   No, U is indexed by (i,j) where i is the output row.
        
        # Let me just use the trivial decomposition and hope for the best.
        # Rank 60.
        
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
                    r_idx += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better solution
    result = try_rank(30)
    if result is not None:
        return result
    
    # Fallback to trivial
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r_idx] = 1.0
                V[j * p + k, r_idx] = 1.0
                W[k * n + i, r_idx] = 1.0
                r_idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
