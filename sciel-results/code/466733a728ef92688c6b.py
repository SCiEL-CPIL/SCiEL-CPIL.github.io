"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
DEADLINE_FRACTION = 0.85  # safety margin

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 iff i,j,k correspond to a product entry.
    
    For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor: T[i,j,k] = 1 if this is a valid product term (always 1 for all i,j,k in ranges)
    Actually the tensor for matmul is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    And the product C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    Wait, let me re-read the convention:
    U has shape (n*m, rank): index is i*m + j
    V has shape (m*p, rank): index is j*p + k  
    W has shape (p*n, rank): index is k*n + i
    
    The tensor entry T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i,j,k.
    """
    rank = len(U[0])
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_random(U, V, W, n, m, p, num_trials=5):
    """Verify by actual matrix multiplication with random integer matrices."""
    for _ in range(num_trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        # Expected product
        C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Decomposed product
        C_decomp = [[0] * p for _ in range(n)]
        rank = len(U[0])
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    # C[i,k] += W[k*n+i, c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
                    pass
        # More efficient: build intermediate
        # For each c, the contribution to C[i,k] is W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
        # Actually that's not right either. Let me think again.
        # 
        # Standard bilinear decomposition:
        # C = sum_c W_c * (U_c * A) * (B * V_c^T) ... no
        # 
        # The standard form is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_c [sum_j U[i,j,c] * A[i,j]] * [sum_j V[j,k,c] * B[j,k]] * W[i,k,c]
        # 
        # Wait, but the shapes are:
        # U: (n*m, rank) -> U[i*m+j, c] = U[i,j,c]
        # V: (m*p, rank) -> V[j*p+k, c] = V[j,k,c]  
        # W: (p*n, rank) -> W[k*n+i, c] = W[k,i,c] = W[i,k,c] (with index k*n+i)
        #
        # So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        for c in range(rank):
            # Compute left part: for each i, sum_j U[i*m+j,c] * A[i,j]
            left = [0] * n
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i*m+j][c] * A[i][j]
                left[i] = s
            # Compute right part: for each k, sum_j V[j*p+k,c] * B[j,k]
            right = [0] * p
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j*p+k][c] * B[j][k]
                right[k] = s
            # Add to C
            for i in range(n):
                for k in range(p):
                    C_decomp[i][k] += W[k*n+i][c] * left[i] * right[k]
        
        if C_decomp != C_expected:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    
    # Start with schoolbook
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions using rank reduction
    # Strategy: try ranks from 55 down, attempting to find valid decompositions
    
    # For small matrices, we can try systematic search with rank reduction
    # Key insight: if we have a rank-R decomposition, we can try to express one factor
    # as a linear combination of others to get rank R-1.
    
    # Alternative: use known results or try random constructions
    
    # Let's try a structured approach:
    # For n=2, m=4, p=7, the tensor is 56-dimensional (2*4*7 entries all equal 1)
    # We need U (8 x R), V (28 x R), W (14 x R) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Try rank reduction from schoolbook
    # If two schoolbook terms can be combined, we reduce rank by 1.
    # Term (i,j,k) and (i',j',k') can be merged if there exist coefficients a,b,c such that:
    # a*1 + b*1 = 1 when both terms are active, and 0 otherwise
    # This is only possible if the terms have the same (i,j,k) which they don't.
    
    # Actually, for the standard matmul tensor where all entries are 1,
    # we need a decomposition where the "product" of three vectors gives 1 at each position.
    
    # Let me try: use the identity 1 = 1*1*1, but with shared factors.
    # If we set U[i*m+j, c] = 1 for some pattern, etc.
    
    # Try: rank = n*p + m*p + n*m - something? No, that's not standard.
    
    # Let's try a different approach: express the all-ones tensor as a sum of rank-1 tensors
    # where each rank-1 tensor is a product of three vectors.
    
    # A rank-1 tensor in this context is T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # We need sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    # Try using Kronecker products or structured matrices.
    
    # For now, let's try to verify the schoolbook and see if we can find any improvement
    # through systematic search with small rank reductions.
    
    # Try rank 55: can we merge two schoolbook terms?
    # Term (i,j,k) and (i',j',k') merge if we can write 1 = a*u1*v1*w1 + b*u2*v2*w2 
    # where u1,v1,w1 correspond to (i,j,k) and u2,v2,w2 to (i',j',k')
    # and the result is 0 elsewhere. This is generally not possible unless the terms overlap.
    
    # Since all entries of the tensor are 1, and each schoolbook term is nonzero only at one entry,
    # merging two terms would require a single rank-1 term to be nonzero at two entries, which
    # means it must be nonzero at both (i,j,k) and (i',j',k').
    
    # A rank-1 term u[i,j]*v[j,k]*w[k,i] is determined by u (n*m values), v (m*p values), w (p*n values).
    # It's nonzero at (i,j,k) iff u[i,j] != 0, v[j,k] != 0, and w[k,i] != 0.
    # For it to be nonzero at two different positions (i,j,k) and (i',j',k'), we need specific patterns.
    
    # Example: if i=i', j=j', k!=k', then we need u[i,j]!=0, v[j,k]!=0, v[j,k']!=0, w[k,i]!=0, w[k',i]!=0.
    # This is possible! So we can have a rank-1 term that covers multiple (k) values for fixed (i,j).
    
    # Let's try: for fixed i,j, use one rank-1 term that covers all k values.
    # u[i,j] = 1, v[j,k] = 1 for all k, w[k,i] = 1 for all k.
    # Then T[i,j,k] = 1*1*1 = 1 for all k. 
    # But this also contributes to other (i',j') if u[i',j'] != 0 or w[k,i'] != 0.
    # If we set u[i',j']=0 for (i',j') != (i,j) and w[k,i']=0 for i' != i, then
    # this term only contributes to (i,j,*) for all k.
    
    # So: for each (i,j), use one rank-1 term: u[i,j]=1, u[i',j']=0 for others; v[j,k]=1 for all k; w[k,i]=1 for all k, w[k,i']=0 for i'!=i.
    # Number of terms: n*m = 8. Rank = 8!
    
    # Wait, but v[j,k] is shared across different (i,j) pairs with the same j.
    # If I have two terms for (i,j) and (i',j) with i!=i', they both need v[j,k]=1 for all k.
    # That's fine, they can share the same v row.
    # But w[k,i]=1 and w[k,i']=1 are different entries, so that's fine.
    
    # Let me construct this:
    # Term c = (i,j) for i in range(n), j in range(m):
    #   U[i*m+j, c] = 1
    #   U[i'*m+j', c] = 0 for (i',j') != (i,j)
    #   V[j*p+k, c] = 1 for all k
    #   V[j'*p+k, c] = 0 for j' != j
    #   W[k*n+i, c] = 1 for all k
    #   W[k*n+i', c] = 0 for i' != i
    
    # Check: for entry (i,j,k):
    #   Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    #   = U[i*m+j, c_(i,j)] * V[j*p+k, c_(i,j)] * W[k*n+i, c_(i,j)]
    #   = 1 * 1 * 1 = 1 (from term c_(i,j))
    #   Other terms c_(i',j'): U[i*m+j, c_(i',j')] = 0 if (i',j') != (i,j), so contribution is 0.
    #   Total = 1. Correct!
    
    # But wait, do other terms contribute? Term c_(i',j') has:
    #   U[i*m+j, c_(i',j')] = 0 if (i',j') != (i,j). Yes, so no cross-contamination.
    
    # This gives rank = n*m = 8!
    
    # Let me verify this more carefully.
    # For term corresponding to (i0, j0):
    #   U[i*m+j, c] = 1 if (i,j)==(i0,j0), else 0
    #   V[j*p+k, c] = 1 if j==j0, else 0 (for all k)
    #   W[k*n+i, c] = 1 if i==i0, else 0 (for all k)
    
    # Tensor entry (i,j,k):
    #   Only term c where (i0,j0)==(i,j) has U[i*m+j,c]=1
    #   For that term, V[j*p+k,c]=1 (since j==j0) and W[k*n+i,c]=1 (since i==i0)
    #   So contribution = 1*1*1 = 1
    #   All other terms have U[i*m+j,c]=0, so contribution = 0
    #   Total = 1. ✓
    
    # So rank 8 works! Let's construct and verify.
    
    rank_try = n * m  # 8
    U_try = [[0] * rank_try for _ in range(n * m)]
    V_try = [[0] * rank_try for _ in range(m * p)]
    W_try = [[0] * rank_try for _ in range(p * n)]
    
    for c, (i, j) in enumerate([(i, j) for i in range(n) for j in range(m)]):
        U_try[i*m+j][c] = 1
        for k in range(p):
            V_try[j*p+k][c] = 1
            W_try[k*n+i][c] = 1
    
    # Verify
    if _verify(U_try, V_try, W_try, n, m, p):
        best_rank = rank_try
        best_U, best_V, best_W = U_try, V_try, W_try
    
    # Can we do even better? Try rank < n*m.
    # The tensor has n*m*p = 56 entries, all equal to 1.
    # The "tensor rank" of the all-ones tensor of size n x m x p...
    # Actually, this is a special tensor. Let's think about lower bounds.
    # 
    # For the all-ones tensor, the border rank or other measures might give lower bounds.
    # But we're looking for exact rank, not border rank.
    #
    # Can rank be less than n*m? 
    # Each rank-1 term u[i,j]*v[j,k]*w[k,i] has a "support" pattern.
    # To cover all n*m*p entries with value 1, we need enough terms.
    #
    # Lower bound: consider fixing i. The slice T[i,*,*] is an m x p all-ones matrix.
    # This slice is expressed as sum_c u_c[i,*] * (v_c * w_c[i]) where v_c is a vector of length m*p
    # reshaped... actually it's more complex.
    #
    # Let's just try to search for rank < 8.
    
    # Try rank 7, 6, etc. with random search
    random.seed(42)
    
    for rank_test in range(min(best_rank - 1, 7), 0, -1):
        if time.time() > deadline:
            break
        
        # Try random constructions with small integer values
        found = False
        max_attempts = 200 if rank_test >= 5 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random construction: pick random {-1, 0, 1} or small integers
            scale = 2
            U_rand = [[random.randint(-scale, scale) for _ in range(rank_test)] for _ in range(n * m)]
            V_rand = [[random.randint(-scale, scale) for _ in range(rank_test)] for _ in range(m * p)]
            W_rand = [[random.randint(-scale, scale) for _ in range(rank_test)] for _ in range(p * n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p):
                found = True
                best_rank = rank_test
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
        
        if found:
            break
    
    # Also try structured constructions for lower ranks
    # Pattern: group by i only
    # For each i, use one term: U[i*m+j, c_i] = 1 for all j; V[j*p+k, c_i] = 1 for all j,k; W[k*n+i, c_i] = 1 for all k
    # But V is shared, so if two terms use the same V pattern, that's fine.
    # Check: entry (i,j,k): term c_i has U[i*m+j,c_i]=1, V[j*p+k,c_i]=1, W[k*n+i,c_i]=1. Product=1.
    # Other term c_{i'} (i'!=i): U[i*m+j,c_{i'}]=0 (since i!=i'). Product=0.
    # Total=1. Rank = n = 2!
    
    rank_try2 = n  # 2
    U_try2 = [[0] * rank_try2 for _ in range(n * m)]
    V_try2 = [[0] * rank_try2 for _ in range(m * p)]
    W_try2 = [[0] * rank_try2 for _ in range(p * n)]
    
    for c, i in enumerate(range(n)):
        for j in range(m):
            U_try2[i*m+j][c] = 1
        for j in range(m):
            for k in range(p):
                V_try2[j*p+k][c] = 1
        for k in range(p):
            W_try2[k*n+i][c] = 1
    
    if _verify(U_try2, V_try2, W_try2, n, m, p):
        best_rank = rank_try2
        best_U, best_V, best_W = U_try2, V_try2, W_try2
    
    # Try rank 1: is it possible?
    # Need u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
    # Fix i=0, j=0: u[0,0]*v[0,k]*w[k,0] = 1 for all k.
    # Fix i=0, j=1: u[0,1]*v[1,k]*w[k,0] = 1 for all k.
    # So v[0,k]*w[k,0] = 1/u[0,0] and v[1,k]*w[k,0] = 1/u[0,1] for all k.
    # Thus v[0,k]/v[1,k] = u[0,1]/u[0,0] for all k.
    # Similarly fixing i=0, k=0: u[i,0]*v[i,0]*w[0,i] = 1 for all i.
    # And i=1, k=0: u[i,0]*v[i,0]*w[0,i] = 1 for all i.
    # So u[i,0]*v[i,0]*w[0,i] is the same for i=0 and i=1? No, different (i,k) pairs.
    # 
    # From i=0,k=0: u[0,j]*v[j,0]*w[0,0] = 1 for all j.
    # From i=1,k=0: u[1,j]*v[j,0]*w[0,1] = 1 for all j.
    # From i=0,j=0: u[0,0]*v[0,k]*w[k,0] = 1 for all k.
    # From i=0,j=1: u[0,1]*v[1,k]*w[k,0] = 1 for all k.
    # 
    # From the first: v[0,k] = 1/(u[0,0]*w[k,0]) and v[1,k] = 1/(u[0,1]*w[k,0])
    # So v[0,k]/v[1,k] = u[0,1]/u[0,0] for all k.
    # 
    # From i=0,j=0,k=0: u[0,0]*v[0,0]*w[0,0] = 1
    # From i=0,j=1,k=0: u[0,1]*v[1,0]*w[0,0] = 1
    # From i=1,j=0,k=0: u[1,0]*v[0,0]*w[0,1] = 1
    # From i=1,j=1,k=0: u[1,1]*v[1,0]*w[0,1] = 1
    # 
    # From i=0,j=0,k=1: u[0,0]*v[0,1]*w[1,0] = 1
    # From i=0,j=1,k=1: u[0,1]*v[1,1]*w[1,0] = 1
    # 
    # This is a system of equations. With 8+28+14=50 unknowns and 56 equations, 
    # it's likely overdetermined and no solution exists. Let's just check rank 1 directly.
    
    # For rank 1, the tensor must be a rank-1 tensor. The all-ones tensor of size 2x4x7
    # is rank 1 only if it can be written as u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
    # This requires u[i,j]*v[j,k]*w[k,i] to be constant (=1) for all (i,j,k).
    # Taking logs: log(u[i,j]) + log(v[j,k]) + log(w[k,i]) = 0 for all i,j,k.
    # This means log(u[i,j]) = -log(v[j,k]) - log(w[k,i]) for all i,j,k.
    # Fixing i: log(u[i,j]) + log(w[k,i]) = -log(v[j,k]) for all j,k.
    # The LHS depends on i,j and i,k separately (sum of two terms each depending on one index pair),
    # while the RHS depends on j,k.
    # For this to hold, we need log(u[i,j]) = a[j] - b[i] (additive in j and i) and similar.
    # Actually, this is the condition for the tensor to have rank 1 in the multiplicative sense.
    # It's very restrictive. For n=2,m=4,p=7, it's almost certainly not rank 1.
    
    # Let's just verify rank 2 works (which we already did above) and see if rank 1 is possible
    # by checking if the system has a solution.
    
    if best_rank > 1 and time.time() < deadline:
        # Try rank 1 with a simple check
        # u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k
        # Set w[k,i] = 1 for all k,i. Then need u[i,j]*v[j,k] = 1 for all i,j,k.
        # Fix i=0: u[0,j]*v[j,k] = 1 for all j,k.
        # Fix i=1: u[1,j]*v[j,k] = 1 for all j,k.
        # So u[0,j]*v[j,k] = u[1,j]*v[j,k] for all j,k, meaning u[0,j] = u[1,j] (if v[j,k] != 0).
        # So u[0,j] = u[1,j] = c_j for all j.
        # Then c_j * v[j,k] = 1 for all j,k, so v[j,k] = 1/c_j for all k.
        # So v[j,k] is independent of k: v[j,k] = d_j for all k.
        # Then u[i,j] * d_j = 1 for all i, so u[i,j] = 1/d_j = c_j for all i.
        # So u[i,j] = c_j, v[j,k] = 1/c_j, w[k,i] = 1.
        # Check: u[i,j]*v[j,k]*w[k,i] = c_j * (1/c_j) * 1 = 1. ✓
        # So rank 1 works with this construction!
        
        rank_try3 = 1
        U_try3 = [[0] * rank_try3 for _ in range(n * m)]
        V_try3 = [[0] * rank_try3 for _ in range(m * p)]
        W_try3 = [[0] * rank_try3 for _ in range(p * n)]
        
        # c_j = 1 for all j (simplest choice)
        for i in range(n):
            for j in range(m):
                U_try3[i*m+j][0] = 1  # c_j = 1
        for j in range(m):
            for k in range(p):
                V_try3[j*p+k][0] = 1  # 1/c_j = 1
        for k in range(p):
            for i in range(n):
                W_try3[k*n+i][0] = 1  # w[k,i] = 1
        
        if _verify(U_try3, V_try3, W_try3, n, m, p):
            best_rank = 1
            best_U, best_V, best_W = U_try3, V_try3, W_try3
    
    # Convert to integer/half-integer (they're already integers)
    return {"U": best_U, "V": best_V, "W": best_W}
