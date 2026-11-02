import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    # Vectorized verification
    U_arr = np.asarray(U)
    V_arr = np.asarray(V)
    W_arr = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_5x5():
    """
    Construct a Strassen-like decomposition for 5x5 matrix multiplication.
    We decompose 5x5 into blocks and use Strassen's algorithm on each block.
    
    Split 5x5 into:
    - Top-left 2x2, top-right 2x3, bottom-left 3x2, bottom-right 3x3
    
    Actually, let's use a simpler approach: use the fact that 5 = 2 + 3
    and apply Strassen to the 2x2 and 3x3 subproblems.
    
    For now, let's use a direct approach with rank reduction.
    """
    # For 5x5, schoolbook rank is 25
    # Try to find a lower rank decomposition
    
    # Use Strassen's algorithm on 4x4 submatrix, handle 5th row/col separately
    # This gives rank = 7 (Strassen for 4x4) + 4 + 4 + 1 = 16
    
    # Actually, let's construct it properly using Kronecker product structure
    
    # The 4x5x5 tensor = T_4x1x1 ⊗ T_1x5x5
    # T_4x1x1 has rank 4 (each output depends on one input)
    # T_1x5x5 has rank 25 (schoolbook) or less with Strassen
    
    # Let's use Strassen for the 5x5 part
    # Strassen for 5x5: decompose into 2x2 and 3x3 blocks
    
    # For 2x2: rank 7 (Strassen)
    # For 3x3: rank 23 (best known) or 27 (schoolbook)
    
    # Let's just use schoolbook for 5x5 for now and see if we can improve
    # Schoolbook 5x5: rank 25
    
    # The total rank would be 4 * 25 = 100 (same as schoolbook)
    # We need a better approach
    
    # Let's try a different decomposition:
    # Split the 4x5 matrix into two 2x5 matrices
    # Split the 5x5 matrix appropriately
    
    # Actually, the Kronecker product approach:
    # T_4x5x5[i,j,k] = sum over r of U[i*5+j, r] * V[j*5+k, r] * W[k*4+i, r]
    
    # If we can write T_4x5x5 as a sum of products of tensors,
    # we can use the rank of each factor
    
    # For the 4x1x1 part: T[i,0,k] = delta[i,k] if i==k, else 0
    # This has rank 4 (one term per i)
    
    # For the 1x5x5 part: T[0,j,k] = delta[j,k] 
    # This has rank 25 (schoolbook)
    
    # But we can do better! Use Strassen for the 5x5 part.
    
    # Let's construct Strassen for 5x5:
    # Split into 2x2 and 3x3 blocks
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # C11 = A11*B11 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (3x2)
    # C22 = A21*B12 + A22*B22 (3x3)
    
    # This doesn't directly help with rank reduction.
    
    # Let's try a different approach: use the known fact that
    # R(4,5,5) <= R(4,4,4) * something
    
    # Actually, let's just implement a greedy search for rank reduction
    # starting from schoolbook
    
    n, m, p = 4, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try to find a lower rank decomposition using structured block decomposition.
    
    Strategy: Decompose the (n,m,p) tensor into smaller blocks and use
    known efficient decompositions for each block, then combine.
    """
    start_time = time.time()
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Strategy 1: Block decomposition
    # Split j (middle dimension) into parts and handle each block
    # For (4,5,5): split j into 2+3
    
    def _block_decompose(n, m1, m2, p, deadline):
        """Decompose (n, m1+m2, p) using (n,m1,p) and (n,m2,p) blocks."""
        # This just concatenates, so rank = R(n,m1,p) + R(n,m2,p)
        # Not helpful unless we have better sub-decompositions
        pass
    
    # Strategy 2: Use the fact that for fixed output row i, we're computing
    # row i of C = row i of A times B
    # C[i,:] = A[i,:] * B
    # This is a (1,5,5) matrix multiplication for each i
    
    # R(1,5,5) = 5 (schoolbook, and it's optimal since the tensor is 
    # the identity-like structure)
    
    # So R(4,5,5) <= 4 * R(1,5,5) = 20? 
    # Wait, that's not how it works. The tensor isn't a direct sum.
    
    # Let me try a different approach: iterative rank reduction
    # Start with schoolbook and try to find linear combinations that reduce rank
    
    current_U = best_U.copy()
    current_V = best_V.copy()  
    current_W = best_W.copy()
    current_rank = best_rank
    
    # Try to find if any column can be expressed as linear combination of others
    # This is a hard problem, so use a heuristic:
    # For small rank reductions, try specific patterns
    
    # Pattern: For 4x5x5, try to use the structure where
    # some products can be shared
    
    # Let's try: group by output column k
    # For each k, we need to compute C[:,k] = A * B[:,k]
    # This is (4,5,1) matrix multiplication
    # R(4,5,1) = 4 (each output row needs 5 multiplications, but 
    # the tensor structure means rank is actually 4*5=20 for schoolbook)
    
    # Actually R(4,5,1): T[i,j,k] with k fixed = 1 for all i,j
    # The tensor is a 4x5 matrix of ones, which has matrix rank 1
    # But tensor rank... for (n,m,1), the tensor is just a matrix, 
    # and tensor rank = matrix rank for the case p=1? No.
    
    # For (n,m,1): T[(i,j),(j,k),(i,k)] with k=0
    # T[(i,j),(j,0),(i,0)] = 1
    # This is a tensor in R^{(n*m) x (m*1) x (1*n)} = R^{nm x m x n}
    # Entry [(i,j), (j,0), (i,0)] = 1
    # This is the standard matrix multiplication tensor for (n,m,1)
    # Its rank is n*m (schoolbook) and it's known that R(n,m,1) = n*m
    # because each output C[i,0] = sum_j A[i,j]*B[j,0] requires 
    # n*m independent terms (they can't be shared)
    
    # So that approach doesn't help.
    
    # Let me try a completely different approach:
    # Use the known result that R(n,m,p) can be bounded using
    # the "Coppersmith-Winograd" style or other algebraic methods
    # But for small dimensions, let's try explicit constructions.
    
    # Construction for (4,5,5):
    # Think of it as (4,4,5) + (4,1,5)
    # R(4,4,5) + R(4,1,5)
    # R(4,1,5) = 4 (each of 4 rows times 1x5 = 4*5=20 schoolbook, 
    # but tensor rank... for (4,1,5), T[(i,0),(0,k),(i,k)] = 1
    # This is R^{4 x 5 x 4} tensor with T[i,k,i'] = delta[i,i']
    # This has rank 4 (one per i) since T = sum_i e_i (in first) 
    # outer (identity in middle) outer e_i (in third)
    # Actually T[i, k, i'] = delta[i,i'] * 1
    # = sum_i (e_i in first) * (all-ones in middle) * (e_i in third)
    # So rank is 4.
    
    # R(4,4,5): Split as (4,4,4) + (4,4,1)
    # R(4,4,4) is the key. Best known is around 49-64.
    # R(4,4,1) = 16 (schoolbook, and it's tight)
    
    # So R(4,5,5) <= R(4,4,5) + 4 <= R(4,4,4) + 16 + 4
    
    # If R(4,4,4) = 49, then R(4,5,5) <= 49 + 20 = 69
    # If R(4,4,4) = 64 (schoolbook), then R(4,5,5) <= 64 + 20 = 84
    
    # Let me try to construct an explicit R(4,4,4) decomposition
    # using Strassen-like methods.
    
    # For 4x4x4, use the following known decomposition:
    # Split 4 = 2 + 2, apply Strassen to each 2x2 block
    # But that gives R(4,4,4) <= 4 * R(2,2,2) = 4*7 = 28? 
    # No, that's not how it works.
    
    # Actually, for (4,4,4) with 2+2 split:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen on each 2x2 multiplication:
    # Each 2x2 multiplication needs rank 7
    # But we have cross terms that complicate things
    
    # The standard approach: 
    # R(4,4,4) <= R(2,2,2) * 4 + extra for cross terms
    # This isn't straightforward.
    
    # Let me just try to implement a simple rank reduction search.
    
    # Heuristic: Try to find if the current decomposition has redundant terms
    # by checking if any term's contribution is linearly dependent on others.
    
    # For now, return schoolbook as a fallback
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try structured decomposition: split middle dimension
    # R(4,5,5) <= R(4,2,5) + R(4,3,5)
    # R(4,2,5) <= R(4,2,2) + R(4,2,3)
    # R(4,2,2): schoolbook = 16, but maybe lower
    # R(4,2,3): schoolbook = 24
    
    # Actually, let's try a direct block construction
    # Split j into 2+3:
    # C[:,0:2] = A[:,:2] * B[0:2,:] + A[:,2:5] * B[2:5,:]  -- NO, this is wrong
    
    # Correct block decomposition:
    # A is 4x5, B is 5x5
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{4} A[i,j] * B[j,k]
    
    # First part: (4,2,5) matrix multiplication
    # Second part: (4,3,5) matrix multiplication
    
    # R(4,2,5) schoolbook = 40
    # R(4,3,5) schoolbook = 60
    # Total: 100 (same as schoolbook, no improvement)
    
    # Need better sub-decompositions.
    
    # For (4,2,5): Think of it as (4,2,1) + (4,2,1) + (4,2,1) + (4,2,1) + (4,2,1)
    # R(4,2,1) = 8 (schoolbook, tight for this case)
    # So R(4,2,5) <= 5 * 8 = 40 (no improvement)
    
    # Let me try a different approach: use the tensor structure directly.
    # The tensor T[i,j,k] for (4,5,5) has T[i,j,k] = 1 for all i,j,k
    # (in the appropriate indexing).
    
    # Wait, I need to be more careful. The tensor is:
    # T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j', else 0
    # So T[i,j,k] in my simplified notation means:
    # The term A[i,j] * B[j,k] contributes to C[i,k]
    
    # Each term is a rank-1 tensor:
    # u_r[i*m+j] = 1, v_r[j*p+k] = 1, w_r[k*n+i] = 1
    # for the specific (i,j,k)
    
    # The sum over all (i,j,k) gives the full tensor.
    
    # To reduce rank, we need to find linear combinations.
    
    # Key insight: If we have two terms (i1,j1,k1) and (i2,j2,k2) where
    # i1=i2, j1=j2, but k1 != k2, then:
    # Term1: u has 1 at (i1,j1), v has 1 at (j1,k1), w has 1 at (k1,i1)
    # Term2: u has 1 at (i1,j1), v has 1 at (j1,k2), w has 1 at (k2,i1)
    
    # These can't be directly merged because they contribute to different outputs.
    
    # However, if i1=i2, j1!=j2, k1=k2:
    # Term1: u at (i1,j1), v at (j1,k1), w at (k1,i1)
    # Term2: u at (i1,j2), v at (j2,k1), w at (k1,i1)
    # Same w vector! So we could potentially combine:
    # (u1+u2) * v_avg * w + correction terms
    
    # This is the basis for Strassen-like methods.
    
    # Let me try to implement a basic version:
    # Group terms by (i,k) pair (output position)
    # For each (i,k), there are m=5 terms (one for each j)
    # These 5 terms all have the same w vector (at position k*n+i)
    # So we can write:
    # sum_j A[i,j]*B[j,k] * w_{k,i}
    # = (sum_j u_{i,j} * v_{j,k}) * w_{k,i}  -- but this is still rank 5
    
    # Unless we can express sum_j u_{i,j} * v_{j,k} with fewer terms.
    # This is exactly the (1,5,1) matrix multiplication problem!
    # R(1,5,1) = 5 (schoolbook, and it's tight)
    
    # So for each (i,k), we need rank 5, and there are 4*5=20 output positions.
    # Total rank = 20 * 5 = 100 (back to schoolbook)
    
    # The only way to reduce is to share terms across different (i,k) pairs.
    # This requires the u and v vectors to overlap in useful ways.
    
    # Try rank reduction via linear dependency search
    if time.time() < deadline - 20:
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        # Greedy rank reduction: try to remove one column at a time
        # For each column r, check if it can be expressed as linear combination
        # of other columns while preserving the decomposition
        
        # This is expensive, so limit the number of attempts
        max_removal_attempts = 200
        attempts = 0
        
        for _ in range(max_removal_attempts):
            if time.time() > deadline - 15:
                break
            if current_rank <= 1:
                break
            
            # Try to remove a random column
            r = np.random.randint(0, current_rank)
            attempts += 1
            
            # Check if column r can be removed
            # The constraint is that for all (i,j,k):
            # sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] == 1
            # AND U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 0 for all (i,j,k)
            # The second condition means the column must be "inactive"
            # which only happens if U, V, or W has zeros in that column.
            
            # For schoolbook, each column has exactly one non-zero in each of U, V, W.
            # So we can't simply remove a column.
            
            # Instead, try to merge two columns:
            # Replace columns r1, r2 with a single column r_new
            # such that the tensor decomposition is preserved.
            
            if current_rank < 2:
                break
            
            r1 = np.random.randint(0, current_rank)
            r2 = np.random.randint(0, current_rank)
            if r1 == r2:
                continue
            
            # Try to merge r1 and r2
            # New column: U_new = U[:,r1] + U[:,r2], etc.
            # This doesn't work directly because the products don't add up.
            
            # A valid merge: if we can find alpha, beta such that
            # (alpha*u1 + beta*u2) * (alpha*v1 + beta*v2) * (alpha*w1 + beta*w2)
            # = u1*v1*w1 + u2*v2*w2 + cross terms that cancel
            # This is generally not possible.
            
            # For now, just continue
            pass
        
        # Try a different approach: use the structure of the problem
        # For (4,5,5), try to use a 4x4x4 decomposition for the main part
        # and handle the extra row/column separately.
        
        # Construct: (4,5,5) = (4,4,5) + (4,1,5)
        # (4,4,5) = (4,4,4) + (4,4,1)
        # (4,1,5) has rank 4
        # (4,4,1) has rank 16
        # (4,4,4): use Strassen-like method
        
        # For (4,4,4), use the following construction:
        # Split 4 = 2+2
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Each 2x2 multiplication has rank 7 (Strassen)
        # But we have 8 sub-multiplications (4 diagonal + 4 cross)
        # Using Strassen on each: 8 * 7 = 56
        # Plus we need to combine the results, which adds more terms.
        
        # Actually, the standard block Strassen for 4x4:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 matrix multiplication (rank 7)
        # Total: 7 * 7 = 49 terms
        
        # So R(4,4,4) <= 49
        
        # Let me construct this explicitly!
        if time.time() < deadline - 10:
            # Build (4,4,4) decomposition using Strassen
            n_sub, m_sub, p_sub = 4, 4, 4
            rank_sub = 49
            
            # Strassen for 4x4x4
            # Indices: i,j in {0,1,2,3} for A (4x4)
            # j,k in {0,1,2,3} for B (4x4)
            # i,k in {0,1,2,3} for C (4x4)
            
            # Block structure:
            # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
            # A21 = A[2:4,0:2], A22 = A[2:4,2:4]
            # Similarly for B, C
            
            # Strassen intermediate matrices:
            # M1 = (A11+A22) * (B11+B22)  -- 2x2 multiplication
            # M2 = (A21+A22) * B11          -- 2x2 multiplication
            # M3 = A11 * (B12-B22)          -- 2x2 multiplication
            # M4 = A22 * (B21-B11)          -- 2x2 multiplication
            # M5 = (A11+A12) * B22          -- 2x2 multiplication
            # M6 = (A21-A11) * (B11+B12)    -- 2x2 multiplication
            # M7 = (A12-A22) * (B21+B22)    -- 2x2 multiplication
            
            # Each Mi is computed using Strassen for 2x2 (rank 7)
            # Total rank: 7 * 7 = 49
            
            # Now build the (4,4,4) decomposition
            U_444 = np.zeros((16, 49))
            V_444 = np.zeros((16, 49))
            W_444 = np.zeros((16, 49))
            
            # Strassen for 2x2: rank 7
            # For 2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]]
            # C11 = (a+d)(e+h) - (d-b)(g+h)
            # C12 = (a+b)(h) - (a+d)(e+h) + ... 
            # Standard Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)(e)
            # M3 = (a-b)(f+h)
            # M4 = (d-b)(g+h)  -- wait, let me use the standard form
            # 
            # M1 = (a+d)(e+h)
            # M2 = (c+d)(e)
            # M3 = (a-b)(f+h)
            # M4 = (d-b)(g+h)  -- this is wrong
            # 
            # Correct Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)(e)
            # M3 = (a-b)(f+h)
            # M4 = (d-b)(g+h)  -- no
            # 
            # Let me use the standard:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)(e)
            # M3 = (a-b)(f+h)
            # M4 = (d-b)(g+h)  -- I keep getting confused
            # 
            # Standard Strassen for 2x2:
            # M1 = (a + d) * (e + h)
            # M2 = (c + d) * e
            # M3 = (a - b) * (f + h)
            # M4 = (d - b) * (g + h)  -- NO
            # 
            # Actually:
            # M1 = (a + d) * (e + h)
            # M2 = (c + d) * e
            # M3 = (a - b) * (f + h)
            # M4 = (d - b) * (g + h)  -- still wrong
            # 
            # Let me just look it up mentally:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)(e)
            # M3 = (a-b)(f+h)
            # M4 = (d-b)(g+h)  -- I think M4 = (d-b)(g+h) is wrong
            # 
            # Correct:
            # M1 = (a + d) * (e + h)
            # M2 = (c + d) * e
            # M3 = (a - b) * (f + h)
            # M4 = (d - b) * (g + h)  -- Let me just use a known correct set
            # 
            # From Wikipedia:
            # M1 = (a + d) * (e + h)
            # M2 = (c + d) * e
            # M3 = (a - b) * (f + h)
            # M4 = (d - b) * (g + h)  -- This gives:
            # C11 = M1 - M4 + M3 - M2  -- let me verify
            # C12 = M4 + M5
            # C21 = M2 + M6
            # C22 = M1 - M2 + M5 - M7
            # 
            # Where:
            # M5 = (a + b) * h
            # M6 = (c - a) * (e + f)
            # M7 = (b - d) * (g + h)
            # 
            # Let me verify C11:
            # M1 - M4 + M3 - M2 = (a+d)(e+h) - (d-b)(g+h) + (a-b)(f+h) - (c+d)(e)
            # This doesn't simplify to ae+bg easily.
            # 
            # I think I have the Strassen formulas wrong. Let me use a simpler approach.
            # 
            # For now, let me just use schoolbook for 2x2 (rank 4) instead of Strassen (rank 7)
            # This gives R(4,4,4) <= 7 * 4 = 28, which is worse than Strassen's 49.
            # Wait, 7 Strassen steps each of rank 7 = 49, vs 7 schoolbook steps each of rank 4 = 28.
            # But 28 < 49, so schoolbook on 2x2 blocks is better? That can't be right.
            # 
            # Oh wait, I'm confusing myself. The 7 intermediate matrices M1-M7 are each 2x2 multiplications.
            # If I compute each using schoolbook (rank 4), total rank = 7 * 4 = 28.
            # If I compute each using Strassen (rank 7), total rank = 7 * 7 = 49.
            # So schoolbook is better here? That seems wrong.
            # 
            # The issue is that Strassen reduces the number of multiplications from 4 to 7 for 2x2,
            # but each "multiplication" in Strassen is still a 2x2 multiplication.
            # So using Strassen for the 4x4 case means 7 Strassen steps, each requiring 7 scalar mults,
            # total 49. Using schoolbook for the 4x4 case directly is 16 scalar mults.
            # 
            # I think I'm overcomplicating this. Let me just use the direct schoolbook for (4,4,4)
            # which has rank 16, and then add the overhead for the (4,4,1) and (4,1,5) parts.
            # 
            # R(4,5,5) <= R(4,4,4) + R(4,4,1) + R(4,1,5) = 16 + 16 + 4 = 36
            # 
            # Wait, that's a huge improvement from 100! But is this decomposition valid?
            # 
            # The issue is that R(4,5,5) is NOT simply R(4,4,4) + R(4,4,1) + R(4,1,5).
            # The tensor decomposition doesn't decompose additively like that in general.
            # 
            # The correct bound would be:
            # R(4,5,5) <= R(4,4,5) + R(4,1,5)
            # R(4,4,5) <= R(4,4,4) + R(4,4,1)
            # 
            # But even this isn't straightforward because the tensor for (4,4,5) isn't
            # simply the sum of tensors for (4,4,4) and (4,4,1).
            
            # Let me try a different, more concrete approach.
            # I'll directly construct a decomposition for (4,5,5) with rank < 100.
            
            # Approach: Use the "padding" technique
            # Pad (4,5,5) to (8,8,8) or similar, use known decomposition, then restrict.
            # This is complex.
            
            # Simpler approach: Just try to find any rank reduction via optimization.
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
