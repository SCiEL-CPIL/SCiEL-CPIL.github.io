<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Uses fraction-based Gaussian elimination to find the minimal rank by:
    1. Constructing the matrix representation of the matrix multiplication tensor
    2. Computing its rank over the rationals
    3. Extracting a decomposition at that rank
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,k,j] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    A decomposition of rank r means:
    T[i,k,j] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    We can fix W[i,k,c] and solve for U and V, or vice versa.
    Alternatively, we can use the approach of viewing this as a bilinear map.
    
    Actually, for exact minimal rank, we need to be more careful. Let's use the 
    standard approach: the matrix multiplication tensor M_{n,m,p} has a well-known 
    lower bound. For 2x4 x 4x6, the schoolbook rank is 2*4*6 = 48.
    
    We can potentially do better. Let's use the fact that we can sometimes share
    computations. One approach: use the "border rank" or try to find a decomposition
    with fewer than 48 terms.
    
    A simpler approach that often works: since we need integer/half-integer entries,
    we can try to find a decomposition by setting up a linear system.
    
    Let's think of it this way: we want to find matrices U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all i,j,k:
    T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This is a trilinear system which is hard. Instead, we can use a different formulation.
    
    Alternative: We can use the fact that the rank of the matrix multiplication tensor 
    can be bounded. For small cases, we can try randomized approaches or use known bounds.
    
    Actually, let's use a cleaner approach: construct the "slice" matrices and compute
    the rank of a certain associated matrix.
    
    For matrix multiplication, we can associate a matrix M of size (n*p) x (n*p) where
    M[i*p+k, i'*p+k'] = ... hmm, this isn't quite right.
    
    Let me use a standard result: the rank of the matrix multiplication tensor M_{n,m,p}
    is at least n*p + m*p + n*m - n - m - p + 1 (this is a weak bound).
    
    For n=2, m=4, p=6: lower bound = 12 + 24 + 8 - 2 - 4 - 6 + 1 = 33
    
    But we can do better with known algorithms. Let's try a direct approach:
    
    We'll attempt to find a decomposition of rank r by solving a linear system.
    We fix U and V to have random structure, then solve for W, and check if the 
    decomposition is exact. We iterate over possible ranks.
    
    Actually, a more reliable approach: use the "pseudorandom" method or just try
    to find a good decomposition. Given the small size, we can afford to be exhaustive
    in a smart way.
    
    Let's use the following approach:
    1. Start with rank 48 (schoolbook)
    2. Try to reduce by merging terms where possible
    3. Or, use known lower bounds and try to achieve them
    
    For practical purposes, let's implement a randomized search for low-rank decompositions:
    - For a candidate rank r, randomly initialize U and V with small integers
    - Set up the linear system for W: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if (i,j,k) is a valid triple, 0 otherwise
    - This is still trilinear, so it's not linear in W alone.
    
    Better approach: For a fixed set of U and V, the condition becomes linear in W.
    Specifically, for each (i,j,k), we need:
    W[k*n+i, :] dot (U[i*m+j, :] * V[j*p+k, :]) = T[i,k,j]
    
    Where (U[i*m+j, :] * V[j*p+k, :]) is the element-wise product (Hadamard product).
    
    So we can set up a linear system A_w * w = b where:
    - Each row corresponds to a triple (i,j,k)
    - The row is the Hadamard product of U[i*m+j,:] and V[j*p+k,:]
    - The RHS is 1
    
    Wait, that's not quite right either. Let me re-examine.
    
    We need: for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k]
    
    where delta[i,j,k] = 1 (since we're multiplying, each product term contributes exactly once).
    
    This is a system that's linear in W for fixed U, V. So:
    For each (i,k), let's group by (i,k). The equation for C[i,k] is:
    sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Hmm, this is still coupled. Let me think differently.
    
    Actually, the standard approach for matrix multiplication tensor decomposition is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A decomposition uses "gates" where each gate c computes some linear combination.
    
    For our purposes, let's just use the schoolbook decomposition but see if we can
    optimize. Given the constraints, the minimal rank for 2x4 x 4x6 multiplication
    is likely 33 or nearby based on the lower bound.
    
    Let's implement a proper search: try ranks from the lower bound up to 48,
    and for each rank, try to find U, V, W that satisfy the constraints.
    
    Since this is computationally intensive, let's use a greedy/heuristic approach
    or just return the schoolbook if we can't find better quickly.
    
    Actually, given the 120s timeout, let's be pragmatic: implement the schoolbook
    decomposition as a fallback, and try a simple optimization.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a better decomposition using a linear system approach
    # We'll use the fact that we can fix W to be the identity-like structure
    # and solve for U and V, or use a randomized approach.
    
    # For now, let's use a known good decomposition if available,
    # otherwise fall back to schoolbook.
    
    # A simple optimization: if m is small, we can sometimes share terms.
    # For 2x4 x 4x6, let's just use schoolbook as it's guaranteed to work.
    
    rank = rank_schoolbook
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Uses the matrix multiplication tensor's matrix representation approach:
    - The tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k] can be represented
    - We construct the (n*p) x (n*p) matrix M where M[(i,k),(i',k')] represents 
      the structure of the tensor
    - Compute exact rank using fraction-based Gaussian elimination
    - Extract a decomposition at that rank
    """
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor M_{n,m,p} has a well-known structure.
    # We can represent it as a matrix of size (n*p) x (n*p) where:
    # M[(i,k), (i',k')] = delta_{i,i'} * delta_{k,k'} if we view it appropriately
    # 
    # Actually, the standard approach: the tensor T has indices (i, j, k) where
    # i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1 for all valid indices.
    # 
    # A rank-r decomposition means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    #
    # This is equivalent to saying that the tensor, when viewed as a matrix in 
    # appropriate ways, has rank at most r.
    #
    # Key insight: For matrix multiplication, we can use the "matrix slicing" approach.
    # The tensor M_{n,m,p} can be seen as a matrix of size (n*p) x (n*p) by fixing
    # the middle index j and summing. But more precisely, we use the fact that:
    #
    # The rank of the matrix multiplication tensor equals the rank of a certain 
    # associated matrix. Specifically, we can construct a matrix A of size (n*p) x (n*p)
    # where A[(i,k), (i',k')] = sum_j (1 if i=i' and k=k' else 0) = delta_{i,i'}*delta_{k,k'}
    #
    # Wait, that's just the identity matrix of size n*p, which has rank n*p = 12.
    # That can't be right for the tensor rank.
    #
    # Let me reconsider. The correct approach for finding the rank of the matrix 
    # multiplication tensor:
    #
    # The tensor T in V^* ⊗ V^* ⊗ V^* where V = R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # has entries T[(i,j), (j,k), (i,k)] = 1
    #
    # A rank-1 tensor is U ⊗ V ⊗ W where U[i,j], V[j,k], W[i,k]
    #
    # The rank of T is the minimum number of rank-1 tensors that sum to T.
    #
    # For small cases, we can compute this exactly by:
    # 1. Viewing T as a matrix of size (n*m * m*p) x (p*n) or similar
    # 2. Computing its rank
    #
    # Actually, the standard result: the rank of M_{n,m,p} is at least 
    # max(n*p, m*p, n*m) and at most n*m*p.
    #
    # For exact computation, we can use the following:
    # Flatten the tensor into a matrix. T has shape (n, m, p) with T[i,j,k]=1.
    # 
    # One way to compute the rank: consider T as a matrix of size (n*p) x (m*p)
    # by grouping (i,k) as rows and (j,k') as columns... no, that doesn't work.
    #
    # Let me use a different approach. The tensor M_{n,m,p} can be written as:
    # T[i,j,k] = 1 for all i,j,k
    #
    # This is actually the all-ones tensor of shape (n, m, p)!
    #
    # Wait, no. For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k],
    # the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # So T is indeed the all-ones tensor of shape (n, m, p).
    #
    # The rank of the all-ones tensor of shape (n, m, p) is... 
    # Actually, this is not quite right either. Let me be more careful.
    #
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where e_i are basis vectors. So M_{n,m,p} is the tensor with all entries = 1.
    #
    # Now, the rank of this tensor: we need to write it as a sum of rank-1 tensors.
    # A rank-1 tensor in this space has the form u ⊗ v ⊗ w where u in R^n, v in R^m, w in R^p.
    #
    # The rank of the all-ones tensor of shape (n,m,p) is actually min(n,m,p) if we're 
    # talking about the usual tensor rank? No, that's for the specific case.
    #
    # Actually, I recall that the rank of the all-ones tensor of shape (n,m,p) is 
    # at least max(n,m,p) and the exact value depends on the dimensions.
    #
    # For our case n=2, m=4, p=6, let's compute it properly.
    #
    # We can compute the rank by converting the tensor to a matrix and computing 
    # its rank. One standard way: reshape the tensor to a matrix of size (n*m) x (p) 
    # or (n) x (m*p) or (n*p) x (m), and compute the rank. But this gives a lower bound.
    #
    # The actual tensor rank can be higher. To compute it exactly, we can use 
    # fraction-based Gaussian elimination on a suitable matrix representation.
    #
    # Here's the approach: The tensor T of shape (n,m,p) with T[i,j,k]=1 can be 
    # viewed as a matrix of size (n*p) x (n*p) by the following mapping:
    # 
    # Actually, let me use the correct method. The rank of a 3-tensor T of shape 
    # (n,m,p) can be found by considering it as a matrix of size (n*m) x (p) for 
    # each fixed index, but that gives different lower bounds.
    #
    # For exact rank computation, we can use the following theorem:
    # The rank of T is the minimum r such that there exist matrices U (n x r), 
    # V (m x r), W (p x r) such that T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c].
    #
    # This is a non-linear problem in general. However, for the all-ones tensor,
    # we can use a simpler approach.
    #
    # Let's just compute it by trying different ranks and checking feasibility.
    # For small n,m,p, this is tractable.
    
    # For the all-ones tensor of shape (2, 4, 6), let's find the minimal rank.
    # We know that rank >= max(n, m, p) = 6.
    # 
    # Let's try to find a decomposition of rank r for increasing r.
    
    def can_decompose(n, m, p, r):
        """Check if the all-ones tensor of shape (n,m,p) has rank <= r."""
        # We need to find U (n x r), V (m x r), W (p x r) such that
        # sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all i,j,k
        #
        # This is equivalent to: for each i,j,k, the sum is 1.
        #
        # Let's fix U and V, then solve for W. For fixed U and V, the condition
        # becomes: for each i,j,k: sum_c U[i,c]*V[j,c]*W[k,c] = 1
        #
        # This is linear in W. Let A be a matrix of size (n*m) x (p*r) where
        # A[(i,j), (k,c)] = U[i,c]*V[j,c]
        # Then we need A * w = 1 where w is a vector of size p*r.
        #
        # But we have freedom in choosing U and V, so this is still non-linear.
        #
        # Alternative: use the fact that for the all-ones tensor, there's a known
        # minimal rank. For shape (n,m,p), the rank is min(n,m,p) if we allow
        # real entries? No, that's not right.
        #
        # Let me just try a direct approach: assume U, V, W have a specific structure.
        # For the all-ones tensor, one decomposition of rank r uses:
        # U[i,c] = 1 for c <= n, 0 otherwise (if r >= n)
        # V[j,c] = 1 for c <= m, 0 otherwise (if r >= m)  
        # W[k,c] = 1 for c <= p, 0 otherwise (if r >= p)
        # But this only works if r >= max(n,m,p) and gives sum = min(r, n, m, p)
        # which equals 1 only if min(n,m,p) = 1.
        #
        # Better approach: use the fact that 1 = sum_c a_c * b_c * c_c
        # We can set up a system. Let's use a randomized approach with exact arithmetic.
        
        import random
        from fractions import Fraction
        
        for attempt in range(100):
            # Randomly initialize U and V with fractions
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m)]
            
            # Build the linear system for W
            # For each (i, j, k), we need: sum_c U[i,c] * V[j,c] * W[k,c] = 1
            # Group by k: for each k, we have n*m equations in r unknowns (W[k,0..r-1])
            # Actually, for each k, the equations are: for all i,j: sum_c U[i,c]*V[j,c]*W[k,c] = 1
            # This is n*m equations in r unknowns for each k.
            
            # Let's set up the system for all k simultaneously.
            # Total equations: n*m*p, unknowns: p*r
            # Row (i,j,k): A[(i,j,k), (k',c)] = U[i,c]*V[j,c] if k'=k, else 0
            
            # This is a sparse system. Let's solve it using Gaussian elimination.
            
            num_eqs = n * m * p
            num_unk = p * r
            
            if num_unk < num_eqs:
                return False  # Underdetermined system, but might still have solutions
            
            # Build the augmented matrix
            # A is num_eqs x num_unk, b is num_eqs
            A = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        for c in range(r):
                            unk_idx = k * r + c
                            A[eq_idx][unk_idx] = U[i][c] * V[j][c]
            
            # Gaussian elimination
            rank_A = 0
            for col in range(num_unk):
                # Find pivot
                pivot_row = None
                for row in range(rank_A, num_eqs):
                    if A[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row is None:
                    continue
                
                # Swap rows
                A[rank_A], A[pivot_row] = A[pivot_row], A[rank_A]
                b[rank_A], b[pivot_row] = b[pivot_row], b[rank_A]
                
                # Eliminate
                pivot = A[rank_A][col]
                for row in range(rank_A + 1, num_eqs):
                    if A[row][col] != 0:
                        factor = A[row][col] / pivot
                        for c2 in range(col, num_unk):
                            A[row][c2] -= factor * A[rank_A][c2]
                        b[row] -= factor * b[rank_A]
                
                rank_A += 1
                if rank_A == num_unk:
                    break
            
            # Check consistency
            consistent = True
            for row in range(rank_A, num_eqs):
                if b[row] != 0:
                    consistent = False
                    break
            
            if consistent:
                # Find a solution (set free variables to 0)
                W_flat = [Fraction(0) for _ in range(num_unk)]
                for row in range(rank_A - 1, -1, -1):
                    # Find the leading column
                    lead_col = None
                    for c in range(num_unk):
                        if A[row][c] != 0:
                            lead_col = c
                            break
                    if lead_col is not None:
                        W_flat[lead_col] = b[row]
                        for c in range(lead_col + 1, num_unk):
                            W_flat[lead_col] -= A[row][c] * W_flat[c]
                        W_flat[lead_col] /= A[row][lead_col]
                
                # Verify the solution
                W = [[W_flat[k * r + c] for c in range(r)] for k in range(p)]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(r):
                                s += U[i][c] * V[j][c] * W[k][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return True
        
        return False
    
    # Try ranks from the lower bound up to the schoolbook rank
    lower_bound = max(n, m, p)
    schoolbook_rank = n * m * p
    
    found_rank = None
    for r in range(lower_bound, schoolbook_rank + 1):
        if can_decompose(n, m, p, r):
            found_rank = r
            break
    
    if found_rank is None:
        found_rank = schoolbook_rank
    
    # Now construct the actual decomposition
    # We need to find U, V, W explicitly
    # Use the same approach as above but keep track of the solution
    
    def find_decomposition(n, m, p, r):
        """Find explicit U, V, W for rank-r decomposition of all-ones tensor."""
        import random
        from fractions import Fraction
        
        for attempt in range(1000):
            # Randomly initialize U and V
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m)]
            
            # Build and solve the linear system for W
            num_eqs = n * m * p
            num_unk = p * r
            
            if num_unk < num_eqs:
                continue
            
            A = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        for c in range(r):
                            unk_idx = k * r + c
                            A[eq_idx][unk_idx] = U[i][c] * V[j][c]
            
            # Gaussian elimination with back substitution
            augmented = [A[i] + [b[i]] for i in range(num_eqs)]
            rank_A = 0
            pivot_cols = []
            
            for col in range(num_unk):
                pivot_row = None
                for row in range(rank_A, num_eqs):
                    if augmented[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row is None:
                    continue
                
                augmented[rank_A], augmented[pivot_row] = augmented[pivot_row], augmented[rank_A]
                pivot_cols.append(col)
                
                pivot = augmented[rank_A][col]
                for c2 in range(col + 1, num_unk + 1):
                    augmented[rank_A][c2] /= pivot
                augmented[rank_A][col] = Fraction(1)
                
                for row in range(num_eqs):
                    if row != rank_A and augmented[row][col] != 0:
                        factor = augmented[row][col]
                        for c2 in range(col + 1, num_unk + 1):
                            augmented[row][c2] -= factor * augmented[rank_A][c2]
                        augmented[row][col] = Fraction(0)
                
                rank_A += 1
                if rank_A == num_unk:
                    break
            
            # Check consistency
            consistent = True
            for row in range(rank_A, num_eqs):
                if any(augmented[row][c] != 0 for c in range(num_unk)):
                    if augmented[row][num_unk] != 0:
                        consistent = False
                        break
                elif augmented[row][num_unk] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Find a solution (set free variables to 0)
            W_flat = [Fraction(0) for _ in range(num_unk)]
            for row in range(rank_A):
                # Find leading column
                lead_col = None
                for c in range(num_unk):
                    if augmented[row][c] != 0:
                        lead_col = c
                        break
                if lead_col is not None:
                    W_flat[lead_col] = augmented[row][num_unk]
            
            # Verify
            W = [[W_flat[k * r + c] for c in range(r)] for k in range(p)]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i][c] * V[j][c] * W[k][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        
        return None
    
    U_frac, V_frac, W_frac = find_decomposition(n, m, p, found_rank)
    
    if U_frac is None:
        # Fall back to schoolbook
        rank = schoolbook_rank
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
    
    # Convert fractions to integers/half-integers
    # The entries should already be fractions; convert to float if needed
    rank = found_rank
    
    # Build U, V, W in the required format
    # U: (n*m) x rank, where U[i*m+j, c] = U_frac[i][c]
    # V: (m*p) x rank, where V[j*p+k, c] = V_frac[j][c] 
    # Wait, V should have indices (j, k) but V_frac has indices (j, c)
    # Let me re-examine the indexing.
    #
    # In our decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # But the required format is:
    # U: (n*m) x rank, U[i*m+j, c]
    # V: (m*p) x rank, V[j*p+k, c]
    # W: (p*n) x rank, W[k*n+i, c]
    #
    # And the condition is: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So for the all-ones tensor, T[i,k,j] = 1 for all i,j,k.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is different from what we solved! Let me redo this.
    
    # Actually, I made an error. Let me restart with the correct indexing.
    # The tensor is T[i,j,k] = 1 where i indexes A rows, j indexes A cols/B rows, k indexes B cols.
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So U has (n*m) rows indexed by (i,j), V has (m*p) rows indexed by (j,k), 
    # W has (p*n) rows indexed by (k,i).
    
    def find_decomposition_correct(n, m, p, r):
        """Find explicit U, V, W for rank-r decomposition with correct indexing."""
        import random
        from fractions import Fraction
        
        for attempt in range(500):
            # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m * p)]
            
            # Build linear system for W
            # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Unknowns: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
            # Total unknowns: p * n * r
            # Total equations: n * m * p
            
            num_eqs = n * m * p
            num_unk = p * n * r
            
            if num_unk < num_eqs:
                continue
            
            A = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        for c in range(r):
                            unk_idx = (k * n + i) * r + c
                            A[eq_idx][unk_idx] = U[i * m + j][c] * V[j * p + k][c]
            
            # Gaussian elimination
            augmented = [A[i] + [b[i]] for i in range(num_eqs)]
            rank_A = 0
            pivot_cols = []
            
            for col in range(num_unk):
                pivot_row = None
                for row in range(rank_A, num_eqs):
                    if augmented[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row is None:
                    continue
                
                augmented[rank_A], augmented[pivot_row] = augmented[pivot_row], augmented[rank_A]
                pivot_cols.append(col)
                
                pivot = augmented[rank_A][col]
                for c2 in range(col + 1, num_unk + 1):
                    augmented[rank_A][c2] /= pivot
                augmented[rank_A][col] = Fraction(1)
                
                for row in range(num_eqs):
                    if row != rank_A and augmented[row][col] != 0:
                        factor = augmented[row][col]
                        for c2 in range(col + 1, num_unk + 1):
                            augmented[row][c2] -= factor * augmented[rank_A][c2]
                        augmented[row][col] = Fraction(0)
                
                rank_A += 1
                if rank_A == num_unk:
                    break
            
            # Check consistency
            consistent = True
            for row in range(rank_A, num_eqs):
                row_sum = Fraction(0)
                for c in range(num_unk):
                    if augmented[row][c] != 0:
                        row_sum = Fraction(1)
                        break
                if row_sum != 0 or augmented[row][num_unk] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Find solution
            W_flat = [Fraction(0) for _ in range(num_unk)]
            for row in range(rank_A):
                lead_col = None
                for c in range(num_unk):
                    if augmented[row][c] != 0:
                        lead_col = c
                        break
                if lead_col is not None:
                    W_flat[lead_col] = augmented[row][num_unk]
            
            W = [[W_flat[(k * n + i) * r + c] for c in range(r)] for k in range(p) for i in range(n)]
            # Reshape W to (p*n) x r
            W_matrix = [[W_flat[(k * n + i) * r + c] for c in range(r)] for k in range(p) for i in range(n)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W_matrix[k * n + i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W_matrix
        
        return None
    
    U_frac, V_frac, W_frac = find_decomposition_correct(n, m, p, found_rank)
    
    if U_frac is None:
        # Fall back to schoolbook
        rank = schoolbook_rank
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
    
    # Convert to integer or half-integer representation
    # Fractions are already exact; convert to float for output
    def frac_to_float(f):
        return float(f)
    
    U_out = [[frac_to_float(U_frac[i][c]) for c in range(found_rank)] for i in range(n * m)]
    V_out = [[frac_to_float(V_frac[j][c]) for c in range(found_rank)] for j in range(m * p)]
    W_out = [[frac_to_float(W_frac[k][c]) for c in range(found_rank)] for k in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
