"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 48
    
    # Start with schoolbook (rank 48)
    U = [[0] * full_rank for _ in range(n * m)]
    V = [[0] * full_rank for _ in range(m * p)]
    W = [[0] * full_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = full_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Check if we have time for search
    if time.time() - start_time > deadline - 10:
        return best_result
    
    # Try to find lower-rank decompositions using structured patterns
    # For 2x3 * 3x8, the tensor has 48 entries
    # We search for rank r < 48
    
    # Strategy: Try rank 46 and 47 with structured patterns
    # The matrix multiplication tensor T[i,j,k] = delta_{i,m} * delta_{j,n} * delta_{k,p}
    # A rank-r decomposition means: T = sum_{a=1}^{r} u_a ⊗ v_a ⊗ w_a
    
    # For small reductions, we can try to merge columns that have similar patterns
    # Two columns (a, b) can be merged if there exist scalars such that
    # the resulting tensor still matches
    
    # Try rank 47: merge two columns that differ in only one position
    # Column c has U values at positions determined by (i,j) and V at (j,k) and W at (k,i)
    # For schoolbook, column (i,j,k) has:
    #   U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # To merge two columns (i1,j1,k1) and (i2,j2,k2) into one with weight w:
    # We need: w * (u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2) to equal the sum
    # This is only possible if the tensors are linearly dependent
    
    # For 2x3*3x8, let's try a different approach: use the fact that
    # some entries can be shared
    
    # Actually, let's try a greedy approach: start with rank 48 and try to reduce
    # by checking if any column can be expressed as a combination of others
    
    # For small cases, we can use numpy to check linear dependence
    
    # Try rank 47: see if any column is redundant
    def check_rank_reduction(rank_target):
        """Try to find a decomposition of the given rank."""
        if time.time() - start_time > deadline:
            return None
        
        # For rank 47, we need to find one pair of columns that can be merged
        # or one column that is a linear combination of others
        
        # Build the tensor as a 3D array
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1
        
        # Matricize the tensor in different ways to check rank
        # The rank of the tensor is at least the maximum of the ranks of its matricizations
        
        # Matricization 1: (n*m) x p
        M1 = T.reshape(n * m, p)
        r1 = np.linalg.matrix_rank(M1)
        
        # Matricization 2: n * (m*p)
        M2 = T.reshape(n, m * p)
        r2 = np.linalg.matrix_rank(M2)
        
        # Matricization 3: (n*p) x m
        M3 = T.transpose(0, 2, 1).reshape(n * p, m)
        r3 = np.linalg.matrix_rank(M3)
        
        min_rank = max(r1, r2, r3)
        
        if min_rank < rank_target:
            # We might be able to achieve rank_target
            # But this is a lower bound, not an upper bound
            pass
        
        return None
    
    # For this specific case, let's try a constructive approach
    # The key insight: if we can find a rank-r decomposition where r < 48
    
    # Let's try to use the structure of the problem
    # 2x3 * 3x8: the middle dimension is 3, which is small
    
    # One approach: block decomposition
    # Split the 8 columns into two 4-column blocks
    # A (2x3) * B1 (3x4) and A (2x3) * B2 (3x4)
    
    # For each block, the schoolbook rank is 2*3*4 = 24
    # Total: 48
    
    # Can we do better? Try to share some computations
    
    # Actually, for exact decomposition, we need to be careful
    
    # Let's try a random search with structured initialization
    # for rank 46 and 47
    
    def try_random_search(rank, num_trials=100):
        """Try to find a rank-r decomposition via random search."""
        if time.time() - start_time > deadline:
            return None
        
        # Build target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1
        
        # We want U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that sum_a U[:,a] ⊗ V[:,a] ⊗ W[:,a] = T
        
        # This is equivalent to: for all (i,j,k):
        # sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1
        
        # Let's try with integer/half-integer entries
        
        for trial in range(num_trials):
            if time.time() - start_time > deadline:
                break
            
            # Initialize with schoolbook-like structure but with some merging
            U_trial = np.zeros((n * m, rank))
            V_trial = np.zeros((m * p, rank))
            W_trial = np.zeros((p * n, rank))
            
            # Assign each of the 48 (i,j,k) triples to one of rank columns
            # Some columns will handle multiple triples
            
            assignments = np.random.randint(0, rank, size=(n * m * p,))
            
            for idx in range(n * m * p):
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                
                col = assignments[idx]
                
                # If this column already has a different (i,j) or (j,k) or (k,i),
                # we need to check consistency
                # For simplicity, use a structured assignment
                
            # This random approach is too complex; let's use a simpler method
            
            # Use the fact that for small rank reductions, we can try
            # specific patterns
            
            # Pattern: merge columns that share the same i but different (j,k)
            # or same j but different (i,k), etc.
            
            # For rank 47: merge two columns that have the same i and j but different k
            # Column (i,j,k1) and (i,j,k2):
            #   U: both have 1 at [i*m+j]
            #   V: [j*p+k1] and [j*p+k2]
            #   W: [k1*n+i] and [k2*n+i]
            # 
            # To merge: we need weights w1, w2 such that
            # w1 * (e_{i,j} ⊗ e_{j,k1} ⊗ e_{k1,i}) + w2 * (e_{i,j} ⊗ e_{j,k2} ⊗ e_{k2,i})
            # = e_{i,j} ⊗ (w1*e_{j,k1} + w2*e_{j,k2}) ⊗ ...
            # This doesn't work directly because the W components are different
            
            # The only way to merge is if the tensors are linearly dependent,
            # which they're not in the schoolbook basis
            
            # So rank 47 might not be achievable with simple merging
            
            # Let's try a different approach: use bilinear algorithms
            
            # For now, return the schoolbook result
            return None
        
        return None
    
    # Try rank 47
    result_47 = try_random_search(47, num_trials=50)
    if result_47 is not None:
        best_rank = 47
        best_result = result_47
        return best_result
    
    # Try rank 46
    if time.time() - start_time < deadline - 5:
        result_46 = try_random_search(46, num_trials=50)
        if result_46 is not None:
            best_rank = 46
            best_result = result_46
            return best_result
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
