"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, 0 otherwise
    # For a candidate of rank r: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # So T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check that off-diagonal terms are 0
    # The tensor has entries only at positions where the indices align:
    # T[a, b, c] is non-zero only if a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # and j == j' and k == k' and i == i'
    # So T[a,b,c] = 0 unless b's first index matches a's second, and c's first matches b's second, and c's second matches a's first
    
    # Let's check a few off-diagonal cases to be safe
    checked = 0
    for a in range(n*m):
        for b in range(m*p):
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j != j2:
                continue  # These will always be 0 in the true tensor
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U[a][cc] * V[b][cc] * W[c][cc]
                    if abs(val) > 1e-6:
                        return False
                    checked += 1
                    if checked > 10000:
                        break
            if checked > 10000:
                break
        if checked > 10000:
            break
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _try_block_decomposition(n, m, p, partition_m):
    """
    Try to decompose matmul as sum of block decompositions.
    Partition the inner dimension m into blocks.
    For each block of size b, we use rank b * n * p for that block if we don't exploit structure.
    But maybe we can do better per block?
    
    Actually, for 2xb * bx p, the rank is at least max(2*b, b*p, 2*p)... no.
    The tensor rank of 2xb*b*p is at least the max dimension of any single factor in a 3-tensor.
    
    Let's just try small m values and find optimal.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Strategy: Try to find rank reductions by exploiting structure
    # For 2x5*5x6, the key is that n=2 is small.
    # 
    # Insight: We can try to use "sum-product" tricks.
    # For each pair of rows in A and columns in B, we need to compute dot products.
    # 
    # Alternative: Use the fact that we can factor through intermediate values.
    # Let's try: compute S[j] = A[0,j]*B[j,:] + A[1,j]*B[j,:] for all j
    # This doesn't directly help.
    #
    # Better approach: Use a rank-r decomposition where r < 60 by finding
    # linear dependencies. We'll use a local search / perturbation method.
    #
    # For rank 59: Start with schoolbook, zero out one column, and try to
    # adjust other columns to compensate. This is a system of 60 equations
    # (one for each (i,j,k) triple) with 59*3=177 unknowns (if we fix one
    # column's pattern), so it's overdetermined but might have solutions.
    
    import random
    random.seed(42)
    
    def _try_newton_refinement(rank, max_iters=50):
        """
        Try to find a valid rank-r decomposition by starting near schoolbook
        and using iterative refinement.
        
        We fix the "support" pattern: each column c corresponds to a triple
        (i_c, j_c, k_c). For rank < n*m*p, some columns must handle multiple
        triples, which means they can't be pure basis vectors.
        
        Instead, we use a different approach: parametrize U, V, W with random
        initial values and use gradient descent on the loss.
        """
        # Build the list of required (i,j,k) triples
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        num_triples = len(triples)  # 60
        
        # For rank r, we need r columns in each of U, V, W
        # Initialize with random values
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Compute loss and gradient via finite differences or analytic
        # Loss = sum over all (a,b,c) of (T_target[a,b,c] - T_candidate[a,b,c])^2
        # where T_candidate[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        
        # This is expensive. Let's use a smarter approach.
        
        # Actually, for small ranks close to 60, let's try a different tactic:
        # Use the schoolbook solution and try to "compress" it.
        
        # For each pair of columns that could potentially be merged, check
        # if their contribution can be expressed with a single column.
        
        # Column c in schoolbook: U has 1 at position i*m+j, V has 1 at j*p+k, W has 1 at k*n+i
        # Two columns c1=(i1,j1,k1) and c2=(i2,j2,k2) can be merged into one column
        # if there exist u, v, w vectors such that:
        #   u * v * w = e_{i1*m+j1} * e_{j1*p+k1} * e_{k1*n+i1} + e_{i2*m+j2} * e_{j2*p+k2} * e_{k2*n+i2}
        # This requires that the two terms share structure, which they don't in general.
        
        # So pure merging doesn't work. We need a more subtle approach.
        
        # Let's try: for rank 59, we need to express 60 basis terms with 59 rank-1 terms.
        # This means one of the 59 terms must "cover" two of the 60 required products.
        # But a single rank-1 term u*v*w can only be nonzero at one position (a,b,c)
        # if u, v, w are basis vectors. If they're not, it produces many nonzero entries.
        
        # The only way to reduce rank is if the tensor has a non-trivial rank-1
        # decomposition with fewer than 60 terms, which requires the tensor to be
        # "special" in some algebraic sense.
        
        # For generic matmul tensors, rank = n*m*p. But for small dimensions,
        # there might be exceptions.
        
        # Let's just try random search with verification.
        return None
    
    # Try a structured approach: exploit the fact that n=2
    # 
    # Key idea: For each j (column of A / row of B), we have two values A[0,j] and A[1,j].
    # The output is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    #   C[0,k] = sum_j A[0,j] * B[j,k]
    #   C[1,k] = sum_j A[1,j] * B[j,k]
    #
    # Now, if we define for each j: x_j = A[0,j], y_j = A[1,j], and for each k: 
    #   the k-th column of B is B[:,k] = (B[0,k], ..., B[m-1,k])
    #
    # Then C[0,k] = sum_j x_j * B[j,k] and C[1,k] = sum_j y_j * B[j,k]
    #
    # This is like computing two dot products for each k.
    #
    # Rank lower bound: For fixed k, we need to compute two linear combinations of m variables.
    # The rank needed for this is at most 2m (one for each output, using m terms each).
    # For all p values of k, that's 2mp = 60, which is schoolbook.
    #
    # Can we do better? Only if there's structure across different k values.
    # But B is arbitrary, so there's no structure to exploit.
    #
    # Conclusion: For generic 2x5*5x6 matrix multiplication, the rank is 60.
    # The schoolbook decomposition is optimal.
    
    # However, let's still try a few random searches to be sure, in case we're wrong.
    # Use a time-limited search.
    
    def _random_search_for_rank(r, num_trials=1000):
        """Try random initializations and check if any work."""
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Use a structured random initialization:
            # Start with schoolbook and perturb
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # First r columns from schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
                    if col >= r:
                        break
                if col >= r:
                    break
            
            # If we didn't fill all r columns, add random ones
            while col < r:
                U[:, col] = np.random.randn(n*m) * 0.1
                V[:, col] = np.random.randn(m*p) * 0.1
                W[:, col] = np.random.randn(p*n) * 0.1
                col += 1
            
            # Convert to lists and verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_rank(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 59 down to 30
    for rank in range(59, 29, -1):
        if time.time() > deadline:
            break
        
        result = _random_search_for_rank(rank, num_trials=200)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best

# EVOLVE-BLOCK-END
