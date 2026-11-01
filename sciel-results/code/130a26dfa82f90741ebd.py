import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook solution (rank 56)
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == delta[j, j']
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 130.0

    # WFT approach: Use Winograd's method for matrix multiplication
    # For (2x4)*(4x7), we can use the fact that this is equivalent to
    # computing 2*7 = 14 scalar products, each involving 4 terms.
    
    # Winograd's algorithm for small matrix multiplication uses the fact that
    # we can factor the computation using tensor rank reduction.
    
    # Let's try to construct a WFT-based decomposition.
    # The key insight is that for matrix multiplication C = AB where A is m1 x m2
    # and B is m2 x m3, we can use the Winograd Fourier Transform Technique
    # if m2 is composite or has special structure.
    
    # For m2 = 4 = 2*2, we can use a 2x2 block structure.
    # Actually, let's use the fact that 4 = 2^2 and apply Winograd's method
    # for 2x2 matrix multiplication (Strassen) in a structured way.
    
    # Alternative: Use the fact that we're multiplying 2x4 by 4x7.
    # We can view this as: C = [A1 A2] [B1; B2] where A1, A2 are 2x2 and B1, B2 are 2x7.
    # C = A1*B1 + A2*B2
    
    # For each 2x2 * 2x7 multiplication, we can use a rank-7 decomposition
    # if we can decouple the 7 columns. But we can't directly.
    
    # Let's try a different approach: use the tensor rank of the multiplication
    # operator for (2x4)*(4x7).
    
    # The tensor T has entries T[i,j,k,j',k'] = delta[j,j'] * delta[k,k'] for 
    # appropriate indices. We want to express this as a sum of rank-1 tensors.
    
    # Actually, let's use a more systematic approach. The matrix multiplication
    # tensor for (n x m) * (m x p) has rank at most n*m*p (schoolbook).
    # We can try to find a lower rank decomposition using algebraic methods.
    
    # For (2x4)*(4x7), let's try to use the fact that we can group terms.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # We can split j into j1 + 2*j2 where j1, j2 in {0,1}.
    # C[i,k] = sum_{j1=0}^{1} sum_{j2=0}^{1} A[i, j1+2*j2] * B[j1+2*j2, k]
    
    # This doesn't immediately help with rank reduction.
    
    # Let's try a random search with a better initialization strategy.
    # Use the structure of the problem to guide the search.
    
    def random_search_with_structure(rank, max_iters=200):
        """Search for a rank-r decomposition using structured random initialization."""
        for iter in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n*m, rank))
            V = np.random.randint(-2, 3, size=(m*p, rank))
            W = np.random.randint(-2, 3, size=(p*n, rank))
            
            # Verify
            if verify(U, V, W):
                return U, V, W
        return None

    # Try ranks from 55 down to 40
    for rank in range(55, 39, -1):
        if time.time() > deadline:
            break
        
        # Try structured random search
        result = random_search_with_structure(rank, max_iters=100)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    # If no improvement, try a more sophisticated approach
    if best_rank == 56:
        # Use the fact that we can decompose the problem into smaller subproblems
        # and combine them.
        
        # For (2x4)*(4x7), we can use the following decomposition:
        # Split A into columns: A = [A_col0, A_col1, A_col2, A_col3]
        # Split B into rows: B = [B_row0; B_row1; B_row2; B_row3]
        # C = sum_{j=0}^{3} A_col_j * B_row_j
        
        # Each term A_col_j * B_row_j is an outer product of a 2-vector and a 7-vector,
        # which has rank 1 in the tensor sense? No, that's not right.
        
        # Actually, let's use a different strategy. Use the fact that the
        # multiplication tensor can be expressed using the Kronecker product structure.
        
        # For now, let's try to use a greedy approach to build up the decomposition.
        
        def greedy_decomposition(max_rank=55):
            """Build a decomposition greedily, adding one rank-1 term at a time."""
            # The tensor T[i,j,k,j'] = delta[j,j'] for all i,k
            # We want to approximate this with rank-r tensors.
            
            # Start with zero
            U = np.zeros((n*m, max_rank))
            V = np.zeros((m*p, max_rank))
            W = np.zeros((p*n, max_rank))
            
            # Compute the "residual" tensor
            # T[i,j,k,j'] = delta[j,j']
            # Current approximation: sum_c U[i,j,c] * V[j',k,c] * W[k,i,c]
            
            # For simplicity, let's just use the schoolbook solution as a fallback
            return None
            
        # Try the greedy approach
        result = greedy_decomposition()
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
