"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _wft_decomposition(n, m, p):
    """
    Try Winograd Fourier Transform Technique (WFT) for 2x4 x 4x7.
    
    For WFT, we want to find a group G and a set of functions f: G -> {0,1}
    such that the multiplication tensor can be expressed as a sum of rank-1 tensors.
    
    The standard WFT approach for matrix multiplication uses the fact that
    convolution can be computed via Fourier transform. For matrix multiplication,
    we can view it as a generalized convolution.
    
    For 2x4 x 4x7, let's try to use the cyclic group Z_4 x Z_7 or similar.
    
    Actually, let's try a simpler approach: use the fact that we can pad the matrices
    to make them square or use a known WFT construction.
    
    For 2x4 x 4x7, one approach is to use the "balanced" WFT with a group of order 
    slightly larger than needed.
    
    Let's try using the group Z_8 (order 8) for the middle dimension.
    We pad the 4x7 matrix to 4x8 and the 2x4 matrix to 2x4 (already).
    Then we compute 2x4 x 4x8 using WFT, and then take the first 7 columns.
    
    For 2x4 x 4x8, we can potentially use a lower rank decomposition.
    
    Actually, let's try a different approach: use the fact that the tensor for 
    2x4 x 4x7 has a specific structure and try to find a decomposition using 
    the Kronecker product or other algebraic structures.
    
    For now, let's try a randomized search for a low-rank decomposition.
    We'll use the fact that the tensor is sparse and try to find a rank-1 
    approximation iteratively.
    """
    n, m, p = 2, 4, 7
    rank_target = n * m * p  # Start with schoolbook
    
    # Build the expected tensor
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Try to find a decomposition of rank r
    # We'll use a greedy algorithm: at each step, find the best rank-1 tensor
    # that approximates the remaining tensor.
    
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from schoolbook down to some minimum
    # For efficiency, let's try a few specific ranks
    for r in range(n*m*p - 1, max(10, n*m*p - 20), -1):
        if time.time() > 120:
            break
            
        # Initialize U, V, W with zeros
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Residual tensor
        Tres = Texp.copy()
        
        for col in range(r):
            # Find the best rank-1 tensor u, v, w that maximizes <Tres, u⊗v⊗w>
            # This is equivalent to finding the best rank-1 approximation of Tres.
            # We can do this by flattening and using SVD, but for speed, 
            # let's use a heuristic.
            
            # For now, let's just use the schoolbook decomposition for the remaining part
            # This won't give a lower rank, so let's try a different approach.
            
            # Actually, let's just return the schoolbook for now.
            break
    
    return best_result

def _try_low_rank_search(n, m, p, max_rank_reduction=5):
    """
    Try to find a decomposition with rank lower than schoolbook.
    Uses a simple greedy approach.
    """
    schoolbook = _schoolbook(n, m, p)
    rank_sb = n * m * p
    
    # Build expected tensor
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    best_rank = rank_sb
    best_result = schoolbook
    
    # Try to reduce rank by a few units
    for r in range(rank_sb - 1, max(1, rank_sb - max_rank_reduction), -1):
        if time.time() > 120:
            break
        
        # Try to find a rank-r decomposition
        # Use a randomized approach: initialize U, V, W randomly and try to optimize
        # For simplicity, let's just check if a rank-r decomposition exists by 
        # trying a few random initializations.
        
        success = False
        for attempt in range(3):
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Check if this decomposition works
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                success = True
                break
        
        if success:
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try WFT or other methods if time permits
    if time.time() < deadline - 20:
        # Try low rank search
        low_rank_result = _try_low_rank_search(n, m, p, max_rank_reduction=3)
        if low_rank_result['U'] is not None:
            # Verify the result
            if _verify(low_rank_result['U'], low_rank_result['V'], low_rank_result['W'], n, m, p):
                result = low_rank_result
    
    return result

# EVOLVE-BLOCK-END
