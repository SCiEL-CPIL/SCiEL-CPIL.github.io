"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    def verify_decomposition_fast(U, V, W):
        """
        Verify that the decomposition exactly reproduces the multiplication tensor.
        Uses numpy for speed.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        """
        rank = U.shape[1]
        
        # The tensor T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # And T[a,b,c] = 0 for all other (a,b,c)
        
        # Efficient check: For each pair (a,b), compute the vector of contributions
        # and check against expected values
        
        # Build expected tensor as a 3D array
        T_expected = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_expected[a, b, c] = 1
        
        # Compute reconstructed tensor using einsum
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-way tensor product
        # We can compute this as: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # But that's O(rank * n*m * m*p * p*n) which is too slow for large rank
        
        # Alternative: check only the required entries and a subset of zero entries
        # Actually, we need to check ALL entries for correctness
        
        # Use batched computation:
        # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Let M[b,c] = sum_r V[b,r] * W[c,r]  -- this is a (m*p) x (p*n) matrix
        # Then T[a,b,c] = sum_c' ... no, that's not right.
        
        # Better: T[a,b,c] = (U @ diag? ) no.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U * V[:, None, :] @ W[None, :, :]) ... not quite.
        
        # Let's use: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This can be computed as: for each a, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U[a,:] * V * W^T) ... no.
        
        # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let's compute V_times_W = V @ W.T  -- shape (m*p, rank) @ (rank, p*n) = (m*p, p*n)
        # No, that gives sum_r V[b,r] * W[c,r] for each (b,c), which is what we want!
        # But then T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # This is NOT the same as U @ (V @ W.T) because the sum is over r of products of three terms.
        
        # We need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-way contraction. We can compute it as:
        # For each r, add U[:,r] * V[:,r] * W[:,r] (outer product)
        # Or use einsum: np.einsum('ar,br,cr->abc', U, V, W)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        return np.array_equal(T_recon, T_expected)
    
    # Try to find lower rank decompositions
    # Strategy: structured search with numpy-accelerated verification
    
    # For this problem, we can try:
    # 1. Known low-rank decompositions for small matrix multiplication
    # 2. Random search with structured initialization
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms
    
    # Actually, let's try a different approach:
    # For 3x3 * 3x8, we can view it as 3 independent 3x8 multiplications (one for each j)
    # But they share the A matrix structure.
    
    # Let's try random search with better initialization
    for target_rank in range(40, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        max_attempts = max(10, int((deadline - time.time()) / 2))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate sparse random factors with small integer entries
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # For each term, pick a random (i,j,k) and set entries
            for r in range(target_rank):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                coeff = random.choice([-2, -1, 1, 2])
                
                U[i * m + j, r] = coeff
                V[j * p + k, r] = coeff
                W[k * n + i, r] = coeff
            
            if verify_decomposition_fast(U, V, W):
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
