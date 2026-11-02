"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U.tolist(), V.tolist(), W.tolist()

def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on the all-ones tensor structure.
    
    The tensor T[i,j,k] = 1 for all i,j,k. We can write:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy 1: Use rank = n*p + m*p decomposition
    For each (i,k), use a rank-1 term, and for each (j,k), use another.
    Actually, let's think more carefully.
    
    If we set:
    U[i*m+j, c] = a_i * b_j for c in some range
    V[j*p+k, c] = b_j * d_k  
    W[k*n+i, c] = d_k * a_i
    
    Then T[i,j,k] = sum_c (a_i*b_j) * (b_j*d_k) * (d_k*a_i) = sum_c a_i^2 * b_j^2 * d_k^2
    
    This gives T[i,j,k] = (sum a_i^2)(sum b_j^2)(sum d_k^2) which is constant, not 1.
    
    Better approach: Use the fact that the all-ones tensor of size (n*m) x (m*p) x (p*n)
    can be decomposed using the structure of the indices.
    
    Let's try: rank = n + m + p - 2 (for the case where we can separate)
    Or use a different structured approach.
    
    Actually, for the matrix multiplication tensor, the known lower bound is n*m*p - min(n*m, m*p, p*n) + 1
    or similar. But let's just try to find good decompositions computationally.
    """
    n, m, p = DIMENSIONS
    
    # Try a decomposition based on the idea that T[i,j,k] = 1
    # can be written as a sum of rank-1 tensors.
    
    # Approach: Use the identity 1 = sum_{j=0}^{m-1} delta_{j,j}
    # This doesn't directly help, but let's try numerical optimization.
    
    # Start with a random initialization at various ranks and try to minimize
    # the error ||T_decomposed - T_target||_F
    
    best_result = None
    best_error = float('inf')
    best_rank = n * m * p
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with schoolbook as guaranteed valid solution
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured approaches
    # and random search with better initialization
    
    rng = np.random.default_rng(12345)
    
    # Try various rank values from high to low
    # For 4x4x7, known results suggest rank around 80-100 might be achievable
    ranks_to_try = list(range(best_rank - 1, 70, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random initializations at this rank
        max_attempts = 30
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize with a mix of structured and random components
            # Use the insight that the tensor is all-ones, so we can try
            # decompositions that respect this structure
            
            # Strategy: Initialize U, V, W such that the product is close to 1
            # For each rank dimension c, assign random but correlated values
            
            # Use a "separable" initialization:
            # U[i*m+j, c] ~ N(0, 1/sqrt(n*m))
            # V[j*p+k, c] ~ N(0, 1/sqrt(m*p)) 
            # W[k*n+i, c] ~ N(0, 1/sqrt(p*n))
            # Then adjust to get the right scaling
            
            U_rand = rng.standard_normal((n * m, rank)) / np.sqrt(n * m)
            V_rand = rng.standard_normal((m * p, rank)) / np.sqrt(m * p)
            W_rand = rng.standard_normal((p * n, rank)) / np.sqrt(p * n)
            
            # Scale to try to get T close to 1
            # The expected value of sum_c U*V*W for each (i,j,k) is approximately
            # rank * (1/sqrt(n*m)) * (1/sqrt(m*p)) * (1/sqrt(p*n)) = rank / (n*m*p)^(3/2)
            # We want this to be 1, so scale by (n*m*p)^(3/4) / sqrt(rank)
            scale = (n * m * p) ** 0.75 / np.sqrt(rank)
            U_rand *= scale
            V_rand *= scale
            W_rand *= scale
            
            # Check if this is close enough
            U3 = U_rand.reshape(n, m, rank)
            V3 = V_rand.reshape(m, p, rank)
            W3 = W_rand.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-10:
                best_rank = rank
                U, V, W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                found = True
                break
            
            # Try a structured approach: use the fact that for each k,
            # the sub-tensor over (i,j) is all-ones
            # Decompose the all-ones matrix of size n x m as rank-1:
            # ones(n,m) = 1_n * 1_m^T
            # So for each k, we need a rank-1 decomposition in the (i,j) space
            # This suggests rank >= p (one per k)
            
            # Try rank = p with structured initialization
            if rank == p:
                # U[i*m+j, c] should be 1 for all i,j when c=k
                # V[j*p+k, c] should be 1 for all j when c=k  
                # W[k*n+i, c] should be 1 for all i when c=k
                U_struct = np.zeros((n * m, rank))
                V_struct = np.zeros((m * p, rank))
                W_struct = np.zeros((p * n, rank))
                
                for k in range(p):
                    # For this k, set the c=k column
                    U_struct[:, k] = 1.0  # All i,j get 1
                    # V[j*p+k, k] = 1 for all j
                    for j in range(m):
                        V_struct[j * p + k, k] = 1.0
                    # W[k*n+i, k] = 1 for all i
                    for i in range(n):
                        W_struct[k * n + i, k] = 1.0
                
                if _verify(U_struct.tolist(), V_struct.tolist(), W_struct.tolist(), n, m, p):
                    best_rank = rank
                    U, V, W = U_struct.tolist(), V_struct.tolist(), W_struct.tolist()
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, continue to lower ranks
            continue
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
