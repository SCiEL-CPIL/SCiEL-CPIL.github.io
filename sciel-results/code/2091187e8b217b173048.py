"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    schoolbook_rank = n * m * p  # 60
    
    # Build the multiplication tensor T[i,j,k] = delta_{i+k, j} in flattened form
    # Actually, the tensor is T[a,b,c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[(i,j),(j,k),(k,i)] = 1 if the indices are consistent, else 0
    # More precisely: T is a 3-tensor with dimensions (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for valid i,j,k
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1
        return T
    
    T = build_tensor()
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W decompose T exactly."""
        rank = U.shape[1]
        if rank > schoolbook_rank:
            return False
        # Compute sum over r of U[:,r] * V[:,r] * W[:,r] and compare to T
        # Efficient: reshape and use einsum
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check exact match (with small tolerance for floating point)
        if not np.allclose(T_dec, T, atol=tol, rtol=tol):
            return False
        
        # Also check that entries are close to integers (or half-integers)
        # The problem says "integer/half-integer entries"
        U_rounded = np.round(U_arr * 2) / 2
        V_rounded = np.round(V_arr * 2) / 2
        W_rounded = np.round(W_arr * 2) / 2
        
        if not np.allclose(U_arr, U_rounded, atol=1e-8):
            return False
        if not np.allclose(V_arr, V_rounded, atol=1e-8):
            return False
        if not np.allclose(W_arr, W_rounded, atol=1e-8):
            return False
        
        # Re-verify with rounded values
        U_r = np.array(U_rounded, dtype=np.float64)
        V_r = np.array(V_rounded, dtype=np.float64)
        W_r = np.array(W_rounded, dtype=np.float64)
        T_dec_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if not np.allclose(T_dec_r, T, atol=1e-8, rtol=1e-8):
            return False
        
        return True
    
    def make_decomposition(U, V, W):
        """Convert numpy arrays to lists of lists with rounded values."""
        U_r = np.round(np.array(U, dtype=np.float64) * 2) / 2
        V_r = np.round(np.array(V, dtype=np.float64) * 2) / 2
        W_r = np.round(np.array(W, dtype=np.float64) * 2) / 2
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }
    
    # Best result so far: schoolbook
    best_rank = schoolbook_rank
    best_result = None
    
    # Generate schoolbook decomposition
    U_sb = np.zeros((n*m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m*p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p*n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = make_decomposition(U_sb, V_sb, W_sb)
    
    # Try structured decompositions
    # Strategy 1: Try to use the fact that n=2 is small
    # For 2xm matrices, we can sometimes do better
    
    # Try random search for lower rank
    # We'll try ranks from lower values up to schoolbook
    # For each rank, try random initializations and see if we can find a valid decomposition
    
    # Actually, let's try a more systematic approach:
    # The tensor has a specific structure. Let's try to exploit it.
    
    # For now, let's try some random searches at various ranks
    # But we need to be careful about time
    
    # Try rank values: 50, 52, 54, 56, 58, 60
    # Actually, let's try to be smarter
    
    # Known lower bounds for matrix multiplication tensor rank:
    # For 2xm x mxp, the rank is at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But that's a weak bound. The actual rank is likely close to 60 for these dimensions.
    
    # Let's try a few random searches at ranks slightly below 60
    # to see if we can find anything
    
    random.seed(42)
    
    # Try some structured approaches first
    # Approach: Use the polynomial multiplication structure
    # A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    # This is like a convolution in j
    
    # For now, let's do a limited random search
    # We'll try ranks from 55 down to 40, with a few random attempts each
    
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_rand = np.random.randn(n*m, target_rank) * 0.5
            V_rand = np.random.randn(m*p, target_rank) * 0.5
            W_rand = np.random.randn(p*n, target_rank) * 0.5
            
            # Round to half-integers
            U_rand = np.round(U_rand * 2) / 2
            V_rand = np.round(V_rand * 2) / 2
            W_rand = np.round(W_rand * 2) / 2
            
            # Verify
            if verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_result = make_decomposition(U_rand, V_rand, W_rand)
                found = True
                break
        
        if found:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
