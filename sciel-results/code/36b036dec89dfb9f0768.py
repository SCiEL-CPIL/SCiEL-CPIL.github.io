"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Build the full tensor for verification
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T should be 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k, and 0 elsewhere
    
    # Precompute target entries
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i * m + j, j * p + k, k * n + i))
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W with given rank correctly decompose the tensor."""
        # Compute all entries: T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        # This is expensive: 15*35*21*rank operations
        # Use matrix multiplication trick:
        # For each s, the outer product U[:,s] @ V[:,s] @ W[:,s] contributes
        # But we can compute T as: for each a,b, T[a,b,:] = sum_s U[a,s]*V[b,s]*W[:,s]
        # = (U @ V.T) * W element-wise... no, that's not right.
        
        # Actually: T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # Let M_s = U[:,s] @ V[:,s] (outer product, shape n*m x m*p)
        # Then T[a,b,c] = sum_s M_s[a,b] * W[c,s]
        # This is still expensive.
        
        # Alternative: for each pair (a,b), compute the vector over c:
        # T[a,b,:] = sum_s (U[a,s]*V[b,s]) * W[:,s]
        # = (U[a,:] * V[b,:]) @ W.T  (element-wise product then dot with each row of W.T)
        # Wait: sum_s U[a,s]*V[b,s]*W[c,s] = dot(U[a,:]*V[b,:], W[c,:])
        # So T[a,b,c] = dot(U[a,:]*V[b,:], W[c,:])
        # We can compute this as: for each a,b, the vector T[a,b,:] = (U[a,:]*V[b,:]) @ W.T
        # But W.T is rank x p*n, so (U[a,:]*V[b,:]) @ W.T gives a vector of length p*n
        # That's 15*35 matrix-vector multiplies of length rank.
        
        # Even better: form the matrix P = U * V broadcast... 
        # Let's just do it directly but efficiently with numpy.
        
        # Compute T as a 3D array
        # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # We can do: for each s, add np.outer(np.outer(U[:,s], V[:,s]), W[:,s])
        # But that's rank times, each O(15*35*21)
        
        # Better: use einsum
        # T = np.einsum('as,bs,cs->abc', U, V, W)
        # This might be slow for large rank. Let's try it.
        
        T = np.einsum('as,bs,cs->abc', U, V, W)
        
        # Check target entries
        for (a, b, c) in target_entries:
            if abs(T[a, b, c] - 1) > 1e-9:
                return False
        
        # Check non-target entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_entries:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
                    # else already checked above
        
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        T = np.einsum('as,bs,cs->abc', U, V, W)
        
        # Create target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        
        if not np.allclose(T, T_target, atol=1e-9):
            return False
        return True
    
    # Try ranks from best_rank-1 down
    # Use random search with verification
    for rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random integer/half-integer entries
            scale = 2
            U = (np.random.randint(-scale, scale + 1, (n * m, rank)) * 0.5)
            V = (np.random.randint(-scale, scale + 1, (m * p, rank)) * 0.5)
            W = (np.random.randint(-scale, scale + 1, (p * n, rank)) * 0.5)
            
            if verify_fast(U, V, W, rank):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
