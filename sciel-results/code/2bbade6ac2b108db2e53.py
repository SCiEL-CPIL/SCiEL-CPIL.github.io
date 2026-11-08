"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need rank-r decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Strategy: Try ranks from a lower bound up, using structured patterns
    # Lower bound: at least n*p = 16 (since we need to produce n*p output elements)
    # Upper bound: schoolbook = n*m*p = 48
    
    # For 2x3 x 3x8, let's try to find a good decomposition
    # Known: rank of 2x3 x 3x3 is 14, so for 3x8 we might do better than 14*8/3
    
    # Let's try a systematic construction based on splitting the k dimension
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use rank-r where we share computation across k values
    
    # Try rank values and see if we can construct a valid decomposition
    # For now, use a known good upper bound and try to minimize
    
    # A simple approach: for each (i,k) pair, use m separate terms (schoolbook per output)
    # But we can share: if we fix j, then for all i and k, we can use one term per j
    # Actually, let's think differently.
    
    # The tensor is T[(i,j), (j,k)] = 1
    # A rank-1 term is u[(i,j)] * v[(j,k)] * w[(k,i)]
    # So we need: sum_c u_c[(i,j)] * v_c[(j,k)] * w_c[(k,i)] = 1 for all i,j,k
    
    # This is equivalent to: for each (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Let's try to find a decomposition with rank r
    # Start with r = 24 and see if we can construct one
    
    def try_construct(rank):
        """Try to construct a decomposition of given rank using structured patterns."""
        # Initialize with zeros
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Pattern: For each j in [0,m), we can create terms that handle all (i,k) for that j
        # Term for (j, i, k): U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
        # This is schoolbook with rank n*m*p
        
        # Better: Share across k for fixed (i,j)
        # For fixed i,j, we need sum over k of terms
        # If we use one term per (i,j) with V[j*p+k] varying... no, that doesn't work directly
        
        # Alternative: Use a factorization of the k dimension
        # Since p=8, we can use 3 bits to index k
        # This might allow sharing
        
        # Let's try a direct construction:
        # For each j, create n*p terms? That's m*n*p = 48 again.
        
        # Try: For each (i,j), use one term that covers all k via W
        # U[i*m+j, c] = 1, V[j*p+k, c] needs to be 1 for all k - impossible with one term
        
        # Let's just return schoolbook for now as fallback
        # We'll improve this
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if column < rank:
                        U[i*m+j, column] = 1.0
                        V[j*p+k, column] = 1.0
                        W[k*n+i, column] = 1.0
                        column += 1
        return U, V, W
    
    # Try to find the minimal rank by testing validity
    def is_valid(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p
    best_U, best_V, best_W = try_construct(best_rank)
    
    # Try to find better ranks using structured patterns
    # For 2x3 x 3x8, let's try ranks from 24 up to 47
    
    # A better construction: exploit that n=2
    # For each j and k, we need to compute A[0,j]*B[j,k] and A[1,j]*B[j,k]
    # We can use 2 terms per (j,k) pair, but share across j
    
    # Let's try a construction where we group by k
    for r in range(24, best_rank + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a structured construction for rank r
        # Split the work: m=3, p=8, n=2
        # Total work: 3*8*2 = 48 scalar mults
        # With rank r, we have r "slots"
        
        # Construction idea: Use r terms, each handling a subset of (i,j,k)
        # For simplicity, try a round-robin assignment
        
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        W = np.zeros((p*n, r), dtype=float)
        
        # Assign each (i,j,k) to a rank slot
        # If r >= n*m*p, this is trivial
        # For r < n*m*p, we need to share slots, which requires the values to combine correctly
        
        # For now, only use schoolbook if r >= n*m*p
        if r >= n * m * p:
            U, V, W = try_construct(r)
            if is_valid(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        else:
            # For r < n*m*p, we need a smarter construction
            # Try: for each (j,k), use n terms? That's m*p*n = 48
            # Or for each (i,j), use p terms? That's n*m*p = 48
            
            # Let's try a different approach: use the fact that we can have non-1 values
            # This is complex, so for now, fall back to schoolbook
            pass
    
    # Convert to lists for return
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    # Ensure entries are integer or half-integer
    def clean_entries(arr):
        result = []
        for row in arr:
            clean_row = []
            for x in row:
                # Round to nearest half-integer
                y = round(x * 2) / 2
                clean_row.append(y)
            result.append(clean_row)
        return result
    
    U_out = clean_entries(to_list(best_U))
    V_out = clean_entries(to_list(best_V))
    W_out = clean_entries(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
