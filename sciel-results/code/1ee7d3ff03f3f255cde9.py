"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using structured patterns
    # For 2x2 * 2xp, we can exploit the small n,m dimensions
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_ranked_construction(R):
        """Try to construct a rank-R decomposition using structured patterns."""
        if time.time() > deadline:
            return None
        
        # Strategy: For 2x2 * 2xp, use the fact that we only need to handle
        # 4 input positions (i,j) in {0,1}x{0,1} and p output positions k
        
        # Pattern 1: Group k values and use shared structure
        # Each term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Try various structured decompositions
        # For small R, we can try to find patterns where each term handles multiple (i,j,k)
        
        # Approach: Use the fact that for fixed (i,j), the output is sum over k
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # For 2x2 A and 2xp B:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # We need terms that produce these bilinear forms
        
        # Try: Use R = 2*p terms where each pair handles one k
        if R >= 2 * p:
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            for k in range(p):
                c1 = 2 * k
                c2 = 2 * k + 1
                
                # Term c1: A[0,0]*B[0,k] -> C[0,k]
                U[0, c1] = 1  # i=0,j=0
                V[0, c1] = 1  # j=0,k=k
                W[k*n + 0, c1] = 1  # k,k, i=0
                
                # Term c2: A[0,1]*B[1,k] -> C[0,k] AND A[1,0]*B[0,k] -> C[1,k]
                # This doesn't work cleanly. Let's use 4 terms per k instead.
            
            # Better: 4 terms per k for the 4 (i,j) pairs
            if R >= 4 * p:
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            c = k * n * m + i * m + j
                            U[i * m + j, c] = 1
                            V[j * p + k, c] = 1
                            W[k * n + i, c] = 1
                
                if verify_decomposition(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try a more clever construction using shared structure
        # For 2x2 matrices, we can use Strassen-like ideas
        
        # Strategy: Use R = 3*p + 2 or similar
        # Each k needs to contribute to both rows of C
        
        # Let's try: For each k, use 2 terms that share some structure
        # Term 1: handles (i=0,j=0) and (i=1,j=1) contributions
        # Term 2: handles (i=0,j=1) and (i=1,j=0) contributions
        # But these have different B indices, so they can't be shared easily
        
        # Alternative: Use the fact that we're multiplying by a 2xp matrix
        # The output C is 2xp, so we need 2p output values
        
        # Minimum rank is at least max(n*m, m*p, p*n) = max(4, 30, 30) = 30
        # So R must be at least 30
        
        # Try R = 30: each term handles exactly one (j,k) pair in B and one (i) in output
        if R >= 30:
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # For each (j,k) in B, and for each i, we need a term
            # But we can share: term for (j,k) contributes to both i=0 and i=1
            # if U[i*m+j] is nonzero for both i
            
            # Actually, the tensor entry is T[i,j,k] = 1 for all i,j,k
            # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # For fixed (j,k), the terms with V[j*p+k,c] != 0 must satisfy
            # sum_c U[i*m+j,c] * W[k*n+i,c] = 1 for both i=0 and i=1
            # (assuming V[j*p+k,c] = 1 for those c)
            
            # If we set V[j*p+k,c] = 1 for exactly one c per (j,k), then
            # U[i*m+j,c] * W[k*n+i,c] = 1 for both i
            # This means U[0*m+j,c] = W[k*n+0,c] and U[1*m+j,c] = W[k*n+1,c]
            # and U[0*m+j,c] * W[k*n+0,c] = 1, U[1*m+j,c] * W[k*n+1,c] = 1
            
            # So we can set U[j,c] = 1, W[k*n,c] = 1 for i=0
            # and U[m+j,c] = 1, W[k*n+1,c] = 1 for i=1
            # But that's 2 terms per (j,k), giving 2*2*p = 4p = 60 terms
            
            # To do better, we need to share terms across different (j,k) or (i)
            
            # Try: Use R = 2*p + 2 terms
            # First 2 terms: handle the "diagonal" structure
            # Remaining 2p terms: one per k, handling both j=0 and j=1
            
            R_try = 2 * p + 2
            if R >= R_try:
                U = np.zeros((n * m, R_try))
                V = np.zeros((m * p, R_try))
                W = np.zeros((p * n, R_try))
                
                # Term 0: A[0,0] * (B[0,k] for all k) -> contributes to C[0,k]
                # Can't do this in one term since B[0,k] varies with k
                
                # Different approach: Use the structure of 2x2 matrix multiplication
                # C = A * B where A is 2x2, B is 2xp
                # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
                # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
                
                # We can use 4 "row" terms:
                # Term 0: A[0,0] * B[0,k] for each k -> but that's p terms
                # This is getting complex. Let's just use the schoolbook if we can't find better.
                
                pass
        
        return None
    
    # Try various ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = 30
    min_possible_rank = max(n * m, m * p, p * n)
    
    for R in range(min_possible_rank, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        result = try_ranked_construction(R)
        if result is not None:
            return result
    
    # If no better construction found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
