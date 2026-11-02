import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    def verify(U, V, W):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Return the standard rank-36 decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_rank_r_structured(R):
        """
        Try to find a rank-R decomposition using structured initialization.
        
        Strategy: Start from schoolbook and try to merge terms.
        For rank R < 36, we need to find linear dependencies.
        
        We use the following approach:
        1. Create a candidate decomposition with R terms
        2. Each term is a tensor product u_r ⊗ v_r ⊗ w_r
        3. We need sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
        
        For small R, we can try specific patterns.
        """
        if R >= n*m*p:
            return schoolbook()
        
        # Try a structured decomposition
        # Idea: group by k (output column) and try to share terms
        # For each k, C[:,k] = A @ B[:,k]
        # We can try to find a decomposition where terms are shared across k
        
        # Pattern 1: For rank 24, use 8 terms per k (3 k's * 8 = 24)
        # Each term handles a subset of (i,j) pairs
        if R == 24:
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Try a specific pattern: for each k, use 8 terms
            # Each term covers a 2x2 block or similar
            for k in range(p):
                base = k * 8
                # 8 terms for this k
                # Term 0: i=0,j=0
                U[0*m+0, base+0] = 1
                V[0*p+k, base+0] = 1
                W[k*n+0, base+0] = 1
                # Term 1: i=1,j=0
                U[1*m+0, base+1] = 1
                V[0*p+k, base+1] = 1
                W[k*n+1, base+1] = 1
                # Term 2: i=2,j=0
                U[2*m+0, base+2] = 1
                V[0*p+k, base+2] = 1
                W[k*n+2, base+2] = 1
                # Term 3: i=0,j=1
                U[0*m+1, base+3] = 1
                V[1*p+k, base+3] = 1
                W[k*n+0, base+3] = 1
                # Term 4: i=1,j=1
                U[1*m+1, base+4] = 1
                V[1*p+k, base+4] = 1
                W[k*n+1, base+4] = 1
                # Term 5: i=2,j=1
                U[2*m+1, base+5] = 1
                V[1*p+k, base+5] = 1
                W[k*n+2, base+5] = 1
                # Term 6: i=0,j=2
                U[0*m+2, base+6] = 1
                V[2*p+k, base+6] = 1
                W[k*n+0, base+6] = 1
                # Term 7: i=1,j=2
                U[1*m+2, base+7] = 1
                V[2*p+k, base+7] = 1
                W[k*n+1, base+7] = 1
                # Missing: i=2,j=2 for this k - need to add
                # But we only have 8 terms per k, and 3*3=9 (i,j) pairs
                # This doesn't work directly
                
            # Let's try a different approach: use known results
            # The rank of 3x3x4 is known to be at most 24
            # Try a specific construction from the literature
            
            # For now, return None to indicate this pattern doesn't work
            return None
        
        # Try rank 27: 9 terms per k (3 k's * 9 = 27)
        if R == 27:
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            for k in range(p):
                base = k * 9
                for i in range(n):
                    for j in range(m):
                        idx = base + i*m + j
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            return None
        
        # Try rank 30: 10 terms per k
        if R == 30:
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            for k in range(p):
                base = k * 10
                for i in range(n):
                    for j in range(m):
                        idx = base + i*m + j
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
                # We have 10 slots per k but only 9 (i,j) pairs
                # Leave one slot unused or use for correction
                # For now, this is just schoolbook with extra columns
                pass
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            return None
        
        return None
    
    # Try structured decompositions from low to high rank
    for R in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        result = try_rank_r_structured(R)
        if result is not None:
            return result
    
    # Fallback to schoolbook rank 36
    return schoolbook()
# EVOLVE-BLOCK-END
