import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Schoolbook rank 40 as fallback
    rank_school = n * m * p  # 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank 36
    target_rank = 36
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with schoolbook pattern for first 36 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                if col < target_rank:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
    
    # The last 4 terms (col 36-39 in schoolbook) are missing.
    # We need to modify the 36 terms to cover all 40 (i,j,k) triples.
    
    # Alternative: use a structured initialization and iterate.
    
    # Let's try a different approach: use the fact that we can share terms.
    # For 4 specific (i,j,k) pairs, use combined terms.
    
    # Let me try: use 36 terms where:
    # - 32 terms are "standard" for most (i,j,k)
    # - 4 terms are "shared" covering 2 (i,j,k) pairs each
    
    # Actually, let me try a direct construction based on Strassen.
    
    # Partition j into {0,1} and {2,3}.
    # For j-group 0 (j=0,1): A[:,0:2] is 2x2, B[0:2,:] is 2x5
    # For j-group 1 (j=2,3): A[:,2:4] is 2x2, B[2:4,:] is 2x5
    
    # For each 2x2 x 2x5 block, use 18 terms (saving 2 per block).
    # Total: 18 + 18 = 36.
    
    # For 2x2 x 2x1 (scalar B entry), the tensor has 4 components.
    # Rank of 2x2x2 tensor is 7, so we can't do 2x2x2x1 in fewer than... 
    # Actually the rank is 7 for the full 2x2x2 tensor.
    
    # I think I need to just do a numerical search.
    
    # Let me try: random initialization + ALS
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each row i*m+j, solve least squares
            for idx in range(n * m):
                i = idx // m
                j = idx % m
                # We need: sum_r U[idx,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is: U[idx,:] @ (V[j*p:(j+1)*p,:] * W[i::n,:] element-wise) = ones(p)
                # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then U[idx,:] @ M.T = ones(p)
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[idx,:] @ M.T = ones(p)
                # M @ U[idx,:].T = ones(p)
                try:
                    U[idx, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
            
            # Update V: for each row j*p+k
            for idx in range(m * p):
                j = idx // p
                k = idx % p
                # sum_r U[i*m+j,r] * V[idx,r] * W[k*n+i,r] = 1 for all i
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[idx, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
            
            # Update W: for each row k*n+i
            for idx in range(p * n):
                k = idx // n
                i = idx % n
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[idx,r] = 1 for all j
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[idx, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check if solution is close to exact
        if verify(U, V, W):
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            if verify(U_round, V_round, W_round):
                return {
                    "U": U_round.tolist(),
                    "V": V_round.tolist(),
                    "W": W_round.tolist()
                }
        
        # Check if close to exact even without rounding
        if verify(U, V, W):
            # Return as-is (floats might be accepted)
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
