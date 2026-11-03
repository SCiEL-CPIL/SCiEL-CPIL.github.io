import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a,b,c] where T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as guaranteed fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb
    
    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        r = U_arr.shape[1]
        for t in range(r):
            result += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        return np.allclose(result, T, atol=tol)
    
    # Rank-revealing approach: flatten tensor and use SVD to find minimal rank
    # The tensor T can be viewed as a matrix of shape (nm*mp, pn) or similar
    # For matrix multiplication tensor, we can use the "mode" flattening
    
    # Flatten T as a 3-tensor and try different flattenings
    # Mode-0 flattening: (nm, mp*pn)
    T_flat0 = T.reshape(nm, mp * pn)
    # Mode-1 flattening: (mp, nm*pn)
    T_flat1 = T.reshape(mp, nm * pn).T
    # Mode-2 flattening: (pn, nm*mp)
    T_flat2 = T.reshape(pn, nm * mp).T
    
    # Use SVD to find numerical rank for each flattening
    # This gives us a lower bound on tensor rank
    
    U0, S0, V0t = np.linalg.svd(T_flat0, full_matrices=False)
    U1, S1, V1t = np.linalg.svd(T_flat1, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(T_flat2, full_matrices=False)
    
    # Numerical rank (with tolerance)
    tol_svd = 1e-10
    num_rank0 = np.sum(S0 > tol_svd)
    num_rank1 = np.sum(S1 > tol_svd)
    num_rank2 = np.sum(S2 > tol_svd)
    
    # The tensor rank is at least the maximum of these numerical ranks
    lower_bound = max(num_rank0, num_rank1, num_rank2)
    
    # For 3x3x7, the schoolbook rank is 63
    # Let's try to find a decomposition with rank close to the lower bound
    # But we need to be careful: tensor rank >= max(modes), but can be larger
    
    # Try a structured search: start from schoolbook and try to merge terms
    # using linear algebra to find dependencies
    
    # Approach: Try rank values from lower_bound+1 up to rank_sb-1
    # For each rank, try to find a valid decomposition
    
    # First, let's try a few specific ranks that might work
    # For 3x3x7, we expect the rank to be somewhere between lower_bound and 63
    
    # Try a systematic approach: use the fact that we can express the tensor
    # as a sum of rank-1 tensors, and search for a minimal set
    
    # Given time constraints, try a few candidate ranks
    candidate_ranks = []
    # Add some candidates around the lower bound and in between
    for r in range(max(lower_bound + 1, 20), min(rank_sb, 63)):
        candidate_ranks.append(r)
    
    # Sort candidates to try lower ranks first
    candidate_ranks = sorted(set(candidate_ranks))
    
    # Try each candidate rank with a limited number of attempts
    np.random.seed(42)
    
    for target_rank in candidate_ranks[:10]:  # Limit to 10 attempts
        if time.time() > deadline - 20:
            break
        
        # Try to construct a decomposition of target_rank
        # Start from schoolbook and try to reduce
        
        # Method: Take first target_rank schoolbook terms and see if they span
        # Actually, this won't work directly. Let's use a different approach.
        
        # Use the SVD structure to guide the decomposition
        # For mode-0 flattening, we have T = U0 @ diag(S0) @ V0t
        # This suggests a decomposition where we use the left and right singular vectors
        
        # But tensor decomposition is more complex than matrix SVD.
        
        # Let's try a random search with structured initialization
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            U_arr = np.zeros((nm, target_rank), dtype=np.float64)
            V_arr = np.zeros((mp, target_rank), dtype=np.float64)
            W_arr = np.zeros((pn, target_rank), dtype=np.float64)
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            U_arr = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V_arr = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W_arr = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
            
            # Verify
            if verify(U_arr, V_arr, W_arr):
                best_U = U_arr.copy()
                best_V = V_arr.copy()
                best_W = W_arr.copy()
                best_rank = target_rank
                break
        
        if best_rank < rank_sb:
            break
    
    # If we found a better decomposition, return it; otherwise return schoolbook
    if best_rank < rank_sb:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
