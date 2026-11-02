import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    U: (n*m, rank) = (4, rank)
    V: (m*p, rank) = (8, rank)
    W: (p*n, rank) = (8, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 2, 4
    rank_14 = 14
    
    # First, build the rank-14 Strassen decomposition as fallback
    def build_strassen_14():
        rank = 14
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        for block in range(2):
            for r_local in range(7):
                r_global = block * 7 + r_local
                u_terms, v_terms, w_terms = strassen_terms[r_local]
                
                for idx, val in u_terms:
                    U[idx, r_global] += val
                    
                for idx, val in v_terms:
                    j = idx // 2
                    k = idx % 2
                    k_global = k + block * 2
                    global_v_idx = j * p + k_global
                    V[global_v_idx, r_global] += val
                    
                for idx, val in w_terms:
                    k = idx // 2
                    i = idx % 2
                    k_global = k + block * 2
                    global_w_idx = k_global * n + i
                    W[global_w_idx, r_global] += val
        
        return U, V, W
    
    U14, V14, W14 = build_strassen_14()
    best_U, best_V, best_W = U14.copy(), V14.copy(), W14.copy()
    best_rank = 14
    
    # Build the multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j] * B[j,k]
    # T has shape (n*m, m*p, p*n) = (4, 8, 8)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find rank-13 decomposition by merging two Strassen terms
    # Strategy: try all pairs of the 14 rank-14 terms and see if they can be merged
    # by finding a common structure
    
    # Alternative approach: Use the fact that the 2x2x4 problem has a special structure
    # We can write it as: C_left = A * B_left, C_right = A * B_right
    # where B_left and B_right are 2x2 blocks
    
    # The key insight: if we can find linear dependencies between the rank-14 terms
    # that exploit the rectangular geometry, we might reduce to rank 13
    
    # Let's try a more systematic approach: represent the tensor as a sum of rank-1 terms
    # and use optimization to find a sparse representation
    
    # First, let's check if rank 13 is achievable by trying to eliminate one term
    # by expressing it as a linear combination of others
    
    # The multiplication tensor for 2x2 by 2x4:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # Let's try a different approach: use the known result that the rank of 2x2 by 2x4
    # might be 13, and try to construct it
    
    # Based on the structure, let's try to find a rank-13 decomposition
    # by carefully designing the factor matrices
    
    # Try various rank-13 constructions
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Random initialization with structure
        np.random.seed(42 + attempt)
        
        # Try to build a rank-13 decomposition
        # The idea: use 6 terms for the left 2x2x2 block, 6 for the right, 
        # and 1 shared term that couples them
        
        U13 = np.zeros((4, 13), dtype=float)
        V13 = np.zeros((8, 13), dtype=float)
        W13 = np.zeros((8, 13), dtype=float)
        
        # Left block (cols 0,1 of B)
        for r in range(6):
            # Strassen-like terms for left block
            if r == 0:
                U13[0, r] = 1; U13[3, r] = 1
                V13[0, r] = 1; V13[3, r] = 1
                W13[0, r] = 1; W13[3, r] = 1
            elif r == 1:
                U13[2, r] = 1; U13[3, r] = 1
                V13[0, r] = 1
                W13[1, r] = 1; W13[3, r] = -1
            elif r == 2:
                U13[0, r] = 1
                V13[1, r] = 1; V13[3, r] = -1
                W13[2, r] = 1; W13[3, r] = 1
            elif r == 3:
                U13[3, r] = 1
                V13[2, r] = 1; V13[0, r] = -1
                W13[0, r] = 1; W13[1, r] = 1
            elif r == 4:
                U13[0, r] = 1; U13[1, r] = 1
                V13[3, r] = 1
                W13[0, r] = -1; W13[2, r] = 1
            elif r == 5:
                U13[2, r] = 1; U13[0, r] = -1
                V13[0, r] = 1; V13[1, r] = 1
                W13[3, r] = 1
        
        # Right block (cols 2,3 of B)
        for r in range(6, 12):
            r_local = r - 6
            if r_local == 0:
                U13[0, r] = 1; U13[3, r] = 1
                V13[2, r] = 1; V13[5, r] = 1
                W13[4, r] = 1; W13[7, r] = 1
            elif r_local == 1:
                U13[2, r] = 1; U13[3, r] = 1
                V13[2, r] = 1
                W13[5, r] = 1; W13[7, r] = -1
            elif r_local == 2:
                U13[0, r] = 1
                V13[3, r] = 1; V13[5, r] = -1
                W13[6, r] = 1; W13[7, r] = 1
            elif r_local == 3:
                U13[3, r] = 1
                V13[4, r] = 1; V13[2, r] = -1
                W13[4, r] = 1; W13[5, r] = 1
            elif r_local == 4:
                U13[0, r] = 1; U13[1, r] = 1
                V13[5, r] = 1
                W13[4, r] = -1; W13[6, r] = 1
            elif r_local == 5:
                U13[2, r] = 1; U13[0, r] = -1
                V13[2, r] = 1; V13[3, r] = 1
                W13[7, r] = 1
        
        # Shared/coupling term
        U13[0, 12] = 1; U13[1, 12] = 1
        V13[1, 12] = 1; V13[4, 12] = 1
        W13[1, 12] = 1; W13[6, 12] = 1
        
        # Check if this is valid
        # Compute T_reconstructed = sum_r U13[:,r] * V13[:,r] * W13[:,r]
        T_rec = np.zeros((4, 8, 8), dtype=float)
        for r in range(13):
            T_rec += np.einsum('i,j,k->ijk', U13[:,r], V13[:,r], W13[:,r])
        
        if np.allclose(T_rec, T, atol=1e-10):
            best_U, best_V, best_W = U13, V13, W13
            best_rank = 13
            break
    
    # If we found rank 13, try to see if we can do better
    if best_rank == 13:
        # Try rank 12
        if time.time() < deadline - 5:
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                np.random.seed(100 + attempt)
                # Similar search for rank 12
                # This is more challenging, skip for now
                pass
    
    # Verify the best solution
    T_rec = np.zeros((4, 8, 8), dtype=float)
    for r in range(best_rank):
        T_rec += np.einsum('i,j,k->ijk', best_U[:,r], best_V[:,r], best_W[:,r])
    
    if not np.allclose(T_rec, T, atol=1e-10):
        # Fallback to rank 14
        best_U, best_V, best_W = U14, V14, W14
        best_rank = 14
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
