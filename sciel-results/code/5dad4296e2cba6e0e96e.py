import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p  # 96
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try rank 64 via Kronecker product decomposition
    # 3x4x8 = (3x2x4) kron (1x2x2)
    # Rank = 18 * 7 = 126? No, that's too high.
    # Let's try 3x4x8 = (3x2x2) kron (1x2x4)
    # Rank of 3x2x2 is 9 (schoolbook) or lower?
    # Actually, let's just try a few known decompositions.
    
    # For 3x4x8, the rank is known to be at most 36 (from literature)
    # Let's try to find a rank 36 decomposition.
    
    # Alternative: use the fact that 3x4x8 can be embedded in 4x4x8
    # Rank of 4x4x8 is 4*4*8/2 = 64? No, rank of 4x4x4 is 21.
    # Rank of 4x4x8 is at most 21*2 = 42?
    
    # Let's try a direct approach with ALS for rank 36
    # But first, let's try a simple Kronecker product approach
    
    # 3x4x8: try (3x2x2) with rank 9 and (1x2x4) with rank 8
    # Total rank = 9*8 = 72? That's not better than 96.
    
    # Let's try (3x4x4) with rank 18 and (1x1x2) with rank 2
    # Total rank = 18*2 = 36
    
    # First, get a rank 18 decomposition for 3x4x4
    # 3x4x4: schoolbook rank is 48
    # Known rank for 3x4x4 is 18? Let's check.
    # Actually, rank of 3x4x4 is at most 18 (from literature).
    
    # Let's construct a rank 18 decomposition for 3x4x4
    # Using the decomposition from the literature:
    # 3x4x4 can be decomposed using 18 multiplications
    
    # For now, let's try ALS with rank 36
    for rank in [36, 40, 44, 48]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating least squares
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed i: T[i, :, :] = sum_c U[i, c] (V[:, c] * W[:, c])
                # T_flat[i, :] = U[i, :] @ M_VW.T
                M_VW = np.zeros((rank, m*p*p*n))
                for c in range(rank):
                    M_VW[c, :] = (V[:, c][:, None] * W[:, c][None, :]).flatten()
                
                T_flat = T.reshape(n*m, m*p*p*n)
                try:
                    U = T_flat @ np.linalg.pinv(M_VW.T)
                except:
                    break
                
                # Update V
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed j: T[:, j, :] = sum_c V[j, c] (U[:, c] * W[:, c])
                M_UW = np.zeros((rank, n*m*p*n))
                for c in range(rank):
                    M_UW[c, :] = (U[:, c][:, None] * W[:, c][None, :]).flatten()
                
                T_flat_v = np.zeros((m*p, n*m*p*n))
                for j in range(m*p):
                    T_flat_v[j, :] = T[:, j, :].reshape(n*m, p*n).T.flatten()
                
                try:
                    V = T_flat_v @ np.linalg.pinv(M_UW.T)
                except:
                    break
                
                # Update W
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed k: T[:, :, k] = sum_c W[k, c] (U[:, c] * V[:, c])
                M_UV = np.zeros((rank, n*m*m*p))
                for c in range(rank):
                    M_UV[c, :] = (U[:, c][:, None] * V[:, c][None, :]).flatten()
                
                T_flat_w = np.zeros((p*n, n*m*m*p))
                for k in range(p*n):
                    T_flat_w[k, :] = T[:, :, k].reshape(n*m, m*p).T.flatten()
                
                try:
                    W = T_flat_w @ np.linalg.pinv(M_UV.T)
                except:
                    break
                
                # Check convergence
                T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.allclose(T_approx, T, atol=1e-8):
                    break
            
            # Round to integers/half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_approx = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            if np.allclose(T_approx, T, atol=1e-10):
                found = True
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
