"""Find rank-minimal exact tensor decomposition for 2x3 x 3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    schoolbook_rank = n * m * p  # 54
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build the tensor T[a,b,c] = 1 if there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # Reconstruct tensor: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient check: for each r, the outer product U[:,r] outer V[:,r] outer W[:,r]
        # But we can check by computing the product and comparing to T
        # Use einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol, rtol=0)
    
    # Try to find a lower rank decomposition
    # Strategy: Use structured initialization based on the tensor's symmetry
    # The tensor has the property that T[a,b,c]=1 only when the indices are "compatible"
    
    # Key insight: For a 2x3 x 3x9 matrix multiplication, the tensor rank might be less than 54
    # due to the rectangular structure. Let's try to find a decomposition of rank < 54.
    
    # Approach: Try rank R from 53 down to some minimum, using random initialization
    # and alternating least squares (ALS) to converge to a valid decomposition.
    
    def try_rank_als(R, n_restarts=5, max_iter=50, tol=1e-12):
        """Try to find a rank-R decomposition using ALS."""
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(restart * 1000 + R)
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Update U: for each row a, solve for U[a,:]
                # T[a,:,:] = sum_r U[a,r] * V[:,r] outer W[:,r]
                # This is a linear system in U[a,:]
                VW = np.einsum('br,cr->bc', V, W)  # (mp, pn)
                # T[a] = U[a,:] @ VW^T ... wait, let me think more carefully
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
                # So T[a] = sum_r U[a,r] * (V[:,r] outer W[:,r]).flatten()
                # This is a linear system: T[a].flatten() = U[a,:] @ M where M[r, (b,c)] = V[b,r]*W[c,r]
                
                # Build M: shape (R, mp*pn)
                M = np.einsum('br,cr->rbc', V, W).reshape(R, mp*pn)
                
                for a in range(nm):
                    rhs = T[a].flatten()
                    # Solve U[a,:] M = rhs
                    try:
                        U[a], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        return None
                
                # Update V
                UW = np.einsum('ar,cr->ac', U, W)  # (nm, pn)
                M = np.einsum('ar,cr->rbc', U[:, :, None], W[:, :, None]).reshape(R, nm*pn)
                # Wait, that's not right. Let me redo.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:,b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
                # M[r, (a,c)] = U[a,r]*W[c,r]
                M = np.einsum('ar,cr->rac', U, W).reshape(R, nm*pn)
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    try:
                        V[b], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        return None
                
                # Update W
                UV = np.einsum('ar,br->ab', U, V)  # (nm, mp)
                M = np.einsum('ar,br->rab', U, V).reshape(R, nm*mp)
                for c in range(pn):
                    rhs = T[:, :, c].flatten()
                    try:
                        W[c], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        return None
            
            # Check if we found a valid decomposition
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks from 53 down to 45
    for R in range(schoolbook_rank - 1, 44, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_als(R, n_restarts=3, max_iter=30)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = R
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # Convert to lists for the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
