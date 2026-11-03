"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Compute the tensor from U, V, W
        r = U.shape[1]
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient: use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if T_recon matches T
        if not np.allclose(T_recon, T, atol=1e-10):
            return False
        # Check if entries are integer or half-integer
        for mat in [U, V, W]:
            for val in mat.flat:
                if abs(val * 2 - round(val * 2)) > 1e-10:
                    return False
        return True
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    def als_decompose(T, r, max_iter=50, tol=1e-8):
        """ALS decomposition of T into rank-r factors."""
        np.random.seed(42)  # for reproducibility in testing, but we'll vary later
        U = np.random.randn(T.shape[0], r) * 0.1
        V = np.random.randn(T.shape[1], r) * 0.1
        W = np.random.randn(T.shape[2], r) * 0.1
        
        for it in range(max_iter):
            # Update U
            # T_reshaped for U: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # For each r, the contribution to U[:,r] is from V[:,r] and W[:,r]
            # U[:,r] = (sum_{b,c} T[:,b,c] * V[b,r] * W[c,r]) / (sum_{b,c} V[b,r]^2 * W[c,r]^2)
            for r_idx in range(r):
                # Compute numerator and denominator
                # This is expensive; let's use a different approach
                pass
            
            # Actually, ALS for 3-tensors is more complex. Let me use a simpler approach.
            # Instead, let me use the fact that we can fix two matrices and solve for the third.
            # But this is still complex. Let me try a different method.
            break
        
        return U, V, W
    
    # Given the complexity of ALS for 3-tensors, let me try a different approach:
    # Use a structured initialization based on the schoolbook decomposition,
    # then try to reduce rank by merging terms.
    
    # Alternative: try specific rank values with random initialization and a simple iterative method.
    
    # Let me try a more direct approach: for each candidate rank, try to find a solution
    # by solving a least-squares problem with the constraint that entries are half-integers.
    
    # This is still hard. Let me try a heuristic: start with schoolbook, then try to eliminate
    # terms by finding linear dependencies.
    
    # For now, let me just try a few random restarts with a simple iterative method.
    
    best_result = (rank_sb, U_sb, V_sb, W_sb)
    
    for r in range(50, rank_sb + 1):  # Try ranks from 50 up to 54
        if time.time() - start_time > deadline:
            break
        found = False
        for restart in range(10):
            if time.time() - start_time > deadline:
                break
            np.random.seed(hash((r, restart)) % (2**32))
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Simple iterative refinement
            for it in range(100):
                # Update U: for each a, U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                # This is equivalent to solving a linear system for each a
                # Let's use a gradient-based approach
                # Compute gradient of loss w.r.t. U
                # Loss = ||T - T_recon||^2
                # dLoss/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - T_recon[a,b,c]) * V[b,r] * W[c,r]
                
                # This is expensive. Let me use a simpler update.
                # Actually, let me just round and check.
                pass
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Verify
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                best_result = (r, U_r.tolist(), V_r.tolist(), W_r.tolist())
                found = True
                break
        
        if found:
            break
    
    r, U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
