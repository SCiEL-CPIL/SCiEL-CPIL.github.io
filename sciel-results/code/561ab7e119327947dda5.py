"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x7 matrix.
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Schoolbook rank as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 56
    
    # Build the tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find a lower-rank decomposition using iterative refinement
    # We'll try ranks from 28 down to 1
    
    def try_find_decomposition(r, max_iters=100):
        """Try to find a decomposition of rank r using iterative refinement."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random matrices
        np.random.seed(42)  # Fixed seed for reproducibility
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to avoid overflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Iteratively refine: fix two matrices, solve for the third
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Fix U and V, solve for W
            # For each c, we want sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
            # This is linear in W[c,d] for fixed c.
            # Let's form the system: for each (a,b), sum_d (U[a,d]*V[b,d]) * W[c,d] = T[a,b,c]
            # But this must hold for all c, which is overconstrained.
            
            # Actually, the correct approach is:
            # The tensor reconstruction is: sum_d U[:,d] ⊗ V[:,d] ⊗ W[:,d] = T
            # For fixed U and V, this is linear in W.
            # Let's form the system by flattening.
            
            # For each d, the contribution is U[:,d] ⊗ V[:,d] ⊗ W[:,d]
            # Let's define M_d = U[:,d] ⊗ V[:,d] (flattened to (n*m*m*p,))
            # Then we want sum_d M_d ⊗ W[:,d] = T (flattened)
            
            # This is a linear system in the entries of W.
            # Let's set up the system:
            # For each (a,b,c), sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
            
            # Let's collect all equations:
            # Unknowns: W[c,d] for c in range(p*n), d in range(r)
            # Total unknowns: p*n*r
            
            # Equations: for each (a,b,c), one equation
            # Total equations: n*m*m*p*p*n
            
            # This is a large system. Let's use least squares.
            
            # Form the matrix A_eq and vector b_eq
            # A_eq has shape (n*m*m*p*p*n, p*n*r)
            # b_eq has shape (n*m*m*p*p*n,)
            
            # This might be too large. Let's use a different approach.
            
            # Alternative: use the fact that the tensor is sparse.
            # Only 56 entries are non-zero.
            
            # For now, let's just check if the current decomposition is good.
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            error = np.linalg.norm(recon - T)
            
            if error < 1e-10:
                # Convert to list of lists with integer/half-integer entries
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Update W using least squares
            # For each c, solve for W[c,:]
            # The system is: for each (a,b), sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
            # This is underdetermined in general, so use least squares.
            
            # Let's form the matrix for the system:
            # Let M be the matrix with M[(a,b), d] = U[a,d]*V[b,d]
            # Shape: (n*m*m*p, r)
            # Then for each c, we want M @ W[c,:] ≈ T[:,:,c].flatten()
            
            M = (U[:, :, None, None] * V[None, None, :, :]).reshape(n * m * m * p, r)
            T_flat = T.reshape(n * m * m * p, p * n)
            
            # Solve W = (M^T M)^{-1} M^T T_flat
            # Using least squares
            W_new, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            
            # Check if the solution is good
            recon = np.einsum('ad,bd,cd->abc', U, V, W_new)
            error = np.linalg.norm(recon - T)
            
            if error < 1e-10:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W_new]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            W = W_new
            
            # Similarly, we could update U and V, but let's keep it simple for now.
        
        return None
    
    # Try various ranks
    for r in [28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
