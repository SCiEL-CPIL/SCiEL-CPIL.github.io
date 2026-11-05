"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = delta (where A[i,j] * B[j,k] -> C[i,k])
    can be viewed as a 3-index tensor of shape (n, m, p) = (3, 4, 5).
    
    We seek rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    The schoolbook rank is n*m*p = 3*4*5 = 60.
    
    Known results: For 2x2, rank is 7 (Strassen). For 3x3, best known is 23 (Landsberg).
    For 3x4 * 4x5, we don't have a known optimal, but we can try to find good decompositions.
    
    Strategy: Use alternating least squares / iterative refinement to find low-rank exact decompositions.
    Start from schoolbook and try to reduce rank by finding linear dependencies.
    """
    n, m, p = DIMENSIONS
    
    # The multiplication tensor T[i,j,k] = 1 if the indices are consistent, 0 otherwise
    # Actually, T is the identity-like tensor: T[i,j,k] represents the contribution of A[i,j]*B[j,k] to C[i,k]
    # So T is a 3D array where T[i,j,k] = 1 for all valid (i,j,k), representing the "identity" of the multiplication map.
    
    # More precisely: The tensor T has entries T[i,j,k] which is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # This is just the all-ones tensor of shape (n, m, p) in the appropriate indexing.
    
    # Let's think about this more carefully.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is such that T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # The decomposition T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (with appropriate index conventions)
    
    # Wait, let me re-read the task. The output factor W follows W[k*n+i, r].
    # So W is indexed by (k, i) -> row k*n+i, and U by (i,j) -> row i*m+j, V by (j,k) -> row j*p+k.
    
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes exactly once to C[i,k]).
    
    # Actually no. Let me reconsider. The "multiplication tensor" for matmul is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because the bilinear map (A,B) -> C has C[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]B[j,k]
    # is a separate bilinear form. The tensor of bilinear forms has entries that are 1.
    
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a tensor completion / decomposition problem.
    
    # Let's try a direct approach: start with schoolbook (rank 60) and try to find a lower rank.
    # One approach: use the fact that for small dimensions, we can try rank r and solve the system.
    
    # For rank r, we have U (12 x r), V (20 x r), W (15 x r). Total unknowns: 12r + 20r + 15r = 47r.
    # Constraints: 3*4*5 = 60 equations (one for each (i,j,k)).
    
    # For r = 60, we have 47*60 = 2820 unknowns and 60 equations - very underdetermined.
    # We can try smaller r.
    
    # Let's try to find the minimum r by searching.
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # First, let's just return schoolbook to be safe, then try to improve.
    def schoolbook():
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
        return U, V, W, rank
    
    # Try to find lower rank decompositions
    # We'll use a randomized approach with ALS-like updates
    
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()
    
    # For a given rank r, try to find U, V, W such that the tensor equation holds.
    # This is a non-convex problem. We can use iterative methods.
    
    def try_rank(r, max_iters=200, n_restarts=5):
        """Try to find a rank-r decomposition using iterative refinement."""
        if r >= best_rank:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to avoid scale issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # For each factor, fix the others and solve for the best update.
                # This is like ALS for tensor decomposition.
                
                # Update U: for each (i,j), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # Fix V, W. For fixed (i,j), the equation is:
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                # This is a linear system in U[i*m+j, :].
                
                # Build the system: for each (i,j), we have p equations (one for each k)
                # A[i,j] * x = b[i,j] where A[i,j] is p x r, b[i,j] is p-vector of ones.
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A_mat = np.outer(V[j*p:(j+1)*p, :], W[0:(p)*n:n, :])  # p x r
                        # Wait, W[k*n+i, c] for k=0..p-1
                        W_slice = np.array([W[k*n+i, :] for k in range(p)])  # p x r
                        A_mat = V[j*p:(j+1)*p, :] * W_slice  # p x r (element-wise)
                        b_vec = np.ones(p)
                        # Solve A_mat @ x = b_vec
                        # Least squares
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            x = np.zeros(r)
                        U_new[i*m+j, :] = x
                
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        W_slice = np.array([W[k*n+i, :] for i in range(n)])  # n x r
                        A_mat = U[0:(m)*n:n, :] * W_slice  # n x r
                        # Wait, U[i*m+j, c] for i=0..n-1
                        U_slice = np.array([U[i*m+j, :] for i in range(n)])  # n x r
                        A_mat = U_slice * W_slice  # n x r
                        b_vec = np.ones(n)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            x = np.zeros(r)
                        V_new[j*p+k, :] = x
                
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        U_slice = np.array([U[i*m+j, :] for j in range(m)])  # m x r
                        V_slice = np.array([V[j*p+k, :] for j in range(m)])  # m x r
                        A_mat = U_slice * V_slice  # m x r
                        b_vec = np.ones(m)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            x = np.zeros(r)
                        W_new[k*n+i, :] = x
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence / error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < 1e-10:
                    # Found a valid decomposition
                    return U, V, W, r
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # But for matmul, the rank is at least max(n*p, m*p, n*m) and at most n*m*p.
    # Known: rank(2x2) = 7 > 4 = max(2*2, 2*2, 2*2)
    
    # Let's try ranks from 20 upward
    for r in range(20, min(60, best_rank)):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, rank = result
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-8:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
