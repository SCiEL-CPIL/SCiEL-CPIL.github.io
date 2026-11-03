"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition search."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    """Search for low-rank tensor decomposition of 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Build the target tensor T[i,k,j] where C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,k,j] = 1 for valid multiplication terms
    T = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    
    # Flatten to match the decomposition convention
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T_flat[i*m+j, j*p+k, k*n+i] -> need to reshape properly
    
    # Actually, let's define the tensor as:
    # For each term in the sum: A[i,j] * B[j,k] contributes to C[i,k]
    # The tensor entry is at position (i,j) in U, (j,k) in V, (k,i) in W
    
    # Let's build the target tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T3 = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T3[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    # Initialize schoolbook
    for r in range(best_rank):
        i = r // (m*p)
        j = (r // p) % m
        k = r % p
        best_U[i*m+j, r] = 1.0
        best_V[j*p+k, r] = 1.0
        best_W[k*n+i, r] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U,V,W exactly decompose T3."""
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T3_recon[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This is a tensor product: sum over r of outer product
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T3, atol=tol)
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        rank = U.shape[1]
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(recon - T3, ord='fro')
    
    # Try to find lower rank decompositions using optimization
    # We'll use a gradient-free approach with random restarts
    
    def random_init(rank, scale=1.0):
        U = np.random.randn(n*m, rank) * scale
        V = np.random.randn(m*p, rank) * scale
        W = np.random.randn(p*n, rank) * scale
        return U, V, W
    
    def optimize_decomposition(U0, V0, W0, max_iter=500, tol=1e-12):
        """Optimize the decomposition using a simple iterative method."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        rank = U.shape[1]
        
        # Use alternating least squares / fixed point iteration
        # For each r, we want U[:,r], V[:,r], W[:,r] to contribute to T3
        # This is hard to optimize directly, so let's use a different approach
        
        # Instead, let's use the fact that for exact decomposition,
        # we can try to find rank-r solutions by solving a system
        
        return U, V, W
    
    # Better approach: Use the structure of the problem
    # The tensor T3 has a specific structure - it's the matrix multiplication tensor
    # For 3x3 * 3x7, we can exploit that the 3x3 matrix acts on 7 independent columns
    
    # Let's try a smarter initialization based on the structure
    # The multiplication C = A*B where A is 3x3 and B is 3x7
    # This is equivalent to 7 independent 3x3 multiplications (one for each column of B)
    # But the 3x3 part is shared
    
    # Known results: For m x n * n x p, the rank is at most mn + np + mp - m - p (Strassen-like)
    # But for small cases, we can search
    
    # Let's try ranks from low to high
    for target_rank in range(20, best_rank + 1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 30:
                break
                
            U, V, W = random_init(target_rank, scale=0.5)
            
            # Try to optimize using a simple iterative method
            # We'll use the fact that the tensor is sparse and structured
            
            # For exact decomposition, we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T3[i*m+j, j*p+k, k*n+i]
            
            # This is a system of equations. Let's set up the linear system
            # for fixed U and V, solve for W, and vice versa
            
            # Actually, let's use a different approach: 
            # Since the target tensor has only 1s in specific positions,
            # we can try to find a decomposition where each rank-1 term
            # contributes to multiple entries
            
            # For now, let's just check if we can find a solution by
            # solving a least squares problem iteratively
            
            U_curr, V_curr, W_curr = U, V, W
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                    
                # Fix U and V, solve for W
                # For each (a,b) pair where a=i*m+j, b=j*p+k:
                # sum_r U[a,r] * V[b,r] * W[c,r] = T3[a,b,c] for all c
                # This is: (U * V^T) @ W = T3 reshaped
                
                # Let's compute the effective matrix
                # M[a,c] = sum_r U[a,r] * W[c,r] for each b... this is getting complex
                
                # Simpler: use gradient descent on the error
                error = compute_error(U_curr, V_curr, W_curr)
                if error < 1e-10:
                    found = True
                    break
                
                # Compute gradients numerically or analytically
                # dE/dU[a,r] = sum_{b,c} (recon[a,b,c] - T3[a,b,c]) * V[b,r] * W[c,r]
                recon = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    recon += np.einsum('a,b,c->abc', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                diff = recon - T3
                
                # Gradient for U
                grad_U = np.zeros_like(U_curr)
                for r in range(target_rank):
                    grad_U[:, r] = np.einsum('abc,br,cr->a', diff, V_curr[:, r], W_curr[:, r])
                
                # Gradient for V
                grad_V = np.zeros_like(V_curr)
                for r in range(target_rank):
                    grad_V[:, r] = np.einsum('abc,ar,cr->b', diff, U_curr[:, r], W_curr[:, r])
                
                # Gradient for W
                grad_W = np.zeros_like(W_curr)
                for r in range(target_rank):
                    grad_W[:, r] = np.einsum('abc,ar,br->c', diff, U_curr[:, r], V_curr[:, r])
                
                # Simple gradient descent with adaptive step size
                step = 0.01
                U_curr = U_curr - step * grad_U
                V_curr = V_curr - step * grad_V
                W_curr = W_curr - step * grad_W
                
                # Normalize to prevent divergence
                norm_U = np.linalg.norm(U_curr)
                norm_V = np.linalg.norm(V_curr)
                norm_W = np.linalg.norm(W_curr)
                if norm_U > 0 and norm_V > 0 and norm_W > 0:
                    U_curr = U_curr / norm_U * np.sqrt(norm_U * norm_V * norm_W / (norm_U * norm_V * norm_W))
                    # Actually, let's just rescale to maintain balance
                    total_norm = np.sqrt(norm_U * norm_V * norm_W)
                    if total_norm > 0:
                        scale_U = (total_norm / norm_U) ** (1/3)
                        scale_V = (total_norm / norm_V) ** (1/3)
                        scale_W = (total_norm / norm_W) ** (1/3)
                        U_curr *= scale_U
                        V_curr *= scale_V
                        W_curr *= scale_W
            
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
