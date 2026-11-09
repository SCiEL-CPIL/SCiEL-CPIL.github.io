import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
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
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def try_algebraic_decomposition(r):
        """Try to find a rank-r decomposition using structured algebraic approach."""
        if time.time() > deadline:
            return None
        
        # For small ranks, try to find a structured solution
        # Use the fact that the tensor has a specific pattern
        
        # Try using Kronecker product structure
        # The multiplication tensor can be decomposed using block structure
        
        # Approach: Try to find U, V, W with entries in {0, 1, -1, 0.5}
        # that satisfy the tensor equation
        
        # Use a greedy approach: try to cover the tensor with rank-1 terms
        # Each rank-1 term corresponds to a triple (i,j,k) pattern
        
        # For the 3x3 x 3x7 case, let's try to find a structured decomposition
        
        # Key insight: The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We can try to group terms that share structure
        
        # Try using the fact that for fixed i, the j and k dimensions are coupled
        # For each i, we need to cover a 3x7 sub-matrix
        
        # Let's try a decomposition based on the structure:
        # For each i in range(3), we have a 3x7 matrix multiplication
        # We can use Strassen-like ideas for the 3x3 blocks
        
        # Actually, let's try a more systematic approach:
        # Use the known lower bound and try to construct a valid decomposition
        
        # For 3x3 x 3x7, the lower bound is at least max(9, 21, 21) = 21
        # But the actual rank is likely higher
        
        # Let's try to find a decomposition with rank around 40-50
        
        # Use a heuristic: try to find U, V, W with sparse structure
        # that approximately covers the tensor, then refine
        
        # Initialize with random sparse matrices
        np.random.seed(42)
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Try to assign non-zero entries strategically
        # Each column of U, V, W should correspond to a "pattern" that covers multiple tensor entries
        
        for r_idx in range(r):
            if time.time() > deadline:
                return None
            
            # Choose random positions for non-zero entries
            # But bias towards positions that cover more tensor entries
            
            # For U: positions (i, j) where i in 0..2, j in 0..2
            # For V: positions (j, k) where j in 0..2, k in 0..6
            # For W: positions (k, i) where k in 0..6, i in 0..2
            
            # Try to create a pattern that covers a "block" of the tensor
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k_start = np.random.randint(0, p - 1)
            k_end = min(k_start + 2, p)
            
            # Set U[i*m+j, r_idx] = 1
            U[i * m + j, r_idx] = 1
            
            # Set V[j*p+k, r_idx] for k in k_start..k_end
            for k in range(k_start, k_end):
                V[j * p + k, r_idx] = 1
            
            # Set W[k*n+i, r_idx] for k in k_start..k_end
            for k in range(k_start, k_end):
                W[k * n + i, r_idx] = 1
        
        # Check if this works
        if check_decomposition(U, V, W):
            return (U, V, W)
        
        return None
    
    def try_gradient_descent(r, max_iters=5000, lr=0.005):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute T_rec efficiently
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            
            # Early termination if converged
            if np.max(np.abs(error)) < 1e-8:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                # grad_U[:, r_idx] = sum_{b,c} error[:, b, c] * V[b, r_idx] * W[c, r_idx]
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                # grad_V[:, r_idx] = sum_{a,c} error[a, :, c] * U[a, r_idx] * W[c, r_idx]
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                # grad_W[:, r_idx] = sum_{a,b} error[a, b, :] * U[a, r_idx] * V[b, r_idx]
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence every 100 iterations
            if it % 100 == 0:
                if np.max(np.abs(error)) < 1e-8:
                    return (U, V, W)
        
        # Final check
        if check_decomposition(U, V, W):
            return (U, V, W)
        
        return None
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from schoolbook down to a reasonable lower bound
    # The theoretical lower bound is max(nm, mp, pn) = max(9, 21, 21) = 21
    # But for matrix multiplication, the rank is typically higher
    
    # Try a few key ranks
    ranks_to_try = [62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try algebraic approach first
        result = try_algebraic_decomposition(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        # Try gradient descent
        result = try_gradient_descent(r, max_iters=2000, lr=0.005)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
