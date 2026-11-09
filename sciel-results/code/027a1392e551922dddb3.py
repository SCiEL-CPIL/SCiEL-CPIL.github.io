import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def try_structured_decomposition():
        """
        Try to find a structured low-rank decomposition by exploiting
        the tensor's algebraic structure.
        
        Key idea: For each i (output row), the relevant tensor entries are
        T[i*m+j, j*p+k, k*n+i] for j=0,1,2 and k=0,...,6.
        
        We can try to factor this using shared U and W factors across k.
        """
        if time.time() > deadline:
            return None
        
        # Strategy: Use a block-diagonal approach
        # For the 3x3 x 3x7 multiplication, think of it as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # We can use Strassen-like ideas on the 3x3 part.
        # 
        # Alternative: Try to find U, V, W with entries in {0, 1, -1, 0.5}
        # that satisfy the tensor equation.
        #
        # Let's try a systematic search over small structured families.
        
        # Approach 1: Group k indices into pairs and use shared factors
        # For k pairs (0,1), (2,3), (4,5), (6 alone)
        # This might not save much.
        
        # Approach 2: Use the fact that for fixed i, we need:
        # sum_j A[i,j] * B[j,k] = C[i,k] for all k
        # This is a matrix-vector product for each i.
        
        # Let's try a direct constructive approach:
        # Use the known fact that 3x3 matrix multiplication can be done
        # with rank 23 (or lower with more clever methods).
        
        # Actually, let's try a different structural approach:
        # Decompose based on the diagonal/off-diagonal structure.
        
        # For now, let's try to find a decomposition with rank < 63
        # by using a greedy approach with structured initialization.
        
        # Initialize with a pattern that groups terms
        np.random.seed(42)
        
        # Try rank 50 first with structured initialization
        for r in [55, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30]:
            if time.time() > deadline:
                break
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Structured initialization:
            # For each (i,j) pair, create r/(n*m) columns that cover all k
            cols_per_ij = r // (n * m)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
            
            # If we used all columns, this is just schoolbook
            # Try to merge some columns
            if col < r:
                # We have extra columns, use them to create combined terms
                for r_idx in range(col, r):
                    if time.time() > deadline:
                        break
                    # Create a term that covers multiple (i,j,k) triples
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k1 = np.random.randint(0, p)
                    k2 = np.random.randint(0, p)
                    if k1 != k2:
                        U[i * m + j, r_idx] = 1
                        V[j * p + k1, r_idx] = 1
                        W[k1 * n + i, r_idx] = 0.5
                        V[j * p + k2, r_idx] = 1
                        W[k2 * n + i, r_idx] = 0.5
                        # This doesn't work directly, need to adjust
                        # Actually this creates T[i*m+j, j*p+k1, k1*n+i] += 0.5
                        # and T[i*m+j, j*p+k2, k2*n+i] += 0.5
                        # So we need to compensate elsewhere
        
            if check_decomposition(U, V, W):
                return (U, V, W)
        
        return None
    
    def try_gradient_descent(r, max_iters=3000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute T_rec efficiently
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            err_norm = np.max(np.abs(error))
            
            # Track best
            if err_norm < best_error:
                best_error = err_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Early termination if converged
            if err_norm < 1e-8:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence every 100 iterations
            if it % 100 == 0:
                if err_norm < 1e-8:
                    return (U, V, W)
        
        # Final check
        if check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
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
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # First try structured approach
    result = try_structured_decomposition()
    if result is not None:
        U, V, W = result
        best_rank = U.shape[1]
        best_result = (U, V, W)
    
    # Then try gradient descent with various ranks
    if time.time() < deadline:
        ranks_to_try = list(range(62, 25, -1))
        for r in ranks_to_try:
            if time.time() > deadline:
                break
            if r >= best_rank:
                continue
                
            result = try_gradient_descent(r, max_iters=2000, lr=0.01)
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
