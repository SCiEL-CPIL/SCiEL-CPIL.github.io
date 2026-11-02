"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a randomized search with local optimization to find a low-rank decomposition.
    The schoolbook rank is n*m*p = 48. We aim to find something lower.
    
    Strategy:
    1. Start with random initializations of U, V, W at various ranks
    2. Use alternating least squares / gradient-based optimization
    3. Try to find exact (or near-exact) solutions
    4. Use time budget to explore multiple ranks and initializations
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    # Actually, the matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form, T[i,j,k] = 1 if we're doing the "multiply" operation
    # The tensor we want to decompose is T with T[i,j,k] = 1 for all valid i,j,k
    # But actually, the matrix multiplication tensor has T[i,j,k] = 1 when the
    # corresponding product term exists. For standard matrix mult, T[i,j,k] = 1 
    # for all (i,j,k) in the appropriate ranges.
    
    # The tensor T is of shape (n, m, p) with T[i,j,k] = 1 for all i,j,k
    # We want to find rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Actually, let me reconsider the tensor structure.
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor T has entries T[i,j,k] which correspond to the
    # coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build target tensor
    T = np.ones((n, m, p))
    
    def verify(U, V, W):
        """Verify that U, V, W decompose the tensor T."""
        r = U.shape[1]
        # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    result[i, j, k] = val
        return np.allclose(result, T, atol=1e-9)
    
    def verify_fast(U, V, W):
        """Faster verification using tensor operations."""
        r = U.shape[1]
        # Reshape for efficient computation
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Build the result tensor
        result = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
        return np.allclose(result, T, atol=1e-9)
    
    def verify_vectorized(U, V, W):
        """Vectorized verification."""
        r = U.shape[1]
        result = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # For each (i,j,k), add Uc[i,j] * Vc[j,k] * Wc[k,i]
            for i in range(n):
                for j in range(m):
                    result[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
        return np.max(np.abs(result - T))
    
    def error_metric(U, V, W):
        """Compute the max error of the decomposition."""
        r = U.shape[1]
        result = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    result[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
        return np.max(np.abs(result - T))
    
    # Try different ranks from low to high
    best_solution = None
    best_rank = n * m * p  # schoolbook rank
    
    # First, provide the schoolbook solution as fallback
    def schoolbook():
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
    
    U_school, V_school, W_school = schoolbook()
    best_solution = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Known lower bounds and upper bounds for matrix multiplication:
    # For 2x4 by 4x6, the rank is between some lower bound and 48.
    # Let's try ranks from a reasonable lower bound up to schoolbook.
    
    # A simple lower bound: the rank must be at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But tighter bounds exist. Let's try from 24 upward.
    
    start_time = time.time()
    time_limit = 130  # seconds, leave margin for the 150s limit
    
    # Try ranks from low to high
    for rank in range(24, best_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations for this rank
        max_iters_per_init = 500
        max_inits = 10
        
        for init in range(max_inits):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            np.random.seed(42 + init * 1000 + rank)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to get reasonable magnitudes
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Alternating least squares
            # For fixed V, W: solve for U
            # For fixed U, W: solve for V
            # For fixed U, V: solve for W
            
            # The equations are:
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # For fixed V, W, we have for each (i,j,k):
            # T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
            # This is a linear system in U for each fixed (j,k) pattern...
            # Actually, it's coupled. Let's use a simpler approach.
            
            # Use gradient descent on the objective:
            # f(U,V,W) = 0.5 * ||T - sum_c U_c ⊗ V_c ⊗ W_c||^2
            
            # But this is non-convex. Let's use a simpler alternating approach.
            
            # Actually, let's use a different approach: 
            # Treat it as a bilinear/trilinear optimization.
            
            # For now, let's try a simple random search with some structure.
            
            # Alternative: use the fact that we can fix one matrix and solve for others.
            # If we fix W, then for each c, we have a rank-1 update.
            
            # Let's try a more systematic approach using tensor train / CP decomposition ideas.
            
            # Actually, for small problems, let's just try many random initializations 
            # and check if any happen to be close to a solution.
            
            # Better: use a local search from random starts.
            # We'll use a simple coordinate descent / alternating minimization.
            
            # Define the tensor product operation
            def tensor_product(U, V, W):
                r = U.shape[1]
                result = np.zeros((n, m, p))
                for c in range(r):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            result[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
                return result
            
            # Alternating least squares
            # For fixed V, W: minimize ||T - sum_c U[:,c] ⊙ (V[:,c] ⊗ W[:,c] reshaped)||^2
            # where ⊙ denotes element-wise product after appropriate reshaping.
            
            # This is tricky because of the index coupling. Let's use gradient-based optimization.
            
            # Simple gradient descent
            lr = 0.01
            for it in range(max_iters_per_init):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute current tensor
                current = tensor_product(U, V, W)
                error = current - T
                
                # Compute gradients
                # d/dU[i*m+j,c] = sum_k error[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = error[i, j, k]
                                dU[i*m+j, c] += e * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += e * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += e * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if it % 50 == 0:
                    err = np.max(np.abs(error))
                    if err < 1e-9:
                        break
            
            # Check if we found a good solution
            err = error_metric(U, V, W)
            if err < 1e-9:
                best_solution = (U, V, W)
                best_rank = rank
                break  # Found a solution at this rank, try lower ranks
        
        if best_rank == rank:
            break  # Found solution at this rank
    
    # Return the best solution found
    U_best, V_best, W_best = best_solution
    
    # Convert to lists as required
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
