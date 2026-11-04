"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # The tensor T has entries T[i, j, k] = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # In the standard decomposition: T = sum_r U[:, r] ⊗ V[:, r] ⊗ W[:, r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, r] corresponds to A_{i,j}
    # V[j*p + k, r] corresponds to B_{j,k}  
    # W[k*n + i, r] corresponds to (AB)_{i,k}
    
    # The tensor T is defined by: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # More precisely, the multiplication tensor has a non-zero entry at position (i*m+j, j*p+k, k*n+i) = 1
    
    # Known lower bounds: For matrix multiplication C = A*B where A is n×m, B is m×p
    # The rank is at least max(n*m, m*p, n*p) and at most n*m*p
    # For 3×5 × 5×6: lower bound = max(15, 30, 18) = 30, upper bound = 90
    
    # Let's try to find a decomposition with rank close to the lower bound
    # We'll use a randomized approach with verification
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        n, m, p = DIMENSIONS
        
        # Build the tensor from the decomposition
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # The multiplication tensor has 1s at specific positions
        # Let's check by reconstructing the multiplication
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The entry T[i*m+j, j*p+k, k*n+i] should be 1
                    # And all other entries should be 0
                    
                    # Compute the sum for this specific (i,j,k) triple
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    if abs(val - 1.0) > tol:
                        return False
        
        # Also check that cross terms are zero
        # For any (i1,j1,k1) and (i2,j2,k2) where the indices don't match,
        # the tensor entry should be 0
        # Actually, let's just verify the full tensor is correct
        # The tensor T has entries T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
        # Otherwise T[a,b,c] = 0
        
        # Let's build the expected tensor and compare
        expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute actual tensor from decomposition
        actual = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            actual += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        if not np.allclose(actual, expected, atol=tol):
            return False
        
        return True
    
    # Start with schoolbook and try to reduce rank
    # Schoolbook rank = n*m*p = 90
    # Try ranks from lower bound upward
    
    # Known: For rectangular matrix multiplication, there are better decompositions
    # Let's try a structured approach
    
    # First, let's try the trivial schoolbook to ensure we have a valid baseline
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
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    best_result = (U0, V0, W0, n*m*p)
    
    # Try to find lower rank decompositions using random initialization + optimization
    # We'll use a simple approach: random Gaussian initialization, then try to drive
    # the residual to zero using gradient descent on the log-sum-exp of squared residuals
    
    def try_random_decomposition(rank, max_iters=5000, lr=0.01, seed=None):
        """Try to find a rank-r decomposition via random init + gradient descent."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # The target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            residual = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                residual += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            residual -= T
            
            # Compute gradient
            # d/dU[a,r] sum_{b,c} residual[a,b,c]^2 = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            # This is expensive. Let's use a simpler approach.
            
            # Actually, for the non-zero entries of T, we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For all other (a,b,c), the sum should be 0.
            
            # Let's use a different strategy: since the tensor is sparse,
            # we can focus on the constraints.
            
            # There are n*m*p = 90 non-zero constraints, and (n*m + m*p + p*n)*rank variables
            # For rank=30: (15+30+18)*30 = 63*30 = 1890 variables, 90 constraints
            # This is highly underdetermined, so solutions should exist if rank >= 30
            
            # Let's just check if we can find a solution by solving the linear system
            # for the non-zero entries
            pass
        
        return None  # Not using this approach directly
    
    # Better approach: The tensor is a "matrix multiplication" tensor.
    # For C = AB, we can write this as a sum of rank-1 terms.
    # The key insight: if we can factor the middle dimension m, we might reduce rank.
    
    # Let's try a structured decomposition based on the structure of the problem.
    # For A (n×m) times B (m×p):
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # One approach: group the j indices.
    # If m is composite, we might be able to use a Kronecker product structure.
    # m=5 is prime, so this doesn't help directly.
    
    # Let's try to find a decomposition with rank = max(n*p, m*max(n,p)) or similar.
    # Actually, let's just try to verify if rank 30 is achievable.
    
    # For rank r, we need to find U (n*m × r), V (m*p × r), W (p*n × r) such that
    # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in (n*m + m*p + p*n)*r unknowns.
    # Most equations are trivially satisfied (both sides are 0).
    # The non-trivial ones are the n*m*p = 90 equations where T=1.
    
    # So we need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # And for all other (a,b,c) combinations, the sum should be 0.
    
    # Let's try a greedy approach: for each (i,j,k), assign a unique rank index.
    # That gives rank = 90 (schoolbook).
    
    # To reduce rank, we need to share rank indices between different (i,j,k) triples.
    # This is possible if the "pattern" of non-zero entries allows it.
    
    # Let's try rank values from 30 to 90 and see which ones work.
    # We'll use a randomized local search.
    
    def random_search(rank, num_restarts=10, max_iters_per_restart=2000):
        """Randomized search for a valid rank-r decomposition."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 1000)
            
            # Initialize with random values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Use simple gradient descent on the constraint violations
            lr = 0.001
            for it in range(max_iters_per_restart):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual for the non-zero entries
                residuals = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residuals.append(val - 1.0)
                
                # Also check some random zero entries
                for _ in range(20):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    # Check if this is a non-zero entry
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k2 = b % p
                    k1 = c // n
                    i2 = c % n
                    if not (i1 == i2 and j1 == j2 and k1 == k2):
                        val = 0.0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        residuals.append(val)
                
                residuals = np.array(residuals)
                loss = np.sum(residuals**2)
                
                if loss < 1e-10:
                    return U, V, W
                
                # Compute gradients (approximate)
                # This is expensive, so let's use a simpler update
                # Just do a small random perturbation biased towards reducing loss
                if it % 100 == 0:
                    # Check if we're making progress
                    pass
                
                # Simple stochastic update
                idx = np.random.randint(0, len(residuals))
                if residuals[idx] != 0:
                    # Perturb one of the factors
                    factor = np.random.randint(0, 3)
                    if factor == 0:
                        a = idx // (m*p) if idx < n*m*p else np.random.randint(0, n*m)
                        # This is getting complicated. Let's use a different approach.
                        pass
                
                # Let's just do a simple random walk with occasional larger steps
                step_size = 0.01 * (1 + 0.1 * np.random.randn())
                if np.random.random() < 0.1:
                    step_size *= 5
                
                U += step_size * np.random.randn(n*m, rank)
                V += step_size * np.random.randn(m*p, rank)
                W += step_size * np.random.randn(p*n, rank)
            
            # Final check
            T_actual = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_actual += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            if np.allclose(T_actual, T, atol=1e-6):
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, n*p)  # 30
    upper_bound = n*m*p  # 90
    
    # Try a few specific ranks
    ranks_to_try = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = random_search(rank, num_restarts=5, max_iters_per_restart=500)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition(U, V, W):
                best_result = (U, V, W, rank)
                break  # Found the lowest rank
    
    # If we didn't find anything better, use schoolbook
    U, V, W, rank = best_result
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
