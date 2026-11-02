"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    
    The output tensor W has indices W[k*n + i, r] where:
    - i is the row index of A (0..n-1)
    - k is the column index of B (0..p-1) 
    - r is the rank index
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = 115.0 # 5s safety margin
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid exact decomposition."""
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We need T[i,j,k] == 1 for all i,j,k
        
        rank = U.shape[1]
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    # Check if total is close to 1.0
                    # Since we want integer/half-integer, if we rounded, it should be exact.
                    # But due to float precision, we check closeness.
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        return valid

    def try_rank(r, max_attempts=500):
        """
        Try to find a rank-r decomposition using local optimization and rounding.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            # Using a mix of integers and half-integers in the initialization might help
            # But gradient descent works on reals.
            
            # Scale: The product U*V*W should be ~1. 
            # If U,V,W are ~1, product is ~1.
            # Let's initialize with uniform distribution in [-2, 2]
            
            U = np.random.uniform(-2, 2, size=(n*m, r))
            V = np.random.uniform(-2, 2, size=(m*p, r))
            W = np.random.uniform(-2, 2, size=(p*n, r))
            
            # Gradient descent to minimize sum of squared errors
            # Error = sum_{i,j,k} (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
            
            # This is a non-convex problem, but we can try a few steps.
            # A simple gradient step might be too slow. 
            # Instead, let's try a "simulated annealing" like approach or just random restarts with rounding.
            
            # Actually, a better heuristic:
            # 1. Randomly initialize.
            # 2. Perform a few steps of coordinate descent or gradient descent.
            # 3. Round to nearest 0.5.
            # 4. Check.
            
            # Let's implement a simple gradient step for a few iterations.
            # This is computationally expensive if done naively.
            
            # Alternative: Just try random integer/half-integer vectors with a smarter distribution.
            # The previous code tried integers in [-3,3].
            # Let's try half-integers in a wider range, say [-4, 4].
            
            # Generate random half-integers
            U_int = np.random.randint(-8, 9, size=(n*m, r))
            V_int = np.random.randint(-8, 9, size=(m*p, r))
            W_int = np.random.randint(-8, 9, size=(p*n, r))
            
            U = U_int / 2.0
            V = V_int / 2.0
            W = W_int / 2.0
            
            if check_decomposition(U, V, W):
                return U_int, V_int, W_int
        
        return None

    # Try ranks from 20 up to 23
    # Rank 24 is the schoolbook, which we can construct directly.
    
    # First, let's ensure we have a valid fallback.
    # Construct schoolbook rank 24
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower ranks
    # We use random search with half-integers.
    # To make it faster, we can limit the number of attempts based on time.
    
    for r in range(20, 24):
        if time.time() > deadline - 5.0:
            break
            
        # Increase attempts as we go higher in rank? No, lower rank is harder.
        # Let's give more attempts to lower ranks.
        attempts = 2000 if r == 20 else 1000
        
        result = try_rank(r, attempts)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            # We found a valid one, can we go lower?
            # The loop goes 20, 21, 22, 23.
            # If we find 20, we break? No, we want minimal.
            # But 20 is the lowest we try.
            # If we find 20, we can break immediately as it's the best possible in our search space.
            if r == 20:
                break
        
    U, V, W = best_result
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
