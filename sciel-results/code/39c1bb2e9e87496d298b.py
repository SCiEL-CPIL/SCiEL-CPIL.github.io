"""Rewritten program for rank-minimal exact tensor decomposition of 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 18)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i] = 1 if j==j' and i==i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Known lower bound for rank of M_{2,3,3} is 10 (from literature).
    # We'll try to find exact integer/half-integer decompositions for ranks 10, 11, 12, ...
    # Strategy: use randomized search with rounding to nearest half-integer.
    # For small dimensions, we can use a structured approach.
    
    # Key insight: the tensor has a specific structure. We can try to find decompositions
    # by searching over small sets of candidate vectors.
    
    # For each rank r, we need to find U (6 x r), V (9 x r), W (6 x r)
    # such that sum_c U[:,c] outer V[:,c] outer W[:,c] = T
    
    # We'll use a hybrid approach:
    # 1. Try known constructions from the literature (if we can encode them)
    # 2. Use randomized local search with half-integer rounding
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=1e-10)
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2.0
    
    def try_rank(rank, max_iter=50000):
        """Try to find a valid decomposition of given rank."""
        if time.time() > deadline - 10:
            return None
        
        # Multiple random restarts
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + rank * 1000 + restart)
            
            # Initialize with small random values, biased toward half-integers
            U = np.random.randint(-2, 3, size=(n*m, rank)) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, rank)) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, rank)) / 2.0
            
            # Local search: try to minimize error by perturbing individual entries
            best_error = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                diff = T - T_approx
                error = np.sum(diff**2)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-10:
                    # Found exact solution
                    U_rounded = round_to_half(U)
                    V_rounded = round_to_half(V)
                    W_rounded = round_to_half(W)
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    break
                
                # Try a random perturbation
                # Pick a random matrix and entry to perturb
                choice = np.random.randint(0, 3)
                if choice == 0:
                    idx = (np.random.randint(n*m), np.random.randint(rank))
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    U[idx] += delta
                elif choice == 1:
                    idx = (np.random.randint(m*p), np.random.randint(rank))
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    V[idx] += delta
                else:
                    idx = (np.random.randint(p*n), np.random.randint(rank))
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    W[idx] += delta
            
            # After local search, check best solution
            if best_error < 1e-10:
                U_rounded = round_to_half(best_U)
                V_rounded = round_to_half(best_V)
                W_rounded = round_to_half(best_W)
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound for M_{2,3,3}: known to be 10 from algebraic complexity literature
    for rank in range(10, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
