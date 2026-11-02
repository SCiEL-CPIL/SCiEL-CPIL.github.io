import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1
    
    # Schoolbook rank-50 solution as fallback
    R_max = n * m * p
    U_school = np.zeros((n * m, R_max), dtype=np.int64)
    V_school = np.zeros((m * p, R_max), dtype=np.int64)
    W_school = np.zeros((p * n, R_max), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = R_max
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Key insight for M_{2,5,5}:
    # We can decompose the 2x5x5 multiplication by exploiting the structure.
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid triple.
    # 
    # Strategy: Try to find a decomposition with rank < 50 using structured search.
    # For matrix multiplication, we know that:
    # - Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # - Upper bound: 50 (schoolbook)
    # 
    # Let's try a systematic approach:
    # 1. For each rank R from 49 down to 25, try to find a valid decomposition
    # 2. Use structured initialization based on the matrix multiplication structure
    
    def verify_decomposition(U, V, W, R):
        """Check if the decomposition exactly matches T_target"""
        T_calc = np.einsum('ar,br,cr->abc', U[:, :R], V[:, :R], W[:, :R])
        return np.array_equal(T_calc, T_target)
    
    # Structured approach: 
    # The matrix multiplication tensor for M_{n,m,p} has a specific structure.
    # For M_{2,5,5}, we can try to combine terms.
    
    # One key identity: (a+b)(c+d) = ac + ad + bc + bd
    # But this increases rank. We need to find ways to SHARE terms.
    
    # Let's try a different approach: use the fact that the tensor is sparse
    # and try to find a low-rank decomposition using iterative methods.
    
    # For rank R, we can try to solve:
    # T_target[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
    
    # This is a 3-way tensor decomposition problem.
    # We can use ALS (Alternating Least Squares) with integer constraints.
    
    # Better: try structured candidates based on known matrix multiplication algorithms.
    
    # For M_{2,5,5}, let's try to use a block structure:
    # A = [A0; A1] where A0, A1 are 1x5 vectors
    # B is 5x5
    # C0 = A0 * B (1x5)
    # C1 = A1 * B (1x5)
    
    # The tensor for C0 is T0[i,j,k] = 1 if i=0, and T1[i,j,k] = 1 if i=1.
    
    # We can try to find a common decomposition for both.
    
    # Let's try random search with better initialization:
    # Start with schoolbook and try to merge terms.
    
    # For each pair of schoolbook terms, check if they can be merged.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    # coefficients such that the sum of the two rank-1 tensors equals a single rank-1 tensor.
    # This is generally not possible, but for specific structures it might be.
    
    # Alternative: use the fact that for fixed j, we have terms for all i,k.
    # For fixed j, the sub-tensor is T[i,j,k] which is 1 for all valid (i,k).
    
    # Let's try a more direct approach: random search with small integers,
    # but with a much larger search space and better pruning.
    
    # For rank R, the number of free parameters is R * (n*m + m*p + p*n) = R * 45
    # For R=25, that's 1125 parameters. This is too many for brute force.
    
    # Let's try a hybrid approach:
    # 1. Fix W to be a random integer matrix
    # 2. Solve for U and V using linear algebra
    # 3. Check if the solution is integer-valued
    
    def try_fixed_W(W, R):
        """Given W, try to find U and V such that the decomposition is exact."""
        # We need: T_target[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
        # For fixed alpha, beta: sum_r U[alpha, r] * V[beta, r] * W[gamma, r] = T_target[alpha, beta, gamma]
        # This is a linear system in U[alpha, r] * V[beta, r]
        
        # Let X[alpha, beta, r] = U[alpha, r] * V[beta, r]
        # Then: sum_r X[alpha, beta, r] * W[gamma, r] = T_target[alpha, beta, gamma]
        
        # For each (alpha, beta), we have a system:
        # W^T * X[alpha, beta, :] = T_target[alpha, beta, :]
        
        # If W is invertible (or has full row rank), we can solve for X.
        # But W is (p*n, R) = (10, R). For R <= 10, W can be full rank.
        # For R > 10, the system is underdetermined.
        
        # Let's try R <= 10 first (though the lower bound is 25, so this won't work).
        
        # Actually, the lower bound is 25, so R >= 25.
        # For R >= 25, W is (10, R) which has rank at most 10.
        # The system W^T * x = b has solutions only if b is in the column space of W^T.
        
        # This approach is complex. Let's try a different strategy.
        
        return None
    
    # Let's try a more practical approach:
    # Use the known lower bound of 25 and try to find a rank-25 decomposition.
    # If that fails, try 26, 27, etc.
    
    # For rank R, we can try to use the following structure:
    # U[alpha, r] where alpha = i*m + j, r = 0..R-1
    # V[beta, r] where beta = j*p + k, r = 0..R-1
    # W[gamma, r] where gamma = k*n + i, r = 0..R-1
    
    # Let's try to use a greedy algorithm:
    # 1. Start with residual = T_target
    # 2. For each of R steps, find the best rank-1 term that reduces the residual
    # 3. Use a heuristic to choose the rank-1 term
    
    def greedy_decomposition(R, max_iter=1000):
        """Try to find a rank-R decomposition using greedy approach."""
        residual = T_target.copy()
        U = np.zeros((n * m, R), dtype=np.int64)
        V = np.zeros((m * p, R), dtype=np.int64)
        W = np.zeros((p * n, R), dtype=np.int64)
        
        for r in range(R):
            if time.time() > deadline:
                return None
            
            # Find the best rank-1 term that reduces the residual
            # Try all possible rank-1 terms with small integer entries
            best_term = None
            best_score = np.inf
            
            # For each possible (u, v, w) with entries in {-1, 0, 1}
            # This is too many: 3^45 possibilities
            # Instead, try a subset of structured terms
            
            # Try terms based on the structure of matrix multiplication
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Try the schoolbook term
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i] = 1
                        
                        term = np.einsum('a,b,c->abc', u, v, w)
                        score = np.sum((residual - term) ** 2)
                        
                        if score < best_score:
                            best_score = score
                            best_term = (u, v, w)
            
            if best_term is None:
                return None
            
            u, v, w = best_term
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            residual = residual - np.einsum('a,b,c->abc', u, v, w)
        
        # Verify
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if np.array_equal(T_calc, T_target):
            return U, V, W
        return None
    
    # Try greedy for ranks from 25 to 49
    for R in range(25, 50):
        if time.time() > deadline:
            break
        
        result = greedy_decomposition(R)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If greedy didn't find a better rank, try random search for lower ranks
    for R in range(25, min(50, best_rank)):
        if time.time() > deadline:
            break
        
        # Random search with small integers
        max_attempts = max(100, 1000 - 10 * R)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            if np.array_equal(T_calc, T_target):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == R:
            break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
