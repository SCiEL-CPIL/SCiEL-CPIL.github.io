import numpy as np
import time
from scipy.optimize import nnls

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0
    
    # Schoolbook rank-50 solution as fallback
    R_max = n * m * p
    U_school = np.zeros((n * m, R_max), dtype=np.float64)
    V_school = np.zeros((m * p, R_max), dtype=np.float64)
    W_school = np.zeros((p * n, R_max), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = R_max
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Strategy: for a given rank R, we can try to decompose by fixing W and solving for U, V
    # via a structured approach. But this is complex.
    
    # Alternative: use the fact that for matrix multiplication, we can use Strassen-like 
    # or other algorithmic decompositions. But for 2x5x5, let's try a more systematic search.
    
    # Key insight: The tensor can be viewed as a 3-tensor. We can try to find a decomposition
    # by iteratively constructing rank-1 terms that best match the remaining tensor.
    
    def compute_residual(U, V, W, R):
        """Compute T_target - sum_r U[:,r] * V[:,r] * W[:,r]"""
        T_calc = np.einsum('ar,br,cr->abc', U[:, :R], V[:, :R], W[:, :R])
        return T_target - T_calc
    
    def verify_decomposition(U, V, W, R):
        """Check if the decomposition exactly matches T_target"""
        T_calc = np.einsum('ar,br,cr->abc', U[:, :R], V[:, :R], W[:, :R])
        return np.allclose(T_calc, T_target, atol=1e-8)
    
    # For rank R, try a greedy approach:
    # 1. Start with residual = T_target
    # 2. For each of R steps, find the best rank-1 term that reduces the residual
    # 3. This is NP-hard in general, but for small tensors we can try structured candidates
    
    # Better approach: Use the structure of matrix multiplication.
    # For M_{n,m,p}, we know that the schoolbook algorithm uses n*m*p terms.
    # We can try to combine terms using identities like (a+b)(c+d) = ac + ad + bc + bd
    # But this increases rank, not decreases it.
    
    # The actual rank of M_{2,5,5}: 
    # - Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # - Upper bound: 50 (schoolbook)
    # - For M_{2,2,2}, rank = 7 < 8 = 2*2*2
    # - For M_{1,m,p}, rank = m*p (no savings possible)
    # - For M_{2,5,5}, we can think of it as two M_{1,5,5} problems.
    #   If we could share some structure between the two rows, we might save rank.
    
    # Let's try a different structural approach:
    # The tensor T[i,j,k] = 1 if the chain is consistent.
    # We can try to use polynomial identity-based decompositions.
    
    # Actually, let's try a more direct search with better initialization:
    # For rank R, we can try random search but with more attempts and better verification.
    
    # First, let's try ranks from 49 down to 25
    for R in range(min(49, R_max - 1), 24, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Number of attempts scales inversely with R (smaller R is harder)
        max_attempts = max(50, 2000 - 30 * R)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random search with small integers
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            if np.array_equal(T_calc.astype(np.int64), T_target.astype(np.int64)):
                found = True
                best_rank = R
                best_U = U.astype(np.float64)
                best_V = V.astype(np.float64)
                best_W = W.astype(np.float64)
                break
        
        if found:
            continue
    
    # If no lower rank found, try a more systematic approach using the structure
    # of the matrix multiplication tensor.
    
    # For M_{2,5,5}, consider the following:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # We can try to find a decomposition that exploits linearity.
    # Define new variables that are linear combinations of A and B entries.
    
    # One approach: use the fact that for each k, we're computing two dot products.
    # For fixed k, C[:,k] = A @ B[:,k]
    # This is a 2x5 matrix times a 5x1 vector.
    
    # The tensor for this is T[i,j,k] which equals 1 when (i,j,k) form a valid triple.
    
    # Let's try a structured search: for rank R, we can try to find U, V, W
    # where the columns have a specific structure.
    
    # For now, if we haven't found a better rank, return the schoolbook solution.
    
    # Ensure we return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
