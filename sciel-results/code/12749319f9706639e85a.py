import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the matrix multiplication tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_decomposition(U, V, W, T_target, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    # Schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m*p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p*n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    if not verify_decomposition(best_U, best_V, best_W, T):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    rng = np.random.RandomState(42)
    
    # Strategy: Try to find lower rank by exploiting structure
    # Key insight: M_{2,5,5} can potentially be decomposed using block structure
    
    # Approach 1: Try structured decompositions based on partitioning
    # Partition the 5 columns into 2+3, and 5 rows into 2+3
    
    # For C = A*B where A is 2x5, B is 5x5, C is 2x5
    # A = [A12 A23] (2x2, 2x3), B = [B12; B23] (2x5, 3x5)
    # C = A12*B12 + A23*B23
    
    # M_{2,2,5} sub-tensor: rank at most 20 schoolbook
    # M_{2,3,5} sub-tensor: rank at most 30 schoolbook
    
    # But we can try to find lower rank for these sub-problems
    
    # Let's try a more direct approach: random search with better patterns
    # Use sparse random matrices with various densities
    
    def try_random_decomposition(target_rank, n_iter=200):
        """Try to find a rank-target_rank decomposition using random search"""
        for attempt in range(n_iter):
            if time.time() > deadline - 15:
                return None, None, None, False
            
            # Choose random sparsity pattern
            density = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Choose random positions to place 1s
                u_idx = rng.choice(n*m, size=max(1, int(n*m * density)), replace=False)
                v_idx = rng.choice(m*p, size=max(1, int(m*p * density)), replace=False)
                w_idx = rng.choice(p*n, size=max(1, int(p*n * density)), replace=False)
                
                # Random signs
                u_sign = rng.choice([-1.0, 1.0], size=len(u_idx))
                v_sign = rng.choice([-1.0, 1.0], size=len(v_idx))
                w_sign = rng.choice([-1.0, 1.0], size=len(w_idx))
                
                U[u_idx, r] = u_sign
                V[v_idx, r] = v_sign
                W[w_idx, r] = w_sign
            
            if verify_decomposition(U, V, W, T):
                return U, V, W, True
        
        return None, None, None, False
    
    # Try ranks from 49 down to 30
    for target_rank in range(49, 29, -1):
        if time.time() > deadline - 20:
            break
        
        U, V, W, found = try_random_decomposition(target_rank, n_iter=50)
        if found:
            best_rank = target_rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break
    
    # If we haven't found improvement, try a more targeted approach
    # Look for linear dependencies in the schoolbook decomposition
    # This is equivalent to finding if the 50 rank-1 tensors are linearly dependent
    
    # Alternative: Try using known algorithms for small matrix multiplication
    # M_{2,5,5} might be related to M_{2,2,2} x M_{1,5,5} or similar
    
    # Try embedding: if we have a rank-r decomposition of M_{3,5,5}, 
    # we can restrict to first 2 rows to get M_{2,5,5}
    
    # For now, return best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
