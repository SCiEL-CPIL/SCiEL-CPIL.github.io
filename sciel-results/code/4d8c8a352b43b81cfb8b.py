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
    
    # Search for lower rank
    # Try ranks from 25 to 49
    for target_rank in range(25, 50):
        if time.time() > deadline - 10:
            break
        
        # We will try to find a solution with entries in {-1, 0, 1}
        # This is a hard constraint satisfaction problem.
        # We use a random restart strategy.
        
        found = False
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            U = rng.randint(-1, 2, (n*m, target_rank)).astype(float)
            V = rng.randint(-1, 2, (m*p, target_rank)).astype(float)
            W = rng.randint(-1, 2, (p*n, target_rank)).astype(float)
            
            # Verify
            if verify_decomposition(U, V, W, T):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
