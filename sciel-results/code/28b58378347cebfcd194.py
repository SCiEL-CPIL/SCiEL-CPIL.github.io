import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build the multiplication tensor T[i,j,k] = delta(i,j,k) for C[i,k] = sum_j A[i,j]*B[j,k]
    # T is (n*m) x (m*p) x (p*n) = 16 x 24 x 24
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i, else 0
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Strategy: Try to find rank-r decomposition
    # Start with rank 70, try to reduce
    
    best_rank = None
    best_decomp = None
    
    # First, create a baseline schoolbook decomposition (rank 96)
    schoolbook_rank = n * m * p  # 96
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify_decomp(U, V, W):
        """Check if U,V,W form a valid decomposition of T"""
        rank = U.shape[1]
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_recon += np.einsum('a,b,c->abc', u, v, w)
        
        # Check if T_recon matches T
        return np.allclose(T_recon, T, atol=1e-6)
    
    # Try sparse linear combination approach
    # Start with a subset of schoolbook terms and optimize
    
    # For now, return schoolbook to ensure correctness
    # Then try to improve
    
    # Convert to lists
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
