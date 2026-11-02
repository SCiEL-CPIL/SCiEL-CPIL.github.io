import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def _strassen_2x2(A, B):
    """
    Compute A*B where A is 2x2 and B is 2x2 using Strassen's algorithm.
    Returns the 2x2 product.
    """
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    e, f = B[0, 0], B[0, 1]
    g, h = B[1, 0], B[1, 1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    return np.array([
        [m1 + m4 - m5 + m7, m3 + m5],
        [m2 + m4, m1 - m2 + m3 + m6]
    ])

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    # Fallback: Schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    # Strategy: Use Strassen-like block decomposition
    # Split 3x4 into two 3x2 blocks, and 4x6 into two 2x6 blocks
    # For each pair of 3x2 and 2x6 blocks, we can use a rank-12 algorithm
    # based on Strassen for 2x2 multiplication applied to the 2-dimensional middle
    
    # The idea: C = A*B where A is 3x4, B is 4x6
    # Split A = [A1, A2] where A1, A2 are 3x2
    # Split B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    
    # For each term Ai*Bj, we can decompose using Strassen-like approach
    # Actually, let's think differently.
    
    # A better approach: Use the fact that 3x4x6 can be decomposed by
    # partitioning the middle dimension into two 2-dimensions
    
    # Let's try a direct rank-24 construction using Strassen on 2x2 blocks
    
    # Actually, the cleanest approach: 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Split j into j1 in {0,1} and j2 in {2,3}
    # C[i,k] = sum_{j1=0}^1 A[i,j1]*B[j1,k] + sum_{j2=0}^1 A[i,2+j2]*B[2+j2,k]
    
    # For the first sum (2 terms), we can use a rank-2 decomposition (trivial)
    # But we want to use Strassen-style rank reduction.
    
    # Let me reconsider. The rank of 2x2x2 matrix multiplication is 7 (Strassen).
    # So if we can reduce our problem to multiple 2x2x2 problems, we can use rank 7 each.
    
    # Alternative: Use known results. The rank of 3x4x6 is known to be at most 24.
    # Let me try to construct it explicitly.
    
    # Actually, let's just try to find a low-rank decomposition numerically,
    # but start from a better initial guess based on block structure.
    
    # For now, let's try ranks from 24 upward to see if we can find a solution
    
    for r in range(24, 35):
        if time.time() > deadline:
            break
            
        # Try to find a rank-r decomposition
        # Initialize with random values scaled appropriately
        np.random.seed(42 + r)
        U_init = np.random.randn(n * m, r) * 0.5
        V_init = np.random.randn(m * p, r) * 0.5
        W_init = np.random.randn(p * n, r) * 0.5
        
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[0:size_u*r].reshape(size_u, r)
            V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
            W = params[size_u*r + size_v*r:].reshape(size_w, r)
            
            D = np.zeros((n, m, p))
            for c in range(r):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
            
            return (D - 1.0).flatten()
        
        try:
            result = least_squares(residuals, params0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
            
            if result.cost < 1e-8:
                U_sol = result.x[0:size_u*r].reshape(size_u, r)
                V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                W_sol = result.x[size_u*r + size_v*r:].reshape(size_w, r)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify the snapped solution
                D_check = np.zeros((n, m, p))
                for c in range(r):
                    Uc = U_snap[:, c].reshape(n, m)
                    Vc = V_snap[:, c].reshape(m, p)
                    Wc = W_snap[:, c].reshape(p, n)
                    D_check += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
                
                if np.allclose(D_check, 1.0, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
        except Exception:
            pass
            
    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
