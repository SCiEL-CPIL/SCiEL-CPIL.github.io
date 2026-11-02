import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i, j, k] is 1 for all valid i, j, k
    # T_flat index: (i*m+j, j*p+k, k*n+i)
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Strassen-based construction for 2x4 x 4x10 (padding m=3 to 4)
    def strassen_construction():
        # Pad m from 3 to 4
        m_pad = 4
        nm_pad = n * m_pad  # 2*4 = 8
        mp_pad = m_pad * p  # 4*10 = 40
        pn = p * n  # 10*2 = 20
        
        # Strassen rank-7 decomposition for 2x2x2
        # The 7 rank-1 terms for computing C = A*B where A,B are 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For 2x4 x 4x10:
        # A is 2x4, split as [A1 | A2] where A1, A2 are 2x2
        # B is 4x10, split as [B1; B2] where B1, B2 are 2x10
        # C = A1*B1 + A2*B2
        
        # For each 2x2 x 2x10 product, we split the 10 cols into 5 pairs
        # Each 2x2 x 2x2 uses rank 7
        # So each 2x2 x 2x10 uses rank 35
        # Total: 2 * 35 = 70
        
        # This is worse than schoolbook (60), so we won't use this.
        # Instead, let's try a direct random search with more restarts.
        return None
    
    # Improved random search with more restarts and better initialization
    np.random.seed(42)
    
    # Try ranks from low to high, with more restarts for lower ranks
    max_rank_to_try = 55  # Stop before schoolbook
    
    for r in range(3, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # More restarts for lower ranks
        if r <= 10:
            max_restarts = 200
        elif r <= 20:
            max_restarts = 100
        elif r <= 30:
            max_restarts = 50
        else:
            max_restarts = 20
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U and V with random values
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            # Solve for W using least squares for each (k, i) pair
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-9:
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
