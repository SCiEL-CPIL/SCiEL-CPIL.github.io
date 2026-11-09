import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    timeout = 140.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # Trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0
    
    def compute_T(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T
    
    def verify_exact(U, V, W, R):
        """Verify exact decomposition using rational arithmetic"""
        try:
            U_frac = np.array([[Fraction(int(round(u * 2))) / 2 for u in row] for row in U])
            V_frac = np.array([[Fraction(int(round(v * 2))) / 2 for v in row] for row in V])
            W_frac = np.array([[Fraction(int(round(w * 2))) / 2 for w in row] for row in W])
            
            U_r = U_frac.reshape(n, m, R)
            V_r = V_frac.reshape(m, p, R)
            W_r = W_frac.reshape(p, n, R)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(R):
                            s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                        if s != Fraction(1):
                            return False
            return True
        except:
            return False
    
    def try_reduce_rank(U, V, W, R, max_attempts=50):
        """Try to reduce rank by finding linear dependencies"""
        if time.time() - start_time > timeout - 10:
            return None, None, None, R
        
        # Try to find a linear combination of columns that can be eliminated
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 10:
                return None, None, None, R
            
            # Randomly select a subset of columns to try to eliminate
            if R <= 1:
                break
            
            # Try removing one column at a time
            for col_to_remove in range(R):
                if time.time() - start_time > timeout - 10:
                    return None, None, None, R
                
                U_test = np.delete(U, col_to_remove, axis=1)
                V_test = np.delete(V, col_to_remove, axis=1)
                W_test = np.delete(W, col_to_remove, axis=1)
                
                R_test = R - 1
                T_test = compute_T(U_test, V_test, W_test, R_test)
                
                # Check if still exact (allow small numerical error)
                if np.allclose(T_test, 1.0, atol=1e-8):
                    # Verify with rational arithmetic
                    if verify_exact(U_test, V_test, W_test, R_test):
                        return U_test, V_test, W_test, R_test
        
        return U, V, W, R
    
    # Start with trivial decomposition and try to reduce rank
    U_cur = U0.copy()
    V_cur = V0.copy()
    W_cur = W0.copy()
    R_cur = R0
    
    # Iteratively try to reduce rank
    for iteration in range(30):
        if time.time() - start_time > timeout - 15:
            break
        
        U_new, V_new, W_new, R_new = try_reduce_rank(U_cur, V_cur, W_cur, R_cur)
        
        if R_new < R_cur:
            U_cur, V_cur, W_cur, R_cur = U_new, V_new, W_new, R_new
            best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            best_rank = R_cur
            print(f"Reduced rank to {R_cur}")
        else:
            break
    
    # Try a few random perturbations and see if we can find lower rank
    for R_target in range(min(R_cur, 40), 20, -1):
        if time.time() - start_time > timeout - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() - start_time > timeout - 5:
                break
            
            np.random.seed(42 + attempt * 7)
            
            # Initialize with structure: start from trivial and add noise
            U = np.random.randn(size_U, R_target) * 0.5
            V = np.random.randn(size_V, R_target) * 0.5
            W = np.random.randn(size_W, R_target) * 0.5
            
            # Round to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            T = compute_T(U, V, W, R_target)
            
            if np.allclose(T, 1.0, atol=1e-8):
                if verify_exact(U, V, W, R_target):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R_target
                    found = True
                    break
        
        if found:
            break
    
    return best_result
