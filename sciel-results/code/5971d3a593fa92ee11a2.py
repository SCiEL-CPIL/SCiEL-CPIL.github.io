import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Pre-flatten indices for faster access if needed, but T is small enough
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def rationalize_matrix(M, tol=1e-5):
        # M is a numpy array
        # Try to find rational approximations with denominator <= 4 (to cover half-integers and integers)
        # Actually, the problem says "integer/half-integer". So denominator 1 or 2.
        # But ALS might converge to values like 1/3? Unlikely for exact tensor decomposition of 0/1 tensor.
        # Let's try denominators 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 just in case, but prioritize 1,2.
        rows, cols = M.shape
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                found = False
                # Check integers and half-integers first
                for d in [1, 2]:
                    num = round(val * d)
                    frac = Fraction(num, d)
                    if abs(float(frac) - val) < tol:
                        result[i][j] = frac
                        found = True
                        break
                if not found:
                    # Try small denominators up to 10
                    for d in range(3, 11):
                        num = round(val * d)
                        frac = Fraction(num, d)
                        if abs(float(frac) - val) < tol:
                            result[i][j] = frac
                            found = True
                            break
                if not found:
                    return None
        return result

    def verify(U, V, W):
        # U, V, W are lists of lists of Fractions
        # Check if 2*entry is integer (half-integer check)
        # Actually, the problem statement says "integer/half-integer entries".
        # So we must ensure this.
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if (val * 2).denominator != 1:
                        return False
        
        # Check tensor product exactly
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0, 1)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
                        
        # Check zero entries
        # T[a,b,c] should be 0 if it's not a valid (i,j,k) path.
        # Valid path: a = i*m+j, b = j*p+k, c = k*n+i.
        # So for a given (a,b,c), extract i1, j1 from a; j2, k2 from b; k1, i2 from c.
        # It is non-zero ONLY IF i1==i2 AND j1==j2 AND k1==k2.
        # Wait, if i1==i2 and j1==j2, then k1 must equal k2 for the indices to align?
        # Let's check:
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        # If we pick arbitrary a, b, c:
        # i1 = a // m, j1 = a % m
        # j2 = b // p, k2 = b % p
        # k1 = c // n, i2 = c % n
        # The term is non-zero if (i1, j1, k2) is a valid triple and (k1, i2) matches?
        # Actually, the tensor T is defined such that T[a,b,c] = 1 iff there exists i,j,k such that
        # a=i*m+j, b=j*p+k, c=k*n+i.
        # This implies i1=i2, j1=j2, k1=k2.
        # So if i1 != i2 or j1 != j2 or k1 != k2, T[a,b,c] must be 0.
        
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                # If j1 != j2, then no common j, so T must be 0.
                if j1 != j2:
                    for c in range(p*n):
                        s = Fraction(0, 1)
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
                else:
                    # j1 == j2 == j
                    # Now check i and k
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        # Valid if i1 == i2 AND k1 == k2
                        if i1 == i2 and k1 == k2:
                            # This is the 1 case, already checked in the first loop?
                            # The first loop iterates i, j, k.
                            # a = i*m+j, b=j*p+k, c=k*n+i.
                            # So if i1==i2 and k1==k2, it IS the 1 case.
                            # We don't need to check it again, or we can skip it.
                            continue
                        else:
                            # This should be 0
                            s = Fraction(0, 1)
                            for r in range(rank):
                                s += U[a][r] * V[b][r] * W[c][r]
                            if s != 0:
                                return False
        return True

    def als_decompose(T, R, max_iters=100, tol=1e-8):
        np.random.seed(42) # Fixed seed for reproducibility in initial attempt, maybe vary later
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Pre-allocate arrays for speed
        # A_U will be (m*p * p*n) x R
        # We can build it on the fly or precompute the structure.
        
        for it in range(max_iters):
            # Update U
            for a in range(n*m):
                A = np.zeros((m*p * p*n, R))
                y = np.zeros(m*p * p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                
            # Update V
            for b in range(m*p):
                A = np.zeros((n*m * p*n, R))
                y = np.zeros(n*m * p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                
            # Update W
            for c in range(p*n):
                A = np.zeros((n*m * m*p, R))
                y = np.zeros(n*m * m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    # Search for minimal R
    # Start from a lower bound. 
    # Simple bound: n*m*p = 36.
    # Can we go lower?
    # Let's try R from 24 to 35.
    
    # To save time, we try higher ranks first? No, we want minimal.
    # But if 36 is the answer, we waste time on 24-35.
    # However, if we find 35, we save score.
    # Let's try a few restarts for each R.
    
    best_result = None
    
    # We will iterate R from 24 to 35.
    # If time is running out, break and return schoolbook.
    
    for R in range(24, 36):
        if time.time() > deadline - 15:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(20): # Increased restarts
            if time.time() > deadline - 5:
                break
            
            # Vary seed slightly
            np.random.seed(42 + restart)
            U, V, W, err = als_decompose(T, R, max_iters=50)
            
            if err < 1e-4: # Threshold for trying to rationalize
                # Try to rationalize
                U_frac = rationalize_matrix(U, tol=1e-5)
                V_frac = rationalize_matrix(V, tol=1e-5)
                W_frac = rationalize_matrix(W, tol=1e-5)
                
                if U_frac and V_frac and W_frac:
                    if verify(U_frac, V_frac, W_frac):
                        # Success
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        return {"U": U_out, "V": V_out, "W": W_out}
                        
    # Fallback to schoolbook
    return schoolbook()
