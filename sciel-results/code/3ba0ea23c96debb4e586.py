import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
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

    # Schoolbook fallback
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
        # Return list of lists of Fractions or None
        rows, cols = M.shape
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                # Use limit_denominator to find best rational approximation
                # If val is very close to 0, make it 0
                if abs(val) < tol:
                    result[i][j] = Fraction(0, 1)
                    continue
                
                # Try to find a rational with small denominator
                # First try simple fractions
                frac = Fraction(val).limit_denominator(1000)
                
                # Check if it's close enough
                if abs(float(frac) - val) < tol:
                    # Check if it's a half-integer
                    if (frac * 2).denominator == 1:
                        result[i][j] = frac
                    else:
                        # If not half-integer, it's invalid for this task
                        # But maybe we should allow it for now and check later?
                        # The task requires integer/half-integer.
                        # If limit_denominator gives 1/3, it's invalid.
                        # We should try to find a half-integer approximation?
                        # No, if the true solution is 1/3, we fail.
                        # But for matrix mult, solutions are often integers or half-integers.
                        # If we get 1/3, it's likely a convergence artifact.
                        # Let's mark as failure for this matrix.
                        return None
                else:
                    return None
        return result

    def verify(U, V, W):
        # U, V, W are lists of lists of Fractions
        # Check if 2*entry is integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not (val * 2).denominator == 1:
                        return False
        
        # Convert to float arrays for fast verification
        U_arr = np.array([[float(x) for x in row] for row in U])
        V_arr = np.array([[float(x) for x in row] for row in V])
        W_arr = np.array([[float(x) for x in row] for row in W])
        
        # Compute tensor product
        # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Compare with T
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def als_decompose(T, R, max_iters=50, tol=1e-8):
        n, m, p = DIMENSIONS
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        err = float('inf')
        for _ in range(max_iters):
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
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    # Search for minimal R
    # Lower bound heuristic: 18
    for R in range(18, 37):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        found = False
        # Increase restarts for smaller R as they are harder
        restarts = 20 if R < 25 else 10
        for restart in range(restarts):
            if time.time() > deadline - 5:
                break
            U, V, W, err = als_decompose(T, R, max_iters=40)
            if err < 1e-5:
                # Try to rationalize
                U_frac = rationalize_matrix(U, tol=1e-4)
                if U_frac:
                    V_frac = rationalize_matrix(V, tol=1e-4)
                    if V_frac:
                        W_frac = rationalize_matrix(W, tol=1e-4)
                        if W_frac:
                            if verify(U_frac, V_frac, W_frac):
                                # Convert to lists of floats/ints for output
                                U_out = [[float(x) for x in row] for row in U_frac]
                                V_out = [[float(x) for x in row] for row in V_frac]
                                W_out = [[float(x) for x in row] for row in W_frac]
                                return {"U": U_out, "V": V_out, "W": W_out}
        
    # Fallback to schoolbook
    return schoolbook()
