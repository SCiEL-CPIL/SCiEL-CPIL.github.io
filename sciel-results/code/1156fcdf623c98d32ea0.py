"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_school
    
    # Try to find lower rank via numerical optimization
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 53 down
    for rank in range(rank_school - 1, 30, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + trial)
            U_r = np.random.randn(n*m, rank)
            V_r = np.random.randn(m*p, rank)
            W_r = np.random.randn(p*n, rank)
            
            # Simple gradient descent (very limited)
            lr = 0.01
            for step in range(20):
                if time.time() > deadline - 5:
                    break
                D = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                err = D - T
                # Gradient w.r.t. U_r[a,r] = sum_{b,c} err[a,b,c] * V_r[b,r] * W_r[c,r]
                grad_U = np.einsum('abc,br,cr->ar', err, V_r, W_r, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', err, U_r, W_r, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', err, U_r, V_r, optimize=True)
                
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                if np.linalg.norm(err) < 1e-8:
                    break
            
            # Check if we found a valid decomposition
            if np.linalg.norm(D - T) < 1e-8:
                # Try rational reconstruction
                from fractions import Fraction
                U_f = np.zeros_like(U_r, dtype=object)
                V_f = np.zeros_like(V_r, dtype=object)
                W_f = np.zeros_like(W_r, dtype=object)
                ok = True
                for i in range(n*m):
                    for j in range(rank):
                        f = Fraction(U_r[i,j]).limit_denominator(1000)
                        if abs(f - U_r[i,j]) > 1e-6:
                            ok = False
                            break
                        U_f[i,j] = f
                    if not ok:
                        break
                if not ok:
                    continue
                for i in range(m*p):
                    for j in range(rank):
                        f = Fraction(V_r[i,j]).limit_denominator(1000)
                        if abs(f - V_r[i,j]) > 1e-6:
                            ok = False
                            break
                        V_f[i,j] = f
                    if not ok:
                        break
                if not ok:
                    continue
                for i in range(p*n):
                    for j in range(rank):
                        f = Fraction(W_r[i,j]).limit_denominator(1000)
                        if abs(f - W_r[i,j]) > 1e-6:
                            ok = False
                            break
                        W_f[i,j] = f
                    if not ok:
                        break
                if ok:
                    # Verify exactness
                    U_np = np.array([[float(x) for x in row] for row in U_f], dtype=float)
                    V_np = np.array([[float(x) for x in row] for row in V_f], dtype=float)
                    W_np = np.array([[float(x) for x in row] for row in W_f], dtype=float)
                    D_exact = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
                    if np.allclose(D_exact, T, atol=1e-10):
                        best_rank = rank
                        best_result = {
                            "U": [[float(x) for x in row] for row in U_f],
                            "V": [[float(x) for x in row] for row in V_f],
                            "W": [[float(x) for x in row] for row in W_f]
                        }
                        break  # Found valid, move to next rank
        
        if time.time() > deadline - 5:
            break
    
    return best_result

# EVOLVE-BLOCK-END
