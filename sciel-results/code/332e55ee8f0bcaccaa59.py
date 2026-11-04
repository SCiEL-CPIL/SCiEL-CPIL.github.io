import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    U_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_target[i*m + j, j*p + k, k*n + i] = 1.0

    # Try to find rank 23 decomposition
    rank = 23
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # We will try a few random restarts
    best_solution = None
    
    # Initialize with schoolbook minus one term, then optimize
    # Schoolbook has 24 terms.
    # Let's pick a random term to "remove" and see if the others can adjust.
    # Actually, better to just do random initialization for 23 terms.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to match magnitude
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        # Use scipy optimize
        try:
            from scipy.optimize import minimize
            # Flatten parameters
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def error(x):
                U_ = x[:n*m*rank].reshape(n*m, rank)
                V_ = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_ = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Reconstruct tensor
                # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Efficient computation:
                # This is a tensor contraction.
                # We can do:
                # Temp[i, j, r] = U[i,r] * V[j,r] ? No.
                # T_recon = np.einsum('ir,jr,kr->ijk', U_, V_, W_)
                T_recon = np.einsum('ir,jr,kr->ijk', U_, V_, W_, optimize=True)
                err = T_recon - U_target
                return np.sum(err**2)
            
            # This might be too slow for 50 restarts.
            # Let's reduce restarts and use a better method.
            # Or use a specific known solution structure.
            
        except ImportError:
            pass
        except Exception:
            continue

    # If we didn't find it with random search, try a more structured approach or fallback.
    # Given the difficulty, let's try a hardcoded solution if we can find one.
    # Since I don't have the hardcoded solution memorized, I will rely on the search.
    # However, random search for exact rank 23 is very unlikely to succeed quickly with simple gradients.
    
    # Let's try a different strategy:
    # The rank of M_2,3,4 is 23.
    # Is there a simple algebraic manipulation?
    # C = AB.
    # We can compute C using 23 multiplications.
    # This is equivalent to finding a bilinear algorithm with 23 multiplications.
    
    # Let's try to use a library that might have it? No.
    
    # Fallback: Return Schoolbook (Rank 24)
    # But we want to beat 1.0.
    
    # Let's try one more specific initialization:
    # Take the 24 schoolbook terms.
    # Combine two of them?
    # No, the terms are independent.
    
    # Let's try a random search with a time limit of 10s, and if it fails, return 24.
    # The probability of finding it is low, but it's the only way without hardcoding.
    
    # Wait, I can use the fact that the entries are half-integers.
    # This suggests a solution with small coefficients.
    
    # Let's try to solve the system of equations for rank 23 using a numerical solver with a tight time limit.
    
    def try_find_rank_23():
        import scipy.optimize as so
        n, m, p = 2, 3, 4
        rank = 23
        N = n*m
        M = m*p
        P = p*n
        
        T = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Random init
        U = np.random.randn(N, rank)
        V = np.random.randn(M, rank)
        W = np.random.randn(P, rank)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_ = x[:N*rank].reshape(N, rank)
            V_ = x[N*rank : N*rank + M*rank].reshape(M, rank)
            W_ = x[N*rank + M*rank :].reshape(P, rank)
            T_recon = np.einsum('ir,jr,kr->ijk', U_, V_, W_, optimize=True)
            return np.sum((T_recon - T)**2)
        
        # Use L-BFGS-B
        # This is likely to fail to converge to 0 in short time.
        # But let's try.
        
        start = time.time()
        try:
            res = so.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15})
        except:
            return None
        
        if time.time() - start > 10:
            return None
            
        if res.fun < 1e-10:
            U_ = res.x[:N*rank].reshape(N, rank)
            V_ = res.x[N*rank : N*rank + M*rank].reshape(M, rank)
            W_ = res.x[N*rank + M*rank :].reshape(P, rank)
            
            # Snap to half-integers
            U_int = np.round(U_ * 2) / 2
            V_int = np.round(V_ * 2) / 2
            W_int = np.round(W_ * 2) / 2
            
            # Verify
            T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int, optimize=True)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_int, V_int, W_int
        return None

    # Try a few times
    for _ in range(5):
        if time.time() > 120 - 5:
            break
        sol = try_find_rank_23()
        if sol:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
