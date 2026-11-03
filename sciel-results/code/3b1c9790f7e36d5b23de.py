import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Trivial schoolbook decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T
    
    T = make_tensor(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    def verify(U, V, W, T, tol=1e-8):
        R = U.shape[1]
        try:
            Temp = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', Temp, W)
            return np.allclose(T_approx, T, atol=tol)
        except:
            return False
    
    def try_rank(R, max_starts=20, time_per_start=2.0):
        """Try to find a rank-R decomposition"""
        start_time = time.time()
        
        for s in range(max_starts):
            if time.time() > deadline - 10:
                break
            if time.time() - start_time > time_per_start * max_starts:
                break
            
            # Structured initialization: start from trivial structure with random perturbations
            # For each rank term r, assign it to a random (i,j,k) triple and add noise
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            # Assign each rank term to a random (i,j,k)
            for r in range(R):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Add random perturbations
            noise_scale = 0.1
            U += np.random.normal(0, noise_scale, (nm, R))
            V += np.random.normal(0, noise_scale, (mp, R))
            W += np.random.normal(0, noise_scale, (pn, R))
            
            # Local optimization using gradient descent with adaptive learning rate
            lr = 0.05
            best_err = np.inf
            U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Compute error
                Temp = np.einsum('ur,vr->uvr', U, V)
                T_approx = np.einsum('uvr,wr->uvw', Temp, W)
                err = np.sum((T_approx - T)**2)
                
                if err < best_err:
                    best_err = err
                    U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                diff = T_approx - T  # shape (nm, mp, pn)
                
                # Compute gradients
                dW = np.einsum('uvw,ur,vr->wr', diff, U, V)
                dU = np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = np.einsum('uvw,ur,wr->vr', diff, U, W)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if it % 50 == 49 and err > 1e-4:
                    lr *= 0.7
                
                if err < 1e-10:
                    break
            
            # High-precision polish pass
            if best_err < 1e-6:
                U_polish, V_polish, W_polish = U_best.copy(), V_best.copy(), W_best.copy()
                lr_polish = 0.001
                for it in range(1000):
                    if time.time() > deadline - 5:
                        break
                    
                    Temp = np.einsum('ur,vr->uvr', U_polish, V_polish)
                    T_approx = np.einsum('uvr,wr->uvw', Temp, W_polish)
                    err = np.sum((T_approx - T)**2)
                    
                    if err < 1e-14:
                        break
                    
                    diff = T_approx - T
                    dW = np.einsum('uvw,ur,vr->wr', diff, U_polish, V_polish)
                    dU = np.einsum('uvw,vr,wr->ur', diff, V_polish, W_polish)
                    dV = np.einsum('uvw,ur,wr->vr', diff, U_polish, W_polish)
                    
                    U_polish -= lr_polish * dU
                    V_polish -= lr_polish * dV
                    W_polish -= lr_polish * dW
                    
                    if it % 100 == 99 and err > 1e-8:
                        lr_polish *= 0.5
                
                if err < 1e-12:
                    # Try to snap to integers or half-integers
                    for factor in [1, 2, 4, 8]:
                        U_round = np.round(U_polish * factor) / factor
                        V_round = np.round(V_polish * factor) / factor
                        W_round = np.round(W_polish * factor) / factor
                        if verify(U_round, V_round, W_round, T, tol=1e-8):
                            return U_round, V_round, W_round
                    
                    if verify(U_polish, V_polish, W_polish, T, tol=1e-10):
                        return U_polish, V_polish, W_polish
            
            # Check if we found a valid solution
            if best_err < 1e-8:
                for factor in [1, 2, 4]:
                    U_round = np.round(U_best * factor) / factor
                    V_round = np.round(V_best * factor) / factor
                    W_round = np.round(W_best * factor) / factor
                    if verify(U_round, V_round, W_round, T, tol=1e-6):
                        return U_round, V_round, W_round
                
                if verify(U_best, V_best, W_best, T, tol=1e-6):
                    return U_best, V_best, W_best
        
        return None
    
    # Try ranks from low to high
    candidate_ranks = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for R in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        # Use more starts for lower ranks, fewer for higher
        if R <= 20:
            max_starts = 15
            time_per_start = 2.0
        elif R <= 25:
            max_starts = 12
            time_per_start = 1.5
        else:
            max_starts = 8
            time_per_start = 1.0
        
        result = try_rank(R, max_starts=max_starts, time_per_start=time_per_start)
        if result:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: return trivial decomposition
    return {
        "U": U_trivial,
        "V": V_trivial,
        "W": W_trivial
    }
