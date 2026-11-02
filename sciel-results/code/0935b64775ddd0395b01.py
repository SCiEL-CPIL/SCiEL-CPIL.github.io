import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = nm * mp * pn // (n * m * p) # No, schoolbook rank is n*m*p
    schoolbook_rank = n * m * p
    
    # Target tensor T of shape (nm, mp, pn)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Try to find a low rank decomposition
    # Start with a guess rank
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 60 to 72
    # We know 72 is always possible (schoolbook)
    # We want to find < 72
    
    # Heuristic: Random initialization + Gradient Descent
    # Objective: sum (sum_c U[u,c] V[v,c] W[w,c] - T[u,v,w])^2
    
    for rank in range(60, 73):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                # Compute current tensor estimate
                # This is expensive: O(nm * mp * pn * rank)
                # 9 * 24 * 24 * 60 ~ 300k operations per step.
                # 500 steps * 5 restarts * 13 ranks ~ 3.9M * 300k = too slow?
                # 300k is small. 3.9M steps is the issue.
                # We need to be faster.
                
                # Use einsum for speed
                # T_est[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
                T_est = np.einsum('uc,vc,wc->uvw', U, V, W)
                error = T_est - T
                loss = np.sum(error**2)
                
                if loss < 1e-6:
                    # Found a solution
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
                    
                # Compute gradients
                # dL/dU[u,c] = 2 * sum_{v,w} error[u,v,w] V[v,c] W[w,c]
                grad_U = 2 * np.einsum('uvw,vc,wc->uc', error, V, W)
                grad_V = 2 * np.einsum('uvw,uc,wc->vc', error, U, W)
                grad_W = 2 * np.einsum('uvw,uc,vc->wc', error, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if best_rank == rank:
                break
                
        if best_rank == rank:
            break
            
    if best_U is not None:
        # Convert to lists of lists with integer/half-integer entries if possible
        # The problem allows integer/half-integer.
        # We can round to nearest 0.5
        U_out = [[round(x * 2) / 2 for x in row] for row in best_U]
        V_out = [[round(x * 2) / 2 for x in row] for row in best_V]
        W_out = [[round(x * 2) / 2 for x in row] for row in best_W]
        
        # Verify validity
        # Reconstruct and check
        U_np = np.array(U_out)
        V_np = np.array(V_out)
        W_np = np.array(W_out)
        T_check = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        if np.allclose(T_check, T, atol=1e-5):
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            # Fallback to schoolbook
            pass
            
    # Schoolbook fallback
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
