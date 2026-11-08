import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin
    
    # Build the target tensor
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = n*m*p  # schoolbook
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook as fallback
    U0, V0, W0 = schoolbook()
    best_result = (U0, V0, W0)
    best_rank = n*m*p
    
    # Try to find lower rank decompositions
    # Use numerical optimization
    def try_rank(rank, max_iters=500, lr=0.01):
        nonlocal best_result, best_rank
        if rank >= best_rank:
            return False
        
        for attempt in range(3):
            if time.time() > deadline:
                return False
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Use Adam-like optimization
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Compute residual: R[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t] - target[a,b,c]
                # This is expensive to compute fully, so use sampling
                # Sample a subset of entries
                num_samples = 200
                indices = []
                for _ in range(num_samples):
                    a = random.randint(0, n*m-1)
                    b = random.randint(0, m*p-1)
                    c = random.randint(0, p*n-1)
                    indices.append((a,b,c))
                
                loss = 0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for (a,b,c) in indices:
                    # R = sum_t U[a,t] V[b,t] W[c,t]
                    R = np.dot(np.dot(U[a,:], V[b,:]), W[c,:])  # This is sum_t U[a,t]*V[b,t]*W[c,t]
                    # Actually: sum_t U[a,t] * V[b,t] * W[c,t]
                    R = 0
                    for t in range(rank):
                        R += U[a,t] * V[b,t] * W[c,t]
                    err = R - target[a,b,c]
                    loss += err*err
                    
                    # Gradients
                    for t in range(rank):
                        dU[a,t] += 2*err * V[b,t] * W[c,t]
                        dV[b,t] += 2*err * U[a,t] * W[c,t]
                        dW[c,t] += 2*err * U[a,t] * V[b,t]
                
                loss /= num_samples
                
                # Adam update
                for param, mparam, vparam in [(U, mU, vU), (V, mV, vV), (W, mW, vW)]:
                    pass  # Need to handle properly
                
                # Simplified: just use SGD with momentum
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-6:
                    # Try rounding
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    # Check exactness
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for j2 in range(m):
                                for k in range(p):
                                    for k2 in range(p):
                                        for i2 in range(n):
                                            R = 0
                                            for t in range(rank):
                                                R += U_round[i*m+j, t] * V_round[j2*p+k, t] * W_round[k2*n+i2, t]
                                            expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                            err = abs(R - expected)
                                            if err > max_err:
                                                max_err = err
                    if max_err < 1e-10:
                        best_result = (U_round, V_round, W_round)
                        best_rank = rank
                        return True
                    # Also try without rounding if loss is very small
                    if loss < 1e-12:
                        # Check if already close to half-integers
                        U_close = np.all(np.abs(U*2 - np.round(U*2)) < 1e-6)
                        V_close = np.all(np.abs(V*2 - np.round(V*2)) < 1e-6)
                        W_close = np.all(np.abs(W*2 - np.round(W*2)) < 1e-6)
                        if U_close and V_close and W_close:
                            best_result = (U, V, W)
                            best_rank = rank
                            return True
            return False
        
        return False
    
    # Try ranks from lower to upper
    # Lower bound: at least max(n*m, m*p, p*n) = 16
    # But actually the tensor rank lower bound is higher
    # Try from 16 up to best_rank-1
    for rank in range(16, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank):
            break
    
    # Convert best_result to lists
    U, V, W = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
