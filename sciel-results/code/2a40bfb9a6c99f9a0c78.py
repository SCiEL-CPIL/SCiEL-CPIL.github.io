import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140

    # Target: Find rank 16 decomposition of 2x3x3 multiplication tensor
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    def find_decomposition(r, max_time=60):
        t0 = time.time()
        rng = np.random.RandomState(42)
        
        # Initialize U, V, W
        U = rng.randn(n2*m2, r)
        V = rng.randn(m2*p2, r)
        W = rng.randn(p2*n2, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient Descent
        lr = 0.01
        for it in range(50000):
            if time.time() - t0 > max_time:
                break
            
            # Compute gradients
            # err = sum (T - sum_r U[a,r]V[b,r]W[c,r])^2
            # grad_U[a,r] = -2 * sum_{b,c} (T[a,b,c] - val[a,b,c]) * V[b,r] * W[c,r]
            # We can vectorize this.
            
            # Compute val tensor
            val = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T2 - val
            
            # Gradients
            grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = np.sum(diff**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    return best_U, best_V, best_W
            
            # Adjust learning rate if err increases significantly
            if err > best_err * 1.1:
                lr *= 0.5
            else:
                lr *= 1.01
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None

    best_2x3x3 = None
    # Try rank 16
    res = find_decomposition(16, max_time=30)
    if res is not None:
        best_2x3x3 = (16, res)
    else:
        # Try rank 17 as fallback? Or just use schoolbook
        res = find_decomposition(17, max_time=30)
        if res is not None:
            best_2x3x3 = (17, res)

    if best_2x3x3 is not None:
        r, (U1, V1, W1) = best_2x3x3
        total_rank = 2 * r
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Block 1: columns 0-2 of B
        U[:, 0:r] = U1
        V[0:9, 0:r] = V1
        W[0:6, 0:r] = W1
        
        # Block 2: columns 3-5 of B
        U[:, r:2*r] = U1
        V[9:18, r:2*r] = V1
        W[6:12, r:2*r] = W1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
