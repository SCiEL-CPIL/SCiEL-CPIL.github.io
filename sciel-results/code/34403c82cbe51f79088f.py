import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T_target.flatten()
    
    # Try ranks from 16 upward
    for r in [16, 18, 20, 24, 28, 32, 36, 40, 44, 48]:
        for seed in range(10):
            np.random.seed(seed)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Gradient descent
            lr = 0.001
            for it in range(5000):
                # Compute tensor
                T = np.zeros((n*m, m*p, p*n))
                for rr in range(r):
                    T += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                
                diff = T - T_target
                error = np.sum(diff**2)
                if error < 1e-10:
                    break
                
                # Gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for rr in range(r):
                    dU[:,rr] = np.einsum('abc,b,c->a', diff, V[:,rr], W[:,rr])
                    dV[:,rr] = np.einsum('abc,a,c->b', diff, U[:,rr], W[:,rr])
                    dW[:,rr] = np.einsum('abc,a,b->c', diff, U[:,rr], V[:,rr])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check
            T = np.zeros((n*m, m*p, p*n))
            for rr in range(r):
                T += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
            error = np.sum((T - T_target)**2)
            
            if error < 1e-8:
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                T = np.zeros((n*m, m*p, p*n))
                for rr in range(r):
                    T += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                if np.allclose(T, T_target, atol=1e-10):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
