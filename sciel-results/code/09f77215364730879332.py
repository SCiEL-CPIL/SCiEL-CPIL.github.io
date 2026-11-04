import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 6
    rank = 41
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # ALS iterations
    for it in range(50):
        # Update U: for each a, U[a, :] = argmin ||T[a,:,:] - sum_r U[a,r] * V[:,r] * W[:,r]^T||
        # T[a,:,:] is (m*p, p*n)
        # For each r, the contribution is U[a,r] * outer(V[:,r], W[:,r])
        # So T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # This is linear in U[a,:]: T[a,:,:] = M_a @ U[a,:] where M_a[r, (b,c)] = V[b,r]*W[c,r]
        for a in range(n*m):
            # Build M_a of shape (rank, m*p*p*n)
            M_a = np.zeros((rank, m*p * p*n))
            for r in range(rank):
                M_a[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            # Solve M_a @ U[a,:] = T[a,:,:].flatten()
            T_a = T[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(M_a, T_a, rcond=None)[0]
        
        # Update V similarly
        for b in range(m*p):
            M_b = np.zeros((rank, n*m * p*n))
            for r in range(rank):
                M_b[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            T_b = T[:, b, :].flatten()
            V[b, :] = np.linalg.lstsq(M_b, T_b, rcond=None)[0]
        
        # Update W similarly
        for c in range(p*n):
            M_c = np.zeros((rank, n*m * m*p))
            for r in range(rank):
                M_c[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            T_c = T[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(M_c, T_c, rcond=None)[0]
    
    # Check residual
    R = np.zeros_like(T)
    for r in range(rank):
        R += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    residual = np.linalg.norm(R - T)
    
    # Round to nearest half-integer if close
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    R = np.zeros_like(T)
    for r in range(rank):
        R += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    residual = np.linalg.norm(R - T)
    
    if residual < 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 48
        rank48 = n * m * p
        U48 = np.zeros((n*m, rank48))
        V48 = np.zeros((m*p, rank48))
        W48 = np.zeros((p*n, rank48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i*m+j, col] = 1
                    V48[j*p+k, col] = 1
                    W48[k*n+i, col] = 1
                    col += 1
        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
