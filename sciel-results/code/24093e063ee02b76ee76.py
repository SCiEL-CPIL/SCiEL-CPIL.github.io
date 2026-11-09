import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target tensor T
    # T is a 9x9x9 tensor, but sparse.
    # Indices: u = i*m + j, v = j*p + k, w = k*n + i
    # Non-zeros at (u, v, w) = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    def try_find_decomposition(rank, max_iters=200, restarts=10):
        for _ in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize U, V, W
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Simple Gradient Descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor estimate
                # Est[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
                # This is slow if done naively.
                # Use einsum: 'ur,vr,wr->uvw'
                Est = np.einsum('ur,vr,wr->uvw', U, V, W)
                
                # Error
                Err = Est - T
                
                # Gradients
                # dErr/dU[u,r] = sum_{v,w} Err[u,v,w] V[v,r] W[w,r]
                # dErr/dV[v,r] = sum_{u,w} Err[u,v,w] U[u,r] W[w,r]
                # dErr/dW[w,r] = sum_{u,v} Err[u,v,w] U[u,r] V[v,r]
                
                # Compute gradients efficiently
                # G_U[u,r] = einsum('uvw,vr,wr->ur', Err, V, W)
                # G_V[v,r] = einsum('uvw,ur,wr->vr', Err, U, W)
                # G_W[w,r] = einsum('uvw,ur,vr->wr', Err, U, V)
                
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
                
                # Update
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                # Check convergence
                if np.linalg.norm(Err) < 1e-6:
                    break
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            Est_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(Est_r, T, atol=1e-9):
                return U_r, V_r, W_r
                
        return None

    # Try rank 23
    result = try_find_decomposition(23)
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to rank 27
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, column] = 1
                V[v, column] = 1
                W[w, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
