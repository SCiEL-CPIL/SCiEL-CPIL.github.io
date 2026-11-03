import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Construct the target tensor T of shape (num_u, num_v, num_w)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((num_u, num_v, num_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want to find U (num_u x R), V (num_v x R), W (num_w x R)
    # such that T = sum_r U[:, r] outer V[:, r] outer W[:, r]
    
    # Helper to verify a candidate decomposition
    def verify(U, V, W):
        # Check if entries are close to integers or half-integers
        # And if the reconstruction is exact
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snap is close to original
        if not (np.allclose(U, U_snap, atol=1e-6) and 
                np.allclose(V, V_snap, atol=1e-6) and 
                np.allclose(W, W_snap, atol=1e-6)):
            return False
            
        # Reconstruct T
        R = U_snap.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        
        if np.allclose(T, T_recon, atol=1e-6):
            return True
        return False

    # Helper to try a specific rank with random restarts
    def try_rank(R, max_restarts=50):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(num_u, R)
            V = np.random.randn(num_v, R)
            W = np.random.randn(num_w, R)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Simple Gradient Descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute current reconstruction
                T_curr = np.zeros_like(T)
                for r in range(R):
                    T_curr += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Residual
                Res = T - T_curr
                
                # Gradients
                # dL/dU[:, r] = sum_{b,c} Res[a,b,c] V[b,r] W[c,r]
                # dL/dV[:, r] = sum_{a,c} Res[a,b,c] U[a,r] W[c,r]
                # dL/dW[:, r] = sum_{a,b} Res[a,b,c] U[a,r] V[b,r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(R):
                    # dU[:, r]
                    # Res is (num_u, num_v, num_w)
                    # We need sum over b, c of Res[a, b, c] * V[b, r] * W[c, r]
                    # This is (Res * V[r, :] * W[r, :]).sum(axis=(1, 2))
                    # V[r, :] is shape (num_v,)
                    # W[r, :] is shape (num_w,)
                    dU[:, r] = np.einsum('abc,b,c->a', Res, V[:, r], W[:, r])
                    dV[:, r] = np.einsum('abc,a,c->b', Res, U[:, r], W[:, r])
                    dW[:, r] = np.einsum('abc,a,b->c', Res, U[:, r], V[:, r])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodic verification
                if step % 100 == 0:
                    if verify(U, V, W):
                        return (U, V, W)
                        
            if verify(U, V, W):
                return (U, V, W)
                
        return None

    # Try ranks starting from lower bound
    # Lower bound is max(9, 21, 21) = 21
    # Known rank for 3x3x3 is 23.
    # Let's try a range.
    for R in range(21, 64):
        if time.time() > deadline:
            break
            
        result = try_rank(R, max_restarts=10)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists for the return format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook if no better found
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
