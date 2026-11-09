import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # Build target tensor
    # T[u, v, w] = 1 if u=3i+j, v=3j+k, w=3k+i for some i,j,k
    # Else 0
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def unpack(x):
        # x is flat array of size 3 * 9 * 23
        size = n * m * rank
        U = x[:size].reshape(n*m, rank)
        V = x[size:2*size].reshape(m*p, rank)
        W = x[2*size:3*size].reshape(p*n, rank)
        return U, V, W

    def compute_tensor(U, V, W):
        # Returns T[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def loss_and_grad(x):
        U, V, W = unpack(x)
        T = compute_tensor(U, V, W)
        diff = T - T_target
        loss = np.sum(diff**2)
        
        # Gradients
        # dL/dU[u,r] = 2 * sum_{v,w} diff[u,v,w] * V[v,r] * W[w,r]
        # We can use einsum for this too
        # grad_U[u,r] = 2 * einsum('uvw,vr,wr->ur', diff, V, W)
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def verify_solution(U, V, W):
        # Check if entries are close to half-integers
        # And if the tensor matches exactly
        T = compute_tensor(U, V, W)
        # Check exact match with T_target
        # Since we round to 0.5, we should check integer/half-integer nature
        # and exact equality
        if not np.allclose(T, T_target, atol=1e-9):
            return False
        
        # Check if U, V, W are close to half-integers
        # The problem says "integer/half-integer entries"
        # So 2*U should be integer
        if not np.allclose(2*U, np.round(2*U), atol=1e-6):
            return False
        if not np.allclose(2*V, np.round(2*V), atol=1e-6):
            return False
        if not np.allclose(2*W, np.round(2*W), atol=1e-6):
            return False
            
        # Snap to half-integers
        U = np.round(2*U) / 2.0
        V = np.round(2*V) / 2.0
        W = np.round(2*W) / 2.0
        
        # Final exact check
        T_exact = compute_tensor(U, V, W)
        if np.allclose(T_exact, T_target, atol=1e-9):
            return True
        return False

    # Optimization
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    best_result = None
    best_loss = float('inf')
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale to be reasonable
        x0 = np.random.uniform(-1, 1, size=3 * 9 * rank)
        
        # Optimize
        try:
            res = opt.minimize(
                loss_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12}
            )
            
            x_opt = res.x
            U, V, W = unpack(x_opt)
            
            # Round to half-integers
            U_r = np.round(2*U) / 2.0
            V_r = np.round(2*V) / 2.0
            W_r = np.round(2*W) / 2.0
            
            # Verify
            if verify_solution(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # If we found a valid rank 23 solution, we can return it immediately
                # as it's the minimum possible rank for 3x3
                return best_result
                
        except Exception:
            pass

    # Fallback to schoolbook rank 27
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
