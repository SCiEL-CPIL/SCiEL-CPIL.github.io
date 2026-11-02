"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as lists."""
        rank = n * m * p  # 50
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def loss_and_grad(params, rank):
        """Compute loss and gradient for given parameters and rank."""
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        # Gradients
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad
    
    def try_rank(rank, max_restarts=20):
        """Try to find a valid decomposition with the given rank."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            np.random.seed(12345 + restart)
            u0 = np.random.randn(dim_u * rank) * 0.5
            v0 = np.random.randn(dim_v * rank) * 0.5
            w0 = np.random.randn(dim_w * rank) * 0.5
            
            params = np.concatenate([u0, v0, w0])
            
            def fun(params):
                loss, grad = loss_and_grad(params, rank)
                return loss, grad
            
            try:
                # Use L-BFGS-B with tighter tolerances and more iterations
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                
                if res.fun < 1e-10:
                    # Extract final parameters
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    # Snap to nearest half-integer
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    # Verify the snapped solution
                    T_snap = np.einsum('ar,br,cr->abc', u_snap, v_snap, w_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        return (u_snap, v_snap, w_snap)
                    
                    # Try without snapping if it's already very close
                    if np.allclose(T_snap, target, atol=1e-6):
                        # Check if we can clean up small values
                        u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
                        v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
                        w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
                        T_clean = np.einsum('ar,br,cr->abc', u_clean, v_clean, w_clean)
                        if np.allclose(T_clean, target, atol=1e-9):
                            return (u_clean, v_clean, w_clean)
            except Exception:
                continue
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But for matrix multiplication, better bounds exist
    # Let's try from 25 up to 49
    
    best_result = None
    
    # Try to find a decomposition with rank < 50
    for target_rank in range(25, 50):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
