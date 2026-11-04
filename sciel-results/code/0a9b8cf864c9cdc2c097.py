import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit for search, leaving 10s margin
    
    # Trivial solution
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
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial
    
    # Target ranks to try
    # Lower bound is roughly max(n*p, m*p, n*m) * something?
    # For 3x5x6, trivial is 90.
    # Let's try a range of ranks.
    # Known results for small matrices often have rank close to n*m + m*p + n*p - something?
    # Or just search from a reasonable low bound.
    
    # We will use scipy.optimize if available, otherwise fast numpy gradient descent.
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False

    # Define the loss function for scipy or manual GD
    def compute_loss_and_grads(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        
        # Compute tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        
        # Target is all ones
        target = np.ones((n, m, p))
        error = T - target
        loss = np.sum(error**2)
        
        # Gradients
        # dL/dU[i,j,r] = sum_{k} error[i,j,k] * V[j,k,r] * W[k,i,r]
        # 'ijk,jkr,kir->ijr'
        dU = np.einsum('ijk,jkr,kir->ijr', error, Vr, Wr).reshape(n*m, -1)
        # dL/dV[j,k,r] = sum_{i} error[i,j,k] * U[i,j,r] * W[k,i,r]
        # 'ijk,ijr,kir->jkr'
        dV = np.einsum('ijk,ijr,kir->jkr', error, Ur, Wr).reshape(m*p, -1)
        # dL/dW[k,i,r] = sum_{j} error[i,j,k] * U[i,j,r] * V[j,k,r]
        # 'ijk,ijr,jkr->kir'
        dW = np.einsum('ijk,ijr,jkr->kir', error, Ur, Vr).reshape(p*n, -1)
        
        return loss, dU, dV, dW

    def try_rank(rank):
        if time.time() > deadline:
            return None
            
        # Random initialization
        # Use small random values
        scale = 0.1
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        if use_scipy:
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                Ur = U_flat.reshape(n, m, -1)
                Vr = V_flat.reshape(m, p, -1)
                Wr = W_flat.reshape(p, n, -1)
                
                T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                target = np.ones((n, m, p))
                error = T - target
                loss = np.sum(error**2)
                
                # Gradients
                dU = np.einsum('ijk,jkr,kir->ijr', error, Vr, Wr).reshape(-1)
                dV = np.einsum('ijk,ijr,kir->jkr', error, Ur, Wr).reshape(-1)
                dW = np.einsum('ijk,ijr,jkr->kir', error, Ur, Vr).reshape(-1)
                
                grad = np.concatenate([dU, dV, dW])
                return loss, grad

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
            
            if res.success or res.fun < 1e-4:
                U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                Ur = U_snap.reshape(n, m, -1)
                Vr = V_snap.reshape(m, p, -1)
                Wr = W_snap.reshape(p, n, -1)
                T_check = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                
                if np.allclose(T_check, np.ones((n, m, p)), atol=1e-9):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        else:
            # Fallback to manual GD if scipy not available (should be available)
            pass
            
        return None

    # Search ranks
    # Start from a lower bound estimate
    # For matrix multiplication, rank is at least max(n, m, p) * something?
    # Actually, just try a range.
    # 3x5x6: Trivial 90.
    # Let's try 40 to 60.
    for rank in range(40, 70):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            res = try_rank(rank)
            if res is not None:
                U_list, V_list, W_list = res
                if rank < best_rank:
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = rank
                    found = True
                break # Found a valid decomposition for this rank, move to next lower rank? 
                # Wait, we are iterating 40..70. If we find one at 40, that's good.
                # We should iterate from low to high to find the *minimal* rank.
        
    # If we didn't find anything better than trivial, return trivial.
    # The loop above searches 40 to 70. If it finds one at 40, best_rank becomes 40.
    # If it finds one at 45, best_rank becomes 45 (if 40 failed).
    # So this logic works to find the minimal rank in the range.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
