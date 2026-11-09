"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def compute_loss(U, V, W):
        """Compute the squared loss of the decomposition."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.sum((T - 1.0)**2)
    
    def try_rank(target_rank, num_trials=30, max_iter=80):
        """Try to find a decomposition of given rank using fast random + gradient descent."""
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with random values scaled appropriately
            scale = 1.0 / (target_rank ** 0.5)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Also try structured init every few trials
            if trial % 5 == 0:
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                U += np.random.randn(nm, target_rank) * 0.05
                V += np.random.randn(mp, target_rank) * 0.05
                W += np.random.randn(pn, target_rank) * 0.05
            
            # Fast gradient descent with momentum
            momentum_U = np.zeros_like(U)
            momentum_V = np.zeros_like(V)
            momentum_W = np.zeros_like(W)
            lr = 0.5
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                loss = np.sum((T - 1.0)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    return (U.copy(), V.copy(), W.copy())
                
                diff = 2.0 * (T - 1.0)
                
                grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r).reshape(nm, target_rank)
                grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r).reshape(mp, target_rank)
                grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r).reshape(pn, target_rank)
                
                # Momentum update
                momentum_U = 0.9 * momentum_U + 0.1 * grad_U
                momentum_V = 0.9 * momentum_V + 0.1 * grad_V
                momentum_W = 0.9 * momentum_W + 0.1 * grad_W
                
                U -= lr * momentum_U
                V -= lr * momentum_V
                W -= lr * momentum_W
                
                # Decay learning rate
                lr *= 0.995
            
            # Check best state
            if check_decomposition(best_U, best_V, best_W):
                return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from high to low
    for target_rank in [50, 45, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, num_trials=20, max_iter=60)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Try to go lower if time permits
                if time.time() < deadline - 30:
                    continue
                else:
                    break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W):
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
