import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check ones
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zeros
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    if np.linalg.norm(T_recon - T_target) > tol:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank
    rank_sb = n * m * p
    
    # Prepare schoolbook fallback
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def compute_error(U, V, W, r):
        """Compute squared error of the decomposition."""
        T_recon = np.zeros_like(T)
        for rr in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        return np.sum((T - T_recon) ** 2)

    def try_rank(r, time_budget):
        """Try to find a rank-r decomposition using gradient descent with multiple restarts."""
        max_restarts = 30
        lr = 0.01
        momentum = 0.9
        
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(np_, r) * 0.5
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(np_, r) * 0.3
                # Zero out some entries
                for _ in range(r // 2):
                    U[np.random.randint(nm), np.random.randint(r)] = 0
                    V[np.random.randint(mp), np.random.randint(r)] = 0
                    W[np.random.randint(np_), np.random.randint(r)] = 0
            else:
                # Structured initialization based on schoolbook
                U = np.random.randn(nm, r) * 0.2
                V = np.random.randn(mp, r) * 0.2
                W = np.random.randn(np_, r) * 0.2
            
            # Gradient descent with momentum
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_err = compute_error(U, V, W, r)
            patience = 0
            max_patience = 50
            
            for it in range(2000):
                if time.time() > time_budget:
                    break
                
                # Compute gradients
                # d/dU[a,r] = sum_{b,c} (T[a,b,c] - recon[a,b,c]) * V[b,r] * W[c,r]
                # Similarly for V and W
                
                T_recon = np.zeros_like(T)
                for rr in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                
                residual = T - T_recon
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for rr in range(r):
                    # dU[:,rr] = sum over b,c of residual[:,b,c] * V[b,rr] * W[c,rr]
                    # = einsum('abc,bc->a', residual, V[:,rr]*W[:,rr])... no
                    # dU[a,rr] = sum_{b,c} residual[a,b,c] * V[b,rr] * W[c,rr]
                    # This is: for fixed rr, dU[:,rr] = (residual * (V[:,rr] o W[:,rr])).sum(axis=(1,2))
                    # = einsum('abc,b,c->a', residual, V[:,rr], W[:,rr])
                    dU[:, rr] = np.einsum('abc,b,c->a', residual, V[:, rr], W[:, rr])
                    dV[:, rr] = np.einsum('abc,a,c->b', residual, U[:, rr], W[:, rr])
                    dW[:, rr] = np.einsum('abc,a,b->c', residual, U[:, rr], V[:, rr])
                
                # Update with momentum
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                err = compute_error(U, V, W, r)
                
                if err < 1e-12:
                    return U.copy(), V.copy(), W.copy()
                
                if err < prev_err * (1 - 1e-6):
                    prev_err = err
                    patience = 0
                else:
                    patience += 1
                
                if patience > max_patience:
                    break
                
                # Adaptive learning rate
                if patience > 20:
                    lr *= 0.5
                
                if time.time() > time_budget:
                    break
        
        return None

    # Try ranks from low to high
    min_rank = max(nm, mp, np_)
    
    best_result = None
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 10:
            break
        
        time_budget = min(deadline - 5, time.time() + 25)
        result = try_rank(r, time_budget)
        
        if result is not None:
            U, V, W = result
            # Snap to half-integers if possible
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
            elif _check_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W)
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
