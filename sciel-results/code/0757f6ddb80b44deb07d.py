import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check if entries are half-integers
    if not np.allclose(U_np * 2, np.round(U_np * 2), atol=1e-9):
        return False
    if not np.allclose(V_np * 2, np.round(V_np * 2), atol=1e-9):
        return False
    if not np.allclose(W_np * 2, np.round(W_np * 2), atol=1e-9):
        return False
        
    # Check tensor reconstruction
    T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
        
    return True

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    return T_target

def _structured_init(n, m, p, target_rank, strategy=0):
    """Generate structured initializations that partially satisfy constraints."""
    nm, mp, pn = n * m, m * p, p * n
    
    if strategy == 0:
        # Random small
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
    elif strategy == 1:
        # Try to use schoolbook-like structure with noise
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Assign random subsets to each rank component
        for c in range(target_rank):
            # Pick random (i,j,k) pattern
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            # Add to multiple positions to create structure
            for i2 in range(n):
                for j2 in range(m):
                    if random.random() < 0.3:
                        U[i2*m+j2, c] = random.choice([0.5, -0.5, 1.0, -1.0])
            for j2 in range(m):
                for k2 in range(p):
                    if random.random() < 0.3:
                        V[j2*p+k2, c] = random.choice([0.5, -0.5, 1.0, -1.0])
            for k2 in range(p):
                for i2 in range(n):
                    if random.random() < 0.3:
                        W[k2*n+i2, c] = random.choice([0.5, -0.5, 1.0, -1.0])
                        
        # Add small noise
        U += np.random.randn(nm, target_rank) * 0.05
        V += np.random.randn(mp, target_rank) * 0.05
        W += np.random.randn(pn, target_rank) * 0.05
    elif strategy == 2:
        # Block diagonal-like structure
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Divide rank into blocks
        block_size = max(1, target_rank // 3)
        for b in range(3):
            start = b * block_size
            end = min(start + block_size, target_rank)
            
            i = b % n
            j = b % m
            k = b % p
            
            for c in range(start, end):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                
                # Add some cross terms
                if random.random() < 0.5:
                    i2 = random.randint(0, n-1)
                    U[i2*m+j, c] = random.choice([0.5, -0.5])
                if random.random() < 0.5:
                    k2 = random.randint(0, p-1)
                    V[j*p+k2, c] = random.choice([0.5, -0.5])
                if random.random() < 0.5:
                    i2 = random.randint(0, n-1)
                    W[k*n+i2, c] = random.choice([0.5, -0.5])
            
        U += np.random.randn(nm, target_rank) * 0.05
        V += np.random.randn(mp, target_rank) * 0.05
        W += np.random.randn(pn, target_rank) * 0.05
    else:
        # Sparse random
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for c in range(target_rank):
            # Pick a few non-zero entries
            for _ in range(random.randint(2, 5)):
                U[random.randint(0, nm-1), c] = random.choice([0.5, -0.5, 1.0, -1.0])
            for _ in range(random.randint(2, 5)):
                V[random.randint(0, mp-1), c] = random.choice([0.5, -0.5, 1.0, -1.0])
            for _ in range(random.randint(2, 5)):
                W[random.randint(0, pn-1), c] = random.choice([0.5, -0.5, 1.0, -1.0])
        
        U += np.random.randn(nm, target_rank) * 0.02
        V += np.random.randn(mp, target_rank) * 0.02
        W += np.random.randn(pn, target_rank) * 0.02
    
    return U, V, W

def _polish_solution(U, V, W, T_target, n, m, p, max_iters=50):
    """Try to polish a near-solution to exact half-integer values."""
    nm, mp, pn = n * m, m * p, p * n
    target_rank = U.shape[1]
    
    # First, try rounding and verify
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try local search around the rounded solution
    best_err = np.inf
    best_solution = None
    
    for _ in range(max_iters):
        # Random perturbation in half-integer space
        U_try = U_r + np.random.randint(-1, 2, size=(nm, target_rank)) * 0.5
        V_try = V_r + np.random.randint(-1, 2, size=(mp, target_rank)) * 0.5
        W_try = W_r + np.random.randint(-1, 2, size=(pn, target_rank)) * 0.5
        
        if _verify(U_try.tolist(), V_try.tolist(), W_r.tolist(), n, m, p):
            return U_try.tolist(), V_try.tolist(), W_try.tolist()
        
        # Compute error
        T_recon = np.einsum('uc,vc,wc->uvw', U_try, V_try, W_try)
        err = np.linalg.norm(T_recon - T_target)
        
        if err < best_err:
            best_err = err
            best_solution = (U_try.copy(), V_try.copy(), W_try.copy())
    
    if best_solution is not None:
        U_r, V_r, W_r = best_solution
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """Try to find a decomposition of target_rank using improved Gradient Descent with momentum."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T_target = _build_target_tensor(n, m, p)

    # Try multiple initialization strategies and restarts
    strategies = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    
    for attempt in range(80):
        if time.time() > time_limit:
            break
            
        strategy = strategies[attempt % len(strategies)]
        U, V, W = _structured_init(n, m, p, target_rank, strategy)
        
        # Momentum-based gradient descent
        lr = 0.02
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        beta = 0.9
        
        for it in range(2000):
            if time.time() > time_limit:
                break
                
            # Compute T_recon
            T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
            E = T_recon - T_target
            
            # Gradients
            grad_U = np.einsum('uvw,vc,wc->uc', E, V, W)
            grad_V = np.einsum('uvw,uc,wc->vc', E, U, W)
            grad_W = np.einsum('uvw,uc,vc->wc', E, U, V)
            
            # Update momentum
            momentum_U = beta * momentum_U + (1 - beta) * grad_U
            momentum_V = beta * momentum_V + (1 - beta) * grad_V
            momentum_W = beta * momentum_W + (1 - beta) * grad_W
            
            # Update with momentum
            U -= lr * momentum_U
            V -= lr * momentum_V
            W -= lr * momentum_W
            
            # Check for solution periodically
            if it % 10 == 0:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
                # If error is very small, try to polish
                err = np.linalg.norm(E)
                if err < 0.1:
                    polished = _polish_solution(U, V, W, T_target, n, m, p, max_iters=30)
                    if polished:
                        return polished
                    break  # Restart with new initialization
        
        # After each attempt, check once more
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        err = np.linalg.norm(np.einsum('uc,vc,wc->uvw', U, V, W) - T_target)
        if err < 0.1:
            polished = _polish_solution(U, V, W, T_target, n, m, p, max_iters=50)
            if polished:
                return polished

    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    # Try to find a better decomposition
    # Start from rank 27 up to 35
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 8:
            break
        result = _try_find_decomposition(n, m, p, r, deadline - 3)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
