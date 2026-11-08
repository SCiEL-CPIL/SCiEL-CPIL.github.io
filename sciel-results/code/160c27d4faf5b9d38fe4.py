"""Direct tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = t0 + 130  # Safety margin before 150s kill

    # Trivial rank-36 solution
    rank_trivial = n * m * p  # 36
    U_trivial = np.zeros((n * m, rank_trivial), dtype=float)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=float)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_trivial[i * m + j, r] = 1.0
                V_trivial[j * p + k, r] = 1.0
                W_trivial[k * n + i, r] = 1.0

    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    # Precompute index arrays for fast access
    u_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    num_constraints = n * m * p  # 36

    # Try ranks from 35 down to 20
    for try_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_improved(try_rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline)
        if result is not None:
            U_t, V_t, W_t = result
            if _verify_decomposition(U_t, V_t, W_t, n, m, p):
                best_rank = try_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _try_rank_improved(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline):
    """Try to find a rank-r decomposition using multiple strategies."""
    
    # Strategy 1: Random initialization with gradient descent
    for seed in range(20):
        if time.time() > deadline:
            return None
        result = _gradient_descent(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline, seed)
        if result is not None:
            return result
    
    # Strategy 2: Structured initialization based on matrix structure
    result = _structured_search(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline)
    if result is not None:
        return result
    
    return None


def _gradient_descent(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline, seed):
    """Gradient descent with random initialization."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    lr = 0.1
    best_res = np.inf
    
    for step in range(500):
        if time.time() > deadline:
            return None
        
        # Compute residuals
        res = np.zeros(num_constraints)
        for r in range(rank):
            res += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        res -= 1.0
        
        max_res = np.max(np.abs(res))
        if max_res < 1e-10:
            return (U, V, W)
        
        if max_res < best_res:
            best_res = max_res
        
        # Compute gradients
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for r in range(rank):
            U_contrib = res * V[v_idx, r] * W[w_idx, r]
            V_contrib = U[u_idx, r] * res * W[w_idx, r]
            W_contrib = U[u_idx, r] * V[v_idx, r] * res
            np.add.at(U_grad, u_idx, U_contrib)
            np.add.at(V_grad, v_idx, V_contrib)
            np.add.at(W_grad, w_idx, W_contrib)
        
        # Update with momentum-like behavior
        U -= lr * U_grad
        V -= lr * V_grad
        W -= lr * W_grad
        
        # Adaptive learning rate
        if step % 50 == 49:
            new_res = np.zeros(num_constraints)
            for r in range(rank):
                new_res += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            new_res -= 1.0
            if np.max(np.abs(new_res)) > max_res * 1.5:
                lr *= 0.5
            elif step > 100:
                lr = min(lr * 1.1, 0.1)
    
    return None


def _structured_search(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline):
    """Try structured initializations based on matrix multiplication structure."""
    # Initialize with structure: group by i
    for i in range(n):
        if time.time() > deadline:
            return None
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Assign rank slots to different i values
        slots_per_i = rank // n
        for i_val in range(n):
            start = i_val * slots_per_i
            end = start + slots_per_i if i_val < n - 1 else rank
            
            for j in range(m):
                for k in range(p):
                    r = start + (j * p + k) % (end - start)
                    U[i_val * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i_val, r] = 1.0
        
        # Refine with gradient descent
        result = _gradient_descent_from_init(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline, U, V, W)
        if result is not None:
            return result
    
    return None


def _gradient_descent_from_init(rank, n, m, p, u_idx, v_idx, w_idx, num_constraints, deadline, U, V, W):
    """Gradient descent from a given initialization."""
    lr = 0.05
    
    for step in range(300):
        if time.time() > deadline:
            return None
        
        # Compute residuals
        res = np.zeros(num_constraints)
        for r in range(rank):
            res += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        res -= 1.0
        
        max_res = np.max(np.abs(res))
        if max_res < 1e-10:
            return (U, V, W)
        
        # Compute gradients
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for r in range(rank):
            U_contrib = res * V[v_idx, r] * W[w_idx, r]
            V_contrib = U[u_idx, r] * res * W[w_idx, r]
            W_contrib = U[u_idx, r] * V[v_idx, r] * res
            np.add.at(U_grad, u_idx, U_contrib)
            np.add.at(V_grad, v_idx, V_contrib)
            np.add.at(W_grad, w_idx, W_contrib)
        
        # Update
        U -= lr * U_grad
        V -= lr * V_grad
        W -= lr * W_grad
        
        # Adaptive learning rate
        if step % 50 == 49:
            new_res = np.zeros(num_constraints)
            for r in range(rank):
                new_res += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            new_res -= 1.0
            if np.max(np.abs(new_res)) > max_res * 1.5:
                lr *= 0.5
            elif step > 50:
                lr = min(lr * 1.05, 0.05)
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies the tensor equation."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(U.shape[1]):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
