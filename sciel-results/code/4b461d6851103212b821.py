"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = _tensor_from_matrices(U, V, W)
    return np.allclose(result, T, atol=tol)

def _try_rank16_gradient_descent(start_time, deadline):
    """Try to find a rank-16 decomposition using gradient descent."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Get the nonzero positions of T
    nz = np.argwhere(T != 0)
    target_vals = T[nz]
    
    # We want to minimize ||tensor_from_matrices(U,V,W) - T||^2
    # Use Adam optimizer
    
    rng = np.random.default_rng(123)
    
    best_loss = float('inf')
    best_params = None
    
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random small values
        U = rng.normal(0, 0.5, (n_a, rank))
        V = rng.normal(0, 0.5, (n_b, rank))
        W = rng.normal(0, 0.5, (n_c, rank))
        
        # Adam optimizer
        lr = 0.01
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        t = 0
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            t += 1
            
            # Compute current tensor
            T_cur = _tensor_from_matrices(U, V, W)
            
            # Compute loss
            loss = np.sum((T_cur - T)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Compute gradients
            # d/dU[a,r] = sum_{b,c} (T_cur[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            dT = T_cur - T
            dU = np.einsum('abc,bc->ar', dT, V[:, :rank] * W[:, :rank])  # This is wrong, let me fix
            # Actually: dU[a,r] = sum_{b,c} dT[a,b,c] * V[b,r] * W[c,r]
            dU = np.zeros((n_a, rank))
            dV = np.zeros((n_b, rank))
            dW = np.zeros((n_c, rank))
            for r in range(rank):
                dU[:, r] = np.einsum('abc,b,c->a', dT, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,a,c->b', dT, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,a,b->c', dT, U[:, r], V[:, r])
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        if best_loss < 1e-9:
            U_best, V_best, W_best = best_params
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    return None

def _rank17_solution():
    """Return the known rank-17 solution."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Try to find rank-16
    result_16 = _try_rank16_gradient_descent(start_time, deadline)
    if result_16 is not None:
        # Verify
        U = np.array(result_16['U'])
        V = np.array(result_16['V'])
        W = np.array(result_16['W'])
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return result_16
    
    # Fall back to rank-17 solution
    U, V, W = _rank17_solution()
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
