"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _try_rank16_search(start_time, deadline):
    """Try to find a rank-16 decomposition by gradient descent on the 16 kept terms."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(n, m, p)
    
    terms_17 = [
        # Part 1: Strassen
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        # Part 2: Direct right column
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        # Part 3: Direct bottom row
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms_17):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
            
    def _compute_gradients(U, V, W, T):
        """Compute gradients of L2 loss w.r.t U, V, W."""
        rank = U.shape[1]
        # E = T_approx - T
        # T_approx = sum_r U[:,r] V[:,r] W[:,r]
        # Grad_U[a,r] = 2 * sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        
        # Efficiently compute using einsum
        # T_approx can be computed, but we need E for each r's contribution?
        # No, E is global.
        # Let's compute T_approx first.
        T_approx = np.zeros_like(T)
        for r in range(rank):
            T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        E = T_approx - T
        
        # Grad_U[a,r] = 2 * sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        # This is equivalent to:
        # Temp1[a,b,r] = sum_c E[a,b,c] W[c,r] = einsum('abc,cr->abr', E, W)
        # Grad_U[a,r] = 2 * sum_b Temp1[a,b,r] V[b,r] = 2 * einsum('abr,br->ar', Temp1, V)
        
        Temp1 = np.einsum('abc,cr->abr', E, W)
        Grad_U = 2 * np.einsum('abr,br->ar', Temp1, V)
        
        # Similarly for V and W
        # Grad_V[b,r] = 2 * sum_{a,c} E[a,b,c] U[a,r] W[c,r]
        # Temp2[b,c,r] = sum_a E[a,b,c] U[a,r] = einsum('abc,ar->bcr', E, U)
        # Grad_V[b,r] = 2 * einsum('bcr,cr->br', Temp2, W)
        
        Temp2 = np.einsum('abc,ar->bcr', E, U)
        Grad_V = 2 * np.einsum('bcr,cr->br', Temp2, W)
        
        # Grad_W[c,r] = 2 * sum_{a,b} E[a,b,c] U[a,r] V[b,r]
        # Temp3[a,c,r] = sum_b E[a,b,c] V[b,r] = einsum('abc,br->acr', E, V)
        # Grad_W[c,r] = 2 * einsum('acr,ar->cr', Temp3, U)
        
        Temp3 = np.einsum('abc,br->acr', E, V)
        Grad_W = 2 * np.einsum('acr,ar->cr', Temp3, U)
        
        return Grad_U, Grad_V, Grad_W, np.sum(E**2)

    rng = np.random.default_rng(42)
    
    for r0 in range(17):
        if time.time() > deadline - 15:
            break
        keep = [r for r in range(17) if r != r0]
        U0 = U17[:, keep].copy()
        V0 = V17[:, keep].copy()
        W0 = W17[:, keep].copy()
        
        U_try = U0.copy()
        V_try = V0.copy()
        W_try = W0.copy()
        
        # Gradient Descent
        lr = 0.01
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            Grad_U, Grad_V, Grad_W, loss = _compute_gradients(U_try, V_try, W_try, T)
            
            # Check convergence
            if loss < 1e-12:
                break
            
            # Update
            U_try -= lr * Grad_U
            V_try -= lr * Grad_V
            W_try -= lr * Grad_W
            
            # Decay learning rate
            lr *= 0.95
            
        # Final check and snap
        T_try = _tensor_from_matrices(U_try, V_try, W_try)
        err = np.max(np.abs(T_try - T))
        
        if err < 1e-9:
            # Snap to integers or half-integers
            U_snap = np.round(U_try * 2) / 2
            V_snap = np.round(V_try * 2) / 2
            W_snap = np.round(W_try * 2) / 2
            
            T_snap = _tensor_from_matrices(U_snap, V_snap, W_snap)
            if np.allclose(T_snap, T, atol=1e-9):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # If gradient descent didn't work, try a few random restarts with small perturbations
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            U_try = U0 + rng.normal(0, 0.1, U0.shape)
            V_try = V0 + rng.normal(0, 0.1, V0.shape)
            W_try = W0 + rng.normal(0, 0.1, W0.shape)
            
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                Grad_U, Grad_V, Grad_W, loss = _compute_gradients(U_try, V_try, W_try, T)
                if loss < 1e-12:
                    break
                U_try -= lr * Grad_U
                V_try -= lr * Grad_V
                W_try -= lr * Grad_W
                lr *= 0.95
                
            T_try = _tensor_from_matrices(U_try, V_try, W_try)
            err = np.max(np.abs(T_try - T))
            if err < 1e-9:
                U_snap = np.round(U_try * 2) / 2
                V_snap = np.round(V_try * 2) / 2
                W_snap = np.round(W_try * 2) / 2
                T_snap = _tensor_from_matrices(U_snap, V_snap, W_snap)
                if np.allclose(T_snap, T, atol=1e-9):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # First try to find rank-16
    result_16 = _try_rank16_search(start_time, deadline)
    if result_16 is not None:
        return result_16
    
    # Fall back to rank-17 solution
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
