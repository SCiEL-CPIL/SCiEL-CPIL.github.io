import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # Precompute all tensor index triples
    idx_ui = []
    idx_vj = []
    idx_wk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ui.append(i * m + j)
                idx_vj.append(j * p + k)
                idx_wk.append(k * n + i)
    idx_ui = np.array(idx_ui)
    idx_vj = np.array(idx_vj)
    idx_wk = np.array(idx_wk)
    num_entries = len(idx_ui)  # 32
    
    # Construct rank-28 Strassen solution as fallback
    rank_28 = 28
    U_28 = np.zeros((nm, rank_28), dtype=float)
    V_28 = np.zeros((mp, rank_28), dtype=float)
    W_28 = np.zeros((pn, rank_28), dtype=float)
    
    def strassen_terms():
        terms = []
        u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
        terms.append((u, v, w))
        u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
        terms.append((u, v, w))
        u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
        terms.append((u, v, w))
        u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
        terms.append((u, v, w))
        u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
        terms.append((u, v, w))
        u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
        terms.append((u, v, w))
        u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
        terms.append((u, v, w))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0, U_28, V_28, W_28)
    apply_strassen(1, 1, 0, 0, 7, U_28, V_28, W_28)
    apply_strassen(0, 0, 1, 1, 14, U_28, V_28, W_28)
    apply_strassen(1, 1, 1, 1, 21, U_28, V_28, W_28)
    
    best_rank = rank_28
    best_U = U_28.copy()
    best_V = V_28.copy()
    best_W = W_28.copy()
    
    def compute_values(U, V, W):
        Uv = U[idx_ui, :]
        Vv = V[idx_vj, :]
        Wv = W[idx_wk, :]
        return np.sum(Uv * Vv * Wv, axis=1)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        vals = compute_values(U, V, W)
        return bool(np.all(np.abs(vals - 1.0) < tol))
    
    def compute_gradients(U, V, W, vals):
        residuals = vals - 1.0
        Uv = U[idx_ui, :]
        Vv = V[idx_vj, :]
        Wv = W[idx_wk, :]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for t in range(num_entries):
            r = residuals[t]
            ui = idx_ui[t]
            vj = idx_vj[t]
            wk = idx_wk[t]
            grad_U[ui, :] += r * Vv[t] * Wv[t]
            grad_V[vj, :] += r * Uv[t] * Wv[t]
            grad_W[wk, :] += r * Uv[t] * Vv[t]
        return grad_U, grad_V, grad_W
    
    def try_snap(U, V, W):
        # The constraint is strictly integers or half-integers.
        # So we only need to check scale=2 (for half-integers) and scale=1 (integers).
        # However, numerical noise might require slightly larger scales to snap correctly if the true value is integer.
        # Let's try snapping to nearest 0.5 first.
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if verify_decomposition(U_s, V_s, W_s, tol=1e-9):
            return U_s, V_s, W_s
        
        # If that fails, maybe the true solution has integers and we are off by 0.5?
        # Try snapping to nearest integer.
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_decomposition(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i
            
        return None
    
    def adam_step(U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        vals = compute_values(U, V, W)
        residuals = vals - 1.0
        loss = 0.5 * np.sum(residuals ** 2)
        
        grad_U, grad_V, grad_W = compute_gradients(U, V, W, vals)
        
        t = step + 1
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        U_new = U - lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        V_new = V - lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        W_new = W - lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
        
        return U_new, V_new, W_new, m_U, v_U, m_V, v_V, m_W, v_W, loss
    
    def polish(U, V, W, max_steps=5000, lr_start=0.01, lr_end=0.0001):
        """High-precision polish with decaying learning rate."""
        m_U = np.zeros_like(U); v_U = np.zeros_like(U)
        m_V = np.zeros_like(V); v_V = np.zeros_like(V)
        m_W = np.zeros_like(W); v_W = np.zeros_like(W)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            lr = lr_start * (lr_end / lr_start) ** (step / max_steps)
            U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, loss = adam_step(
                U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, step, lr)
            if loss < 1e-20:
                break
        return U, V, W
    
    def try_rank_reduction(target_rank, U_init, V_init, W_init, num_trials=30, max_steps=4000):
        """Try to find a valid decomposition at target_rank starting from given init."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(1000 + target_rank * 100 + trial)
            
            # Initialize: take first target_rank columns of init, add small noise
            U = U_init[:, :target_rank].copy() + np.random.randn(nm, target_rank) * 0.1
            V = V_init[:, :target_rank].copy() + np.random.randn(mp, target_rank) * 0.1
            W = W_init[:, :target_rank].copy() + np.random.randn(pn, target_rank) * 0.1
            
            m_U = np.zeros_like(U); v_U = np.zeros_like(U)
            m_V = np.zeros_like(V); v_V = np.zeros_like(V)
            m_W = np.zeros_like(W); v_W = np.zeros_like(W)
            
            lr = 0.05
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                
                U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, loss = adam_step(
                    U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, step, lr)
                
                if loss < 1e-14:
                    # Polish
                    U, V, W = polish(U, V, W, max_steps=3000, lr_start=0.005, lr_end=0.00001)
                    if verify_decomposition(U, V, W):
                        result = try_snap(U, V, W)
                        if result is not None:
                            return result
                        if verify_decomposition(U, V, W, tol=1e-8):
                            return (U.copy(), V.copy(), W.copy())
                    return None
        
        return None
    
    def random_search(target_rank, num_trials=50, max_steps=5000):
        """Random initialization search."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + target_rank * 100 + trial)
            
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            m_U = np.zeros_like(U); v_U = np.zeros_like(U)
            m_V = np.zeros_like(V); v_V = np.zeros_like(V)
            m_W = np.zeros_like(W); v_W = np.zeros_like(W)
            
            lr = 0.05
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                
                U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, loss = adam_step(
                    U, V, W, m_U, v_U, m_V, v_V, m_W, v_W, step, lr)
                
                if loss < 1e-14:
                    U, V, W = polish(U, V, W, max_steps=3000, lr_start=0.005, lr_end=0.00001)
                    if verify_decomposition(U, V, W):
                        result = try_snap(U, V, W)
                        if result is not None:
                            return result
                        if verify_decomposition(U, V, W, tol=1e-8):
                            return (U.copy(), V.copy(), W.copy())
                    return None
        
        return None
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Strategy 1: Try rank reduction from the Strassen solution
    # We know rank 28 works. Let's try to find lower ranks.
    # The theoretical lower bound for 2x4x4 is not strictly 15, but let's search.
    # Note: The Strassen solution provided in the code is for 2x2x2 blocks.
    # The code constructs a rank 28 solution.
    # Let's attempt to reduce.
    found_better = False
    for target_rank in range(15, rank_28):
        if time.time() > deadline:
            break
        
        result = try_rank_reduction(target_rank, U_28, V_28, W_28, num_trials=10, max_steps=2000)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            found_better = True
            break
    
    # Strategy 2: Random search for lower ranks if we haven't found anything better
    if not found_better:
        for target_rank in range(16, rank_28):
            if time.time() > deadline:
                break
            
            result = random_search(target_rank, num_trials=20, max_steps=2000)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                found_better = True
                break
    
    # Strategy 3: Try to go even lower if we found something
    if found_better:
        for target_rank in range(max(14, best_rank - 5), best_rank):
            if time.time() > deadline:
                break
            
            result = try_rank_reduction(target_rank, best_U, best_V, best_W, num_trials=10, max_steps=1500)
            if result is None:
                result = random_search(target_rank, num_trials=10, max_steps=1500)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                break
    
    def to_list_strict(mat):
        """Convert numpy array to list of ints/half-ints, verifying constraints."""
        result = []
        for row in mat:
            new_row = []
            for x in row:
                # Snap to nearest 0.5
                snapped = np.round(x * 2) / 2.0
                # Check if it's effectively an integer
                if abs(snapped - round(snapped)) < 1e-9:
                    new_row.append(int(round(snapped)))
                else:
                    # It's a half-integer (e.g., 0.5, 1.5)
                    # Represent as float, evaluator should accept 0.5, 1.5 etc.
                    # However, the error says "integers or half-integers".
                    # 0.5 is a half-integer. 1.5 is a half-integer.
                    new_row.append(float(snapped))
            result.append(new_row)
        return result

    # Final verification of the best candidate before returning
    # If the best candidate fails strict verification, fallback to snapped rank-28
    U_final = np.round(best_U * 2) / 2.0
    V_final = np.round(best_V * 2) / 2.0
    W_final = np.round(best_W * 2) / 2.0
    
    if not verify_decomposition(U_final, V_final, W_final, tol=1e-9):
        # Fallback to the known valid rank-28 Strassen solution (which uses integers)
        U_final = np.round(U_28 * 2) / 2.0
        V_final = np.round(V_28 * 2) / 2.0
        W_final = np.round(W_28 * 2) / 2.0
        # U_28 etc are constructed with integers, so this should be safe
        if not verify_decomposition(U_final, V_final, W_final, tol=1e-9):
            raise ValueError("Failed to construct valid decomposition")

    return {"U": to_list_strict(U_final), "V": to_list_strict(V_final), "W": to_list_strict(W_final)}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
