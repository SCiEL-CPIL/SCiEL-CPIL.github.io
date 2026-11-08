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
    U_28 = np.zeros((nm, rank_28), dtype=int)
    V_28 = np.zeros((mp, rank_28), dtype=int)
    W_28 = np.zeros((pn, rank_28), dtype=int)
    
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
    
    def verify_decomposition(U, V, W):
        vals = compute_values(U, V, W)
        return bool(np.all(np.abs(vals - 1.0) < 1e-9))
    
    def compute_gradients(U, V, W, vals):
        residuals = vals - 1.0
        Uv = U[idx_ui, :]
        Vv = V[idx_vj, :]
        Wv = W[idx_wk, :]
        r_vec = residuals[:, None]
        UVW_U = r_vec * Vv * Wv
        UVW_V = r_vec * Uv * Wv
        UVW_W = r_vec * Uv * Vv
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        np.add.at(grad_U, idx_ui, UVW_U)
        np.add.at(grad_V, idx_vj, UVW_V)
        np.add.at(grad_W, idx_wk, UVW_W)
        return grad_U, grad_V, grad_W
    
    def try_snap(U, V, W):
        # Only allow integers and half-integers
        for scale in [1, 2]:
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if verify_decomposition(U_s, V_s, W_s):
                return U_s, V_s, W_s
        return None
    
    def polish(U, V, W, max_steps=2000):
        """High-precision polish using smaller steps"""
        for step in range(max_steps):
            if time.time() > deadline:
                break
            vals = compute_values(U, V, W)
            residuals = vals - 1.0
            loss = 0.5 * np.sum(residuals ** 2)
            if loss < 1e-18:
                break
            grad_U, grad_V, grad_W = compute_gradients(U, V, W, vals)
            # Use very small step size for polishing
            lr_p = 0.01
            U -= lr_p * grad_U
            V -= lr_p * grad_V
            W -= lr_p * grad_W
        return U, V, W
    
    start_time = time.time()
    deadline = start_time + 135
    
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        # More trials for lower ranks, use warm start from rank-28
        num_trials = 200 if target_rank <= 20 else 100
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            
            # Warm start: use first target_rank columns of rank-28 solution
            if trial < 20 and target_rank <= 28:
                U = U_28[:, :target_rank].astype(float).copy()
                V = V_28[:, :target_rank].astype(float).copy()
                W = W_28[:, :target_rank].astype(float).copy()
                # Add small random perturbation to escape local minima
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            else:
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            lr = 0.1
            m_U = np.zeros_like(U); v_U = np.zeros_like(U)
            m_V = np.zeros_like(V); v_V = np.zeros_like(V)
            m_W = np.zeros_like(W); v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            max_steps = 10000 if target_rank <= 20 else 5000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                vals = compute_values(U, V, W)
                residuals = vals - 1.0
                loss = 0.5 * np.sum(residuals ** 2)
                
                if loss < 1e-16:
                    found = True
                    break
                
                grad_U, grad_V, grad_W = compute_gradients(U, V, W, vals)
                
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                U -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                W -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
            
            if found or loss < 1e-12:
                # Polish the solution
                U, V, W = polish(U, V, W, max_steps=1000)
                result = try_snap(U, V, W)
                if result is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = result
                    found = True
                    break
                # Do not accept raw float solutions to ensure half-integer validity
        
        if found:
            break
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                # Snap to nearest integer or half-integer
                r = round(x * 2) / 2.0
                if abs(r - round(r)) < 1e-9:
                    new_row.append(int(round(r)))
                else:
                    new_row.append(r)
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
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
