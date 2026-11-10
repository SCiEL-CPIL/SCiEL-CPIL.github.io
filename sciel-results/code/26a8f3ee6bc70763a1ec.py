# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    idx_a = X.astype(np.int64).ravel()
    idx_b = (X + Y).astype(np.int64).ravel()
    idx_c = (X - Y + 9).astype(np.int64).ravel()
    idx_d = (X + 2 * Y).astype(np.int64).ravel()
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _log_entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return -1e30
    return np.log(_entropy(q))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _soft_max_entropies(qa, qb, qd, temp=0.1):
    ea = _entropy(qa)
    eb = _entropy(qb)
    ed = _entropy(qd)
    arr = np.array([ea, eb, ed])
    if np.max(arr) < 1e-12:
        return 1e-12
    # Log-sum-exp for stability
    max_e = np.max(arr)
    logsumexp = max_e + np.log(np.sum(np.exp((arr - max_e) / temp)) * temp)
    # Wait, that's not quite right. 
    # log(sum(exp(x_i / temp))) * temp = log(sum(exp(x_i/temp))) * temp
    # Actually, soft max with temperature T: T * log(sum(exp(x_i / T)))
    # So:
    scaled = arr / temp
    max_s = np.max(scaled)
    logsumexp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return temp * logsumexp

def _optimize_weights_for_support(mask, idx_a, idx_b, idx_c, idx_d, max_iter=2000, use_log_ratio=True):
    support_indices = np.where(mask.ravel() > 0)[0]
    k = len(support_indices)
    if k == 0:
        return 0.0, None
    
    z0 = np.zeros(k)
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    def _compute_q(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        w = exp_z / exp_z.sum()
        pf = np.zeros(100)
        pf[support_indices] = w
        qa = np.bincount(idx_a, weights=pf, minlength=la)
        qb = np.bincount(idx_b, weights=pf, minlength=lb)
        qc = np.bincount(idx_c, weights=pf, minlength=lc)
        qd = np.bincount(idx_d, weights=pf, minlength=ld)
        return qa, qb, qc, qd, w, pf

    def objective(z):
        qa, qb, qc, qd, w, pf = _compute_q(z)
        D = _entropy(qc)
        M = _soft_max_entropies(qa, qb, qd, temp=0.05)
        if M < 1e-12:
            return 1e9
        if use_log_ratio:
            if D < 1e-12:
                return 1e9
            return -(np.log(D) - np.log(M))
        else:
            return -D / M

    try:
        res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            z_opt = res.x
            z_shifted = z_opt - np.max(z_opt)
            exp_z = np.exp(z_shifted)
            w = exp_z / exp_z.sum()
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
        else:
            w = np.ones(k) / k
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
    except Exception:
        return 0.0, None

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    best_val = -1e9
    best_P = np.full((10, 10), 1.0 / 100.0)
    
    val_u = _value_from_P(best_P, idx_a, idx_b, idx_c, idx_d)
    if val_u > best_val:
        best_val = val_u
        best_P = best_P.copy()
        
    rng = np.random.default_rng(42)
    
    num_restarts = 100
    max_flips_per_restart = 300
    
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Diverse initial masks
        if r % 5 == 0:
            # Random
            mask = rng.random((10, 10)) > 0.7
        elif r % 5 == 1:
            # Diagonal
            mask = np.zeros((10, 10), dtype=bool)
            mask[np.arange(10), np.arange(10)] = True
            # Add some noise
            for _ in range(10):
                mask[rng.integers(0, 10), rng.integers(0, 10)] = True
        elif r % 5 == 2:
            # Anti-diagonal
            mask = np.zeros((10, 10), dtype=bool)
            mask[np.arange(10), 9 - np.arange(10)] = True
            for _ in range(10):
                mask[rng.integers(0, 10), rng.integers(0, 10)] = True
        elif r % 5 == 3:
            # Sparse random
            mask = rng.random((10, 10)) > 0.85
        else:
            # Cluster
            cy, cx = rng.integers(0, 10), rng.integers(0, 10)
            mask = np.zeros((10, 10), dtype=bool)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = (cy + dy) % 10, (cx + dx) % 10
                    if rng.random() > 0.3:
                        mask[ny, nx] = True

        if mask.sum() < 2:
            mask.flat[rng.integers(0, 100)] = True
            mask.flat[rng.integers(0, 100)] = True
            
        current_mask = mask.copy()
        
        val, P = _optimize_weights_for_support(current_mask, idx_a, idx_b, idx_c, idx_d)
        if P is not None and val > best_val:
            best_val = val
            best_P = P.copy()
            
        def _value_uniform_mask(mask):
            pf = np.zeros(100)
            m_flat = mask.ravel()
            if m_flat.sum() == 0:
                return -1e9
            pf[m_flat] = 1.0 / m_flat.sum()
            P = pf.reshape(10, 10)
            return _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            
        current_val_uniform = _value_uniform_mask(current_mask)
        
        for flip_iter in range(max_flips_per_restart):
            if time.time() > deadline - 15.0:
                break
                
            idx_y = rng.integers(0, 10)
            idx_x = rng.integers(0, 10)
            
            current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
            
            if current_mask.sum() < 2:
                current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
                continue
                
            new_val_uniform = _value_uniform_mask(current_mask)
            
            if new_val_uniform > current_val_uniform:
                current_val_uniform = new_val_uniform
            else:
                current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
                
            if flip_iter % 5 == 0:
                val_opt, P_opt = _optimize_weights_for_support(current_mask, idx_a, idx_b, idx_c, idx_d)
                if P_opt is not None and val_opt > best_val:
                    best_val = val_opt
                    best_P = P_opt.copy()
                    
    # Final polish on best_P's support
    best_support = best_P > 1e-6
    if best_support.sum() > 0:
        val_final, P_final_opt = _optimize_weights_for_support(best_support, idx_a, idx_b, idx_c, idx_d, max_iter=5000)
        if P_final_opt is not None and val_final > best_val:
            best_val = val_final
            best_P = P_final_opt.copy()

    P_final = _finalize(best_P, idx_a, idx_b, idx_c, idx_d)
    return P_final

# EVOLVE-BLOCK-END
