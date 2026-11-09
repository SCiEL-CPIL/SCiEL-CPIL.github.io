# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

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
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    D = _entropy(qc)
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    def compute_entropies(P_flat):
        """Compute H(x), H(x+y), H(x-y), H(x+2y) from flat P."""
        qa = np.bincount(idx_a, weights=P_flat, minlength=la)
        qb = np.bincount(idx_b, weights=P_flat, minlength=lb)
        qc = np.bincount(idx_c, weights=P_flat, minlength=lc)
        qd = np.bincount(idx_d, weights=P_flat, minlength=ld)
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        return Ha, Hb, Hc, Hd
    
    def smooth_max(Ha, Hb, Hd, beta):
        """Smooth approximation of max using log-sum-exp."""
        vals = np.array([Ha, Hb, Hd])
        m = np.max(vals)
        if m < 1e-300:
            return 0.0
        log_sum = np.log(np.sum(np.exp(beta * (vals - m)))) / beta + m
        return log_sum
    
    def objective_smooth(z, beta):
        """Objective with smooth max surrogate. Minimize -D/M."""
        P = _softmax(z)
        Ha, Hb, Hc, Hd = compute_entropies(P)
        D = Hc
        M_smooth = smooth_max(Ha, Hb, Hd, beta)
        if M_smooth < 1e-12:
            return 1e9
        return -D / M_smooth
    
    def objective_exact(z):
        """Exact objective. Minimize -D/M."""
        P = _softmax(z)
        Ha, Hb, Hc, Hd = compute_entropies(P)
        D = Hc
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return 1e9
        return -D / M
    
    def objective_grad(z):
        """Objective with gradient for L-BFGS-B."""
        P = _softmax(z)
        Ha, Hb, Hc, Hd = compute_entropies(P)
        D = Hc
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return 1e9, np.zeros_like(z)
        
        val = -D / M
        
        # Compute gradients via finite differences or analytical
        # For simplicity, use numerical gradient
        eps = 1e-7
        grad = np.zeros_like(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            val_plus = objective_exact(z_plus)
            grad[i] = (val_plus - val) / eps
        return val, grad
    
    best_z = None
    best_val = -1e9
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = []
    
    # Uniform
    initial_points.append(np.zeros(100))
    
    # Random starts
    for _ in range(20):
        initial_points.append(rng.normal(0, 1.5, 100))
    
    # Structured: concentrate on anti-diagonals (high x-y spread)
    # For each anti-diagonal k (x-y = k-9), try concentrating there
    for k in range(0, 19):
        z = np.full(100, -8.0)
        for y in range(10):
            x = k + y - 9
            if 0 <= x < 10:
                z[y * 10 + x] = 0.0
        initial_points.append(z.copy())
    
    # Structured: spread across multiple anti-diagonals
    for k1 in range(0, 19, 3):
        for k2 in range(k1 + 3, 19, 3):
            z = np.full(100, -8.0)
            for y in range(10):
                x1 = k1 + y - 9
                if 0 <= x1 < 10:
                    z[y * 10 + x1] = 0.0
                x2 = k2 + y - 9
                if 0 <= x2 < 10:
                    z[y * 10 + x2] = 0.0
            initial_points.append(z.copy())
    
    # Try concentrating on specific columns to reduce H(x)
    for x0 in range(0, 10, 2):
        z = np.full(100, -8.0)
        for y in range(10):
            z[y * 10 + x0] = 0.0
        initial_points.append(z.copy())
    
    # Diagonal concentration (reduces H(x+y))
    for d in range(0, 18, 3):
        z = np.full(100, -8.0)
        for y in range(10):
            x = d - y
            if 0 <= x < 10:
                z[y * 10 + x] = 0.0
        initial_points.append(z.copy())
    
    # Phase 1: Smooth optimization with L-BFGS-B
    beta = 50.0  # temperature for log-sum-exp
    
    for i, z0 in enumerate(initial_points):
        if time.time() > deadline - 150:
            break
        try:
            res = minimize(
                objective_smooth, z0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            P = _softmax(res.x)
            v = _value_from_P(P.reshape(10, 10), idx_a, idx_b, idx_c, idx_d)
            if v > best_val:
                best_val = v
                best_z = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Refine with exact objective using L-BFGS-B
    if best_z is not None:
        if time.time() < deadline - 90:
            try:
                res2 = minimize(
                    objective_exact, best_z,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                P = _softmax(res2.x)
                v = _value_from_P(P.reshape(10, 10), idx_a, idx_b, idx_c, idx_d)
                if v > best_val:
                    best_val = v
                    best_z = res2.copy()
            except Exception:
                pass
    
    # Phase 3: Try different beta values for smooth optimization
    for beta2 in [20.0, 100.0, 200.0]:
        if time.time() > deadline - 90:
            break
        for i in range(min(8, len(initial_points))):
            z0 = initial_points[i]
            try:
                res = minimize(
                    objective_smooth, z0, args=(beta2,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                P = _softmax(res.x)
                v = _value_from_P(P.reshape(10, 10), idx_a, idx_b, idx_c, idx_d)
                if v > best_val:
                    best_val = v
                    best_z = res.x.copy()
            except Exception:
                continue
    
    # Phase 4: Final refinement with exact objective
    if best_z is not None and time.time() < deadline - 40:
        try:
            res3 = minimize(
                objective_exact, best_z,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            P = _softmax(res3.x)
            v = _value_from_P(P.reshape(10, 10), idx_a, idx_b, idx_c, idx_d)
            if v > best_val:
                best_val = v
                best_z = res3.x.copy()
        except Exception:
            pass
    
    # Phase 5: Additional random restarts if time permits
    if best_z is not None and time.time() < deadline - 60:
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            z0 = best_z + rng.normal(0, 0.5, 100)
            try:
                res = minimize(
                    objective_exact, z0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                P = _softmax(res.x)
                v = _value_from_P(P.reshape(10, 10), idx_a, idx_b, idx_c, idx_d)
                if v > best_val:
                    best_val = v
                    best_z = res.x.copy()
            except Exception:
                continue
    
    if best_z is None:
        P = np.full((10, 10), 1.0 / 100.0)
        return P
    
    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
