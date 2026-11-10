import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0
    
    n = 10
    # Create coordinate arrays
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    # P[y, x] so P[i, j] corresponds to Y=i, X=j
    
    # Precompute projection index arrays (flattened)
    proj_m1_flat = (X - Y).ravel() + 9  # range 0..18
    proj_0_flat = X.ravel()             # range 0..9
    proj_1_flat = (X + Y).ravel()       # range 0..18
    proj_2_flat = (X + 2 * Y).ravel()   # range 0..27
    
    def _entropy(p):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        s = p.sum()
        if s <= 1e-30:
            return 0.0
        q = p / s
        q = q[q > 0.0]
        if len(q) == 0:
            return 0.0
        return float(-(q * np.log2(q)).sum())
    
    def _compute_value(P):
        """Compute the ratio D/M for a given distribution P using hard max."""
        p = P.ravel()
        total = p.sum()
        if total <= 1e-30:
            return 0.0
        
        # D: distribution of x - y (slope -1), 19 bins
        d_bins = np.bincount(proj_m1_flat, weights=p, minlength=19)
        hd = _entropy(d_bins)
        
        # M: max of H(x), H(x+y), H(x+2y)
        h0 = _entropy(np.bincount(proj_0_flat, weights=p, minlength=10))
        h1 = _entropy(np.bincount(proj_1_flat, weights=p, minlength=19))
        h2 = _entropy(np.bincount(proj_2_flat, weights=p, minlength=28))
        
        m = max(h0, h1, h2)
        if m <= 1e-15:
            return 0.0
        
        return hd / m
    
    def _compute_value_smooth(P, beta=1.0):
        """Compute a smooth version of D/M using soft-max for M."""
        p = P.ravel()
        total = p.sum()
        if total <= 1e-30:
            return 0.0
        
        d_bins = np.bincount(proj_m1_flat, weights=p, minlength=19)
        hd = _entropy(d_bins)
        
        h0 = _entropy(np.bincount(proj_0_flat, weights=p, minlength=10))
        h1 = _entropy(np.bincount(proj_1_flat, weights=p, minlength=19))
        h2 = _entropy(np.bincount(proj_2_flat, weights=p, minlength=28))
        
        hs = np.array([h0, h1, h2])
        # Soft max with temperature
        shifted = hs / beta
        max_val = np.max(shifted)
        exp_shifted = np.exp(shifted - max_val)
        log_sum_exp = max_val + np.log(np.sum(exp_shifted))
        m_soft = log_sum_exp * beta
        
        if hd <= 1e-15 or m_soft <= 1e-15:
            return 0.0
        
        return hd / m_soft
    
    def _build_from_theta(theta):
        """Build probability array from log-space parameters."""
        theta_clipped = np.clip(theta, -50, 50)
        exp_theta = np.exp(theta_clipped)
        P = exp_theta / np.sum(exp_theta)
        return P.reshape(n, n)
    
    def _objective_smooth(theta, beta=0.01):
        """Objective function for optimization: minimize -log(D/M) using smooth M."""
        P = _build_from_theta(theta)
        p = P.ravel()
        
        d_bins = np.bincount(proj_m1_flat, weights=p, minlength=19)
        hd = _entropy(d_bins)
        
        h0 = _entropy(np.bincount(proj_0_flat, weights=p, minlength=10))
        h1 = _entropy(np.bincount(proj_1_flat, weights=p, minlength=19))
        h2 = _entropy(np.bincount(proj_2_flat, weights=p, minlength=28))
        
        hs = np.array([h0, h1, h2])
        shifted = hs / beta
        max_val = np.max(shifted)
        exp_shifted = np.exp(shifted - max_val)
        log_sum_exp = max_val + np.log(np.sum(exp_shifted))
        m_soft = log_sum_exp * beta
        
        if hd <= 1e-15 or m_soft <= 1e-15:
            return 1e10
        
        return -np.log(hd / m_soft)
    
    def _objective_and_grad(theta, beta=0.01):
        """Objective and gradient using finite differences."""
        obj = _objective_smooth(theta, beta)
        
        eps = 1e-6
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            obj_plus = _objective_smooth(theta_plus, beta)
            grad[i] = (obj_plus - obj) / eps
        
        return obj, grad
    
    def _fast_greedy_sparse(n_steps=50, n_candidates=200):
        """
        Greedy sparse search: start with random support, then add cells
        that improve the ratio the most.
        """
        best_P = None
        best_score = -np.inf
        
        # Try multiple random initial supports
        for trial in range(5):
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin:
                break
            
            # Start with a small random support
            support_size = np.random.randint(3, 8)
            indices = np.random.choice(n * n, size=support_size, replace=False)
            P = np.zeros((n, n))
            P.ravel()[indices] = 1.0
            P = P / P.sum()
            
            current_score = _compute_value(P)
            if current_score <= 1e-15:
                continue
            
            # Greedily add cells
            for step in range(n_steps):
                elapsed = time.time() - start_time
                if elapsed > deadline - safety_margin:
                    break
                
                # Try adding each non-supported cell
                best_cell = -1
                best_new_score = current_score
                best_new_P = None
                
                supported = set(indices.tolist())
                for cell in range(n * n):
                    if cell in supported:
                        continue
                    
                    P_new = P.copy()
                    P_new.ravel()[cell] = 1.0
                    P_new = P_new / P_new.sum()
                    
                    score_new = _compute_value(P_new)
                    if score_new > best_new_score:
                        best_new_score = score_new
                        best_cell = cell
                        best_new_P = P_new.copy()
                
                if best_cell == -1:
                    break
                
                P = best_new_P
                current_score = best_new_score
                indices = np.append(indices, best_cell)
            
            if current_score > best_score:
                best_score = current_score
                best_P = P.copy()
        
        return best_P, best_score
    
    def _coordinate_polish(P, n_rounds=20):
        """
        Coordinate-wise optimization: adjust each cell's weight to improve the ratio.
        """
        P_work = P.copy()
        
        for round_num in range(n_rounds):
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n * n):
                # Try different weights for cell i
                current_score = _compute_value(P_work)
                
                # Try a range of multipliers for this cell
                base_val = P_work.ravel()[i]
                if base_val < 1e-15:
                    # Try adding a small weight
                    for mult in [0.01, 0.1, 0.5, 1.0, 5.0]:
                        P_try = P_work.copy()
                        P_try.ravel()[i] = mult * (1.0 / n)  # small base
                        P_try = P_try / P_try.sum()
                        score_try = _compute_value(P_try)
                        if score_try > current_score:
                            P_work = P_try
                            current_score = score_try
                            improved = True
                    continue
                
                # Try scaling this cell
                for scale in [0.1, 0.5, 2.0, 5.0, 0.01]:
                    P_try = P_work.copy()
                    P_try.ravel()[i] = base_val * scale
                    if P_try.sum() <= 1e-15:
                        continue
                    P_try = P_try / P_try.sum()
                    score_try = _compute_value(P_try)
                    if score_try > current_score:
                        P_work = P_try
                        current_score = score_try
                        improved = True
            
            if not improved:
                break
        
        return P_work
    
    # Main search
    best_P = None
    best_score = -np.inf
    
    # 1. Try uniform distribution
    P_uniform = np.ones((n, n)) / (n * n)
    score_uniform = _compute_value(P_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_P = P_uniform.copy()
    
    # 2. Try structured patterns
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 30:
        # Try lines with various slopes
        for a in [0, 1, 2, -1, 0.5, -0.5, 3, -2]:
            for b in range(-10, 20):
                P_line = np.zeros((n, n))
                count = 0
                for yy in range(n):
                    xx = int(round(a * yy + b))
                    if 0 <= xx < n:
                        P_line[yy, xx] = 1.0
                        count += 1
                if count >= 3:
                    P_line = P_line / count
                    score_l = _compute_value(P_line)
                    if score_l > best_score:
                        best_score = score_l
                        best_P = P_line.copy()
        
        # Try 2D patterns
        patterns = {
            'checkerboard': lambda i, j: (i + j) % 2 == 0,
            'diagonals_mod3': lambda i, j: (i - j) % 3 == 0,
            'cross': lambda i, j: i == n // 2 or j == n // 2,
            'ring': lambda i, j: 3 <= abs(i - n//2) + abs(j - n//2) <= 6,
            'corners': lambda i, j: (i < 3 or i >= 7) and (j < 3 or j >= 7),
        }
        
        for name, func in patterns.items():
            P_pat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if func(i, j):
                        P_pat[i, j] = 1.0
            total = P_pat.sum()
            if total > 0:
                P_pat = P_pat / total
                score_p = _compute_value(P_pat)
                if score_p > best_score:
                    best_score = score_p
                    best_P = P_pat.copy()
    
    # 3. Greedy sparse search
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 60:
        P_greedy, score_greedy = _fast_greedy_sparse(n_steps=30, n_candidates=100)
        if P_greedy is not None and score_greedy > best_score:
            best_score = score_greedy
            best_P = P_greedy.copy()
    
    # 4. Continuous optimization from best found
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 120 and best_P is not None:
        try:
            P_safe = np.clip(best_P, 1e-15, None)
            P_norm = P_safe / P_safe.sum()
            theta_init = np.log(P_norm.ravel())
            
            # Use L-BFGS-B for efficiency
            result = minimize(
                _objective_smooth,
                theta_init,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e9:
                P_opt = _build_from_theta(result.x)
                P_opt = np.clip(P_opt, 0, None)
                P_opt = P_opt / P_opt.sum()
                score_opt = _compute_value(P_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # 5. Coordinate polish on best found
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 30 and best_P is not None:
        P_polished = _coordinate_polish(best_P, n_rounds=10)
        score_polished = _compute_value(P_polished)
        if score_polished > best_score:
            best_score = score_polished
            best_P = P_polished.copy()
    
    # 6. Try a few more random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 60:
        for trial in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin:
                break
            
            # Random initialization with some structure
            theta_rand = np.random.randn(n * n) * 0.5
            P_rand = _build_from_theta(theta_rand)
            score_rand = _compute_value(P_rand)
            
            if score_rand > best_score * 0.8:  # Only optimize if promising
                try:
                    result = minimize(
                        _objective_smooth,
                        theta_rand,
                        args=(0.01,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    if result.fun < 1e9:
                        P_opt = _build_from_theta(result.x)
                        P_opt = np.clip(P_opt, 0, None)
                        P_opt = P_opt / P_opt.sum()
                        score_opt = _compute_value(P_opt)
                        if score_opt > best_score:
                            best_score = score_opt
                            best_P = P_opt.copy()
                except Exception:
                    pass
    
    # Ensure validity
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P
