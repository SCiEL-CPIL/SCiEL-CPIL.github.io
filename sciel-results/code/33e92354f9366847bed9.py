import numpy as np
import time
from scipy.optimize import minimize

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

# Number of bins for each
_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

# Precompute pooling matrices (sparse-style, but dense for 100 cells)
# Each matrix maps flat P to pooled probabilities
def _make_pooling_matrix(vals, n_bins):
    M = np.zeros((n_bins, _N * _N), dtype=np.float64)
    for i in range(_N * _N):
        M[vals[i], i] = 1.0
    return M

_M_D = _make_pooling_matrix(_d_vals, _N_D)
_M_H0 = _make_pooling_matrix(_h0_vals, _N_H0)
_M_H1 = _make_pooling_matrix(_h1_vals, _N_H1)
_M_H2 = _make_pooling_matrix(_h2_vals, _N_H2)

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _smooth_max_entropy(probs, beta=50.0):
    """Smooth approximation of max entropy using log-sum-exp trick.
    Returns a smooth approximation of max(H0, H1, H2) where Hi are entropies."""
    # This is used in the smooth objective for gradient-based optimization
    pass  # Not used directly; we handle it in the negative ratio function

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten()
    
    # Compute pooled probabilities for each slope
    d_probs = _M_D @ w
    h0_probs = _M_H0 @ w
    h1_probs = _M_H1 @ w
    h2_probs = _M_H2 @ w
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _log_softmax_params_to_prob(params):
    """Convert unconstrained params to probability simplex via softmax."""
    params = params - np.max(params)  # numerical stability
    exp_params = np.exp(params)
    return exp_params / exp_params.sum()

def _negative_ratio_smooth(params, beta=100.0):
    """Negative smooth ratio for minimization. Uses smooth max approximation."""
    w = _log_softmax_params_to_prob(params)
    
    d_probs = _M_D @ w
    h0_probs = _M_H0 @ w
    h1_probs = _M_H1 @ w
    h2_probs = _M_H2 @ w
    
    # Compute entropies
    def _ent(p):
        p = np.clip(p, 1e-15, None)
        return -np.sum(p * np.log2(p))
    
    D = _ent(d_probs)
    H0 = _ent(h0_probs)
    H1 = _ent(h1_probs)
    H2 = _ent(h2_probs)
    
    # Smooth max using log-sum-exp
    Hs = np.array([H0, H1, H2])
    max_H = np.max(Hs)
    smooth_max = max_H + np.log(np.sum(np.exp((Hs - max_H) * beta))) / beta
    
    if smooth_max <= 1e-9:
        return 0.0
        
    return -D / smooth_max

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 15
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    # Try several initialization strategies
    def _normalize(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return None
        return P / s
    
    # Strategy 1: Uniform (already computed)
    # Strategy 2: Diagonal-heavy (maximize D by spreading x-y values)
    # Strategy 3: Random Dirichlet-like
    # Strategy 4: Concentrate on cells that give unique x-y values
    
    # For D = H(x-y), we want to spread mass across different (x-y) values
    # For M = max(H0,H1,H2), we want to minimize the max entropy
    # H0 = H(x): spread x values
    # H1 = H(x+y): spread x+y values  
    # H2 = H(x+2y): spread x+2y values
    
    # Key insight: To maximize D/M, we want D high and M low.
    # D is high when mass is spread across many x-y values.
    # M is low when one of H0, H1, H2 is low.
    # H0 is low if mass is concentrated on few x values.
    # So: spread across x-y values but concentrate on few x values.
    # This means: for each x, spread across y values that give different x-y.
    # But x-y = x-y, so for fixed x, different y give different x-y.
    # So for fixed x, all 10 y values give 10 different x-y values.
    # If we put all mass on one x, D can be at most log2(10) bits.
    # But H0 would be 0, making M potentially small.
    
    # Let's try: put all mass on x=0, spread evenly across y
    # Then D = H(-y) = H(y) = log2(10) ≈ 3.32
    # H0 = H(0) = 0
    # H1 = H(y) = log2(10)
    # H2 = H(2y) = log2(10)
    # M = log2(10), D/M = 1.0
    
    # Better: put mass on multiple x values but in a way that minimizes H0,H1,H2
    # while maximizing D.
    
    # Let's use a structured approach: try to find P that maximizes D/M
    # using the precomputed arrays
    
    def _score_fast(P_flat):
        """Fast score computation assuming P_flat is already normalized and non-negative."""
        d_probs = _M_D @ P_flat
        h0_probs = _M_H0 @ P_flat
        h1_probs = _M_H1 @ P_flat
        h2_probs = _M_H2 @ P_flat
        
        D = _entropy_from_counts(d_probs)
        H0 = _entropy_from_counts(h0_probs)
        H1 = _entropy_from_counts(h1_probs)
        H2 = _entropy_from_counts(h2_probs)
        
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 0.0
            
        return D / M
    
    # Phase 1: Gradient-based optimization using scipy L-BFGS-B
    # Work in log-softmax parameter space for unconstrained optimization
    time_budget = timeout - safety_margin
    time_for_gradient = time_budget * 0.6
    time_for_hill = time_budget * 0.4
    
    gradient_start = time.time()
    
    # Multiple restarts for gradient optimization
    num_grad_restarts = 8
    time_per_grad = time_for_gradient / num_grad_restarts
    
    for r in range(num_grad_restarts):
        if time.time() - gradient_start > time_for_gradient:
            break
            
        # Initialize params (log-softmax space)
        if r == 0:
            # Uniform
            init_P = np.ones((n, n)) / (n * n)
        elif r == 1:
            # Concentrate on x=0
            init_P = np.zeros((n, n))
            init_P[0, :] = 1.0 / n
        elif r == 2:
            # Concentrate on x=0 and x=9
            init_P = np.zeros((n, n))
            init_P[0, :] = 0.5 / n
            init_P[9, :] = 0.5 / n
        elif r == 3:
            # Diagonal
            init_P = np.zeros((n, n))
            np.fill_diagonal(init_P, 1.0 / n)
        elif r == 4:
            # Anti-diagonal
            init_P = np.zeros((n, n))
            for i in range(n):
                init_P[i, n-1-i] = 1.0 / n
        elif r == 5:
            # Best so far
            init_P = best_P
        else:
            # Random Dirichlet-like
            init_P = np.random.dirichlet(np.ones(n * n))
            init_P = init_P.reshape(n, n)
        
        # Convert to log-softmax params
        init_P_flat = np.clip(init_P.flatten(), 1e-10, None)
        init_params = np.log(init_P_flat)
        
        # Optimize
        try:
            result = minimize(
                _negative_ratio_smooth,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_params = result.x
                opt_P = _log_softmax_params_to_prob(opt_params).reshape(n, n)
                opt_score = _compute_ratio(opt_P)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_P = opt_P.copy()
        except Exception:
            pass
    
    # Phase 2: Hill climbing polish
    hill_start = time.time()
    time_remaining = timeout - safety_margin - (time.time() - start_time)
    
    if time_remaining > 5:
        # Use remaining time for hill climbing
        num_restarts = min(15, max(3, int(time_remaining / 5)))
        time_per_restart = time_remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() - start_time > timeout - safety_margin:
                break
                
            # Initialize current P
            if r == 0:
                current_P = best_P.copy()
            else:
                # Perturb best found
                current_P = best_P + np.random.normal(0, 0.1, (n, n))
                current_P = _normalize(current_P)
                if current_P is None:
                    current_P = best_P.copy()
                    
            current_score = _compute_ratio(current_P)
            
            # Hill climbing with adaptive step size
            step_size = 0.2
            local_time_limit = time_per_restart
            local_start = time.time()
            no_improve_count = 0
            max_no_improve = 100
            
            while time.time() - local_start < local_time_limit:
                # Try a few random moves per iteration
                for _ in range(3):
                    if time.time() - local_start >= local_time_limit:
                        break
                        
                    i, j = np.random.randint(0, n, 2)
                    delta = np.random.uniform(-step_size, step_size)
                    
                    new_P = current_P.copy()
                    new_P[i, j] += delta
                    
                    if new_P[i, j] < -1e-9:
                        continue
                        
                    new_P = np.clip(new_P, 0, None)
                    s = new_P.sum()
                    if s <= 1e-9:
                        continue
                    new_P = new_P / s
                    
                    new_score = _score_fast(new_P.flatten())
                    
                    if new_score > current_score:
                        current_P = new_P
                        current_score = new_score
                        no_improve_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_P = current_P.copy()
                        step_size *= 0.97
                    else:
                        no_improve_count += 1
                        if no_improve_count > max_no_improve:
                            # Restart local search from a perturbation of best
                            current_P = best_P + np.random.normal(0, 0.1, (n, n))
                            current_P = _normalize(current_P)
                            if current_P is None:
                                current_P = best_P.copy()
                            current_score = _compute_ratio(current_P)
                            no_improve_count = 0
                            step_size = 0.2
                            break
                            
    return best_P
