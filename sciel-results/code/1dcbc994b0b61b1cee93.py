"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 100

# EVOLVE-BLOCK-START
def _overlap_score(h, shifts=None):
    """Compute max overlap of h with shifted (1-h) over period 2."""
    n = len(h)
    if shifts is None:
        shifts = np.arange(n)
    max_ov = 0.0
    for s in shifts:
        shifted = np.roll(h, int(s))
        ov = np.sum(h * (1.0 - shifted)) / n
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(width, pos, order, n=N):
    """
    Construct h on [0,2] discretized into n points.
    h is 0 on [0, pos-width/2), transitions over [pos-width/2, pos+width/2],
    then 1 on [pos+width/2, 2].
    Uses binomial CDF for smooth transition.
    """
    x = np.linspace(0, 2, n, endpoint=False)
    h = np.zeros(n)
    
    # Determine which points are in the transition region
    t_start = pos - width / 2.0
    t_end = pos + width / 2.0
    
    # Points before transition: h = 0
    # Points after transition: h = 1
    # Points in transition: use binomial CDF
    
    in_trans = (x >= t_start) & (x <= t_end)
    after_trans = x > t_end
    
    h[after_trans] = 1.0
    
    if np.any(in_trans):
        # Normalize x within transition window to [0, 1]
        x_in = x[in_trans]
        if width > 1e-10:
            t = (x_in - t_start) / width
            t = np.clip(t, 0, 1)
        else:
            t = np.zeros_like(x_in)
        
        # Binomial CDF: P(Binomial(order, 0.5) <= k) for k = floor(t * order)
        # Use the CDF of a symmetric binomial
        k = np.floor(t * order).astype(int)
        k = np.clip(k, 0, order)
        
        # Compute binomial CDF P(X <= k) where X ~ Binomial(order, 0.5)
        # For symmetry, use: CDF(k) = 0.5 + 0.5 * (2*P(X=k) + P(X<k) - P(X>k))
        # Simpler: use scipy or compute directly
        # P(X <= k) for Binomial(n, 0.5)
        # Use the regularized incomplete beta function
        from scipy.special import betainc
        # CDF of Binomial(n, p) at k = betainc(n-k, k+1, 1-p, p) ... 
        # Actually: P(X <= k) = 1 - betainc(n-k, k+1, p, 1-p) for k < n
        # Let's use a more direct approach
        # P(X <= k) = sum_{i=0}^{k} C(n,i) (0.5)^n
        # For large n, use normal approximation or betainc
        
        # betainc(a, b, x) = regularized incomplete beta function I_x(a,b)
        # P(X <= k) for Binomial(n, p) = betainc(n-k, k+1, 1-p, p)
        # But we need to handle edge cases
        
        cdf_vals = np.zeros_like(k, dtype=float)
        for i, ki in enumerate(k):
            if ki >= order:
                cdf_vals[i] = 1.0
            elif ki < 0:
                cdf_vals[i] = 0.0
            else:
                # P(X <= ki) for Binomial(order, 0.5)
                # = I_{0.5}(order-ki, ki+1) ... let me use scipy.stats
                cdf_vals[i] = 0.5  # fallback, will fix
                # Use: P(X <= k) = betainc(order-k, k+1, 1-0.5, 0.5)
                # Actually the correct formula:
                # P(X <= k) = I_{1-p}(n-k, k+1) where I is regularized incomplete beta
                # For p=0.5: P(X <= k) = I_{0.5}(n-k, k+1)
                if order > 0:
                    cdf_vals[i] = float(_binom_cdf(ki, order, 0.5))
        
        h[in_trans] = cdf_vals
    
    # Enforce integral = 1: sum(h) * (2/n) = 1 => sum(h) = n/2
    current_sum = np.sum(h)
    if current_sum > 0:
        # Scale to get exact integral
        # But we need to keep values in [0,1]
        # If sum is too high, we need to reduce; if too low, increase
        # Simple approach: clip and adjust
        h = np.clip(h, 0, 1)
        new_sum = np.sum(h)
        if new_sum > 0:
            h = h * (n / 2.0) / new_sum
            h = np.clip(h, 0, 1)
            # Final adjustment: if still not exact, do a small uniform shift
            diff = n / 2.0 - np.sum(h)
            if abs(diff) > 1e-15:
                # Distribute the difference uniformly
                h = h + diff / n
                h = np.clip(h, 0, 1)
                # Re-adjust
                diff2 = n / 2.0 - np.sum(h)
                if abs(diff2) > 1e-15:
                    h = h + diff2 / n
    
    return h

def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)"""
    from scipy.stats import binom
    return binom.cdf(k, n, p)

def _score_params(params, n=N):
    """Score for given (width, pos, order)"""
    width, pos, order = params
    width = max(width, 1e-6)
    pos = pos % 2.0
    order = max(int(round(order)), 1)
    h = _construct_h(width, pos, order, n)
    return _overlap_score(h)

def construct():
    """
    Find h minimizing max overlap with shifted (1-h).
    Uses a single transition window with binomial-smoothed profile.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    n = N
    
    best_h = np.full(n, 0.5)
    best_score = _overlap_score(best_h)
    
    # Phase 1: Coarse grid search over width, pos, order
    widths = np.logspace(-2, 0.5, 15)  # 0.01 to ~3.16
    pos_vals = np.linspace(0, 2, 20, endpoint=False)
    orders = [2, 4, 8, 16, 32, 64]
    
    for order in orders:
        for w in widths:
            for p in pos_vals:
                if time.time() > deadline - 30:
                    break
                h = _construct_h(w, p, order, n)
                s = _overlap_score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
    
    # Phase 2: Local refinement using scipy minimize
    if time.time() < deadline - 10:
        # Find best initial point from grid
        best_params = None
        for order in orders:
            for w in widths:
                for p in pos_vals:
                    if time.time() > deadline - 15:
                        break
                    h = _construct_h(w, p, order, n)
                    s = _overlap_score(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
                        best_params = (w, p, float(order))
        
        if best_params is not None:
            # Optimize width and pos for each order around best
            for order in [max(2, int(best_params[2])//2), int(best_params[2]), int(best_params[2])*2]:
                if time.time() > deadline - 10:
                    break
                res = minimize_scalar(
                    lambda w: _score_params((w, best_params[1], order), n),
                    bounds=(max(1e-4, best_params[0]*0.5), min(3.0, best_params[0]*2.0)),
                    method='bounded',
                    options={'xatol': 1e-6, 'maxiter': 50}
                )
                if res.success and res.fun < best_score:
                    best_score = res.fun
                    h = _construct_h(res.x, best_params[1], order, n)
                    best_h = h.copy()
                
                res2 = minimize_scalar(
                    lambda p: _score_params((best_params[0], p % 2.0, order), n),
                    bounds=(0, 2.0),
                    method='bounded',
                    options={'xatol': 1e-6, 'maxiter': 50}
                )
                if res2.success and res2.fun < best_score:
                    best_score = res2.fun
                    h = _construct_h(best_params[0], res2.x % 2.0, order, n)
                    best_h = h.copy()
    
    # Phase 3: Final polish - try to improve with finer local search
    if time.time() < deadline - 5:
        # Try multiple random restarts in a small neighborhood
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Perturb around current best
            h_trial = best_h + rng.normal(0, 0.01, n)
            h_trial = np.clip(h_trial, 0, 1)
            # Enforce sum
            diff = n/2.0 - np.sum(h_trial)
            h_trial = h_trial + diff/n
            h_trial = np.clip(h_trial, 0, 1)
            diff2 = n/2.0 - np.sum(h_trial)
            h_trial = h_trial + diff2/n
            s = _overlap_score(h_trial)
            if s < best_score:
                best_score = s
                best_h = h_trial.copy()
    
    # Final enforcement of constraints
    best_h = np.clip(best_h, 0, 1)
    diff = n/2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        best_h = best_h + diff/n
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
