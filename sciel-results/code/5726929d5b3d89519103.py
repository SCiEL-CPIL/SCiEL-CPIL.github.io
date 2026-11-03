"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Model the step height sequence as the squared modulus of a single complex exponential with a variable phase (a pure tone) plus a small, structured noise term, parameterized by frequency and amplitude, which fundamentally changes the search space from an N-dimensional hypercube to a 2-3 dimensional manifold that naturally captures the smooth, low-autocorrelation structure required to minimize maximum overlap.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _project(h):
    """
    Project h onto the set {x : sum x = N/2, 0 <= x <= 1}.
    This is the projection onto the intersection of a hyperplane and a box.
    We can solve this using a binary search on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    target = N / 2.0
    
    # If sum is already close and within bounds, return
    if np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - target) < 1e-9:
        return h
    
    # Binary search for lambda such that sum(clip(h - lambda, 0, 1)) = target
    # f(lambda) = sum(clip(h - lambda, 0, 1)) is monotonically decreasing in lambda.
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
            
    proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    # Fine tune to ensure exact sum
    diff = target - np.sum(proj)
    if abs(diff) > 1e-12:
        mask = (proj > 0) & (proj < 1)
        if np.any(mask):
            proj[mask] += diff / np.sum(mask)
            proj = np.clip(proj, 0.0, 1.0)
    return proj


def _construct_from_params(freq, amp, phase, noise_freq, noise_amp, noise_phase):
    """
    Construct h from parameters.
    h_i = (1 + amp * cos(2*pi*freq*i/N + phase))^2 / 4 + noise_term
    But we need h in [0,1] and sum N/2.
    Let's use: h_i = 0.5 + 0.5 * amp * cos(2*pi*freq*i/N + phase) + noise_amp * cos(2*pi*noise_freq*i/N + noise_phase)
    Then fix sum and clip.
    """
    i = np.arange(N)
    base = 0.5 + 0.5 * amp * np.cos(2.0 * np.pi * freq * i / N + phase)
    noise = noise_amp * np.cos(2.0 * np.pi * noise_freq * i / N + noise_phase)
    h = base + noise
    return _fix_sum(h)


def _nesterov_pgd(h0, T, max_iter=1000, step_size=0.01):
    """
    Minimize V(h) = sum_k max(0, C_k(h) - T) using Nesterov Accelerated PGD.
    Returns (h, min_violation).
    """
    h = h0.copy()
    h_prev = h0.copy()
    t_k = 1.0
    
    for i in range(max_iter):
        t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        y = h + (t_k - 1) / t_new * (h - h_prev)
        
        # Compute gradient of V(y)
        cs = _linear_overlap_values(y)
        violation = np.maximum(0, cs - T)
        active = (cs > T)
        
        # Gradient of C_k
        # dC_k/dh_j = -(2/N) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<N) )
        # We only need gradient for active k
        grad = np.zeros(N)
        if np.any(active):
            # Vectorized gradient for active constraints
            # For each active k, add contribution to grad
            # This loop is O(N * #active). #active can be up to N. O(N^2).
            # For N=100, 10000 ops per iteration. 1000 iterations -> 10M ops. Fast.
            for k in np.where(active)[0]:
                wk = 1.0 # weight for max(0, .) is 1 if active
                factor = -(2.0 / N) * wk
                if k > 0:
                    grad[k:] += factor * y[:N-k]
                if k < N:
                    grad[:N-k] += factor * y[k:]
        
        # Projected Gradient Step
        h_new = _project(y - step_size * grad)
        
        h_prev = h
        h = h_new
        t_k = t_new
        
        # Early stopping if violation is very small
        v = np.sum(violation)
        if v < 1e-6:
            break
            
    return h, float(np.sum(np.maximum(0, _linear_overlap_values(h) - T)))


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Initial guess: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Parameter space search
    # freq in [0.1, 5.0], amp in [0.0, 1.0], phase in [0, 2*pi]
    # noise_freq in [1.0, 10.0], noise_amp in [0.0, 0.1], noise_phase in [0, 2*pi]
    
    # Grid search over freq and amp first, then refine with PGD
    freqs = np.linspace(0.5, 5.0, 20)
    amps = np.linspace(0.0, 1.0, 10)
    
    for freq in freqs:
        for amp in amps:
            if time.time() - t0 > _DEADLINE - 10:
                break
                
            # Try a few phases
            for phase in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                h = _construct_from_params(freq, amp, phase, 0.0, 0.0, 0.0)
                _update_best(h)
                
                # If this looks promising, refine with PGD
                if best_score < 1.2:
                    h_res, _ = _nesterov_pgd(h, best_score, max_iter=200, step_size=0.05)
                    _update_best(h_res)
    
    # Now do a more detailed search around the best parameters found
    if best_h is not None:
        # Extract approximate parameters from best_h (roughly)
        # For now, just do local random perturbations in parameter space
        best_freq = 1.0  # Placeholder
        best_amp = 0.5   # Placeholder
        
        for _ in range(50):
            if time.time() - t0 > _DEADLINE - 10:
                break
                
            # Random perturbation around best parameters
            freq = best_freq + np.random.uniform(-0.5, 0.5)
            amp = best_amp + np.random.uniform(-0.2, 0.2)
            phase = np.random.uniform(0, 2*np.pi)
            noise_freq = np.random.uniform(1.0, 5.0)
            noise_amp = np.random.uniform(0.0, 0.05)
            noise_phase = np.random.uniform(0, 2*np.pi)
            
            h = _construct_from_params(freq, amp, phase, noise_freq, noise_amp, noise_phase)
            _update_best(h)
            
            # Refine with PGD if improved
            if best_score < 1.15:
                h_res, _ = _nesterov_pgd(h, best_score, max_iter=300, step_size=0.03)
                _update_best(h_res)
    
    # Final polish with SLSQP if time permits, using the best h found by PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        from scipy.optimize import minimize
        
        def _smooth_max_grad(h, beta):
            h = np.asarray(h, dtype=np.float64)
            n = len(h)
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            val = m + np.log(sum_exp) / beta
            grad = np.zeros(n)
            for k in range(n):
                wk = w[k]
                if wk == 0.0:
                    continue
                factor = -(2.0 / n) * wk
                if k > 0:
                    grad[k:] += factor * h[:n-k]
                if k < n:
                    grad[:n-k] += factor * h[k:]
            return val, grad

        def obj_func(x):
            val, _ = _smooth_max_grad(x, 50.0)
            return val
        def jac_func(x):
            _, grad = _smooth_max_grad(x, 50.0)
            return grad

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        try:
            res = minimize(obj_func, best_h.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                _update_best(res.x)
        except:
            pass

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
