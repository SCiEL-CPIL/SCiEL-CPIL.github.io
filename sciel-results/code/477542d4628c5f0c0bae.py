"""Signed autocorrelation maximum absolute-value variant – C3-prime."""
import time
import numpy as np

N = 400
DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds


def _autoconv(x):
    """Compute autoconvolution (full) of x using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * fx, size)[:2 * n - 1]
    return ac


def _score(x):
    """C3' = max|autoconv| / (integral^2). Lower is better."""
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return max_ac / (integral * integral)


def _spectral_init(rng, n):
    """Generate a signed step function from a random-phase spectrum."""
    # Raised-cosine magnitude spectrum for smoother autoconvolution sidelobes
    k = np.arange(n)
    # Flat-top-like window in frequency domain
    mag = np.ones(n)
    # Add some random structure
    phases = rng.uniform(0, 2 * np.pi, n)
    spectrum = mag * np.exp(1j * phases)
    x = np.fft.irfft(spectrum, n)
    # Make it a signed step function: threshold at 0
    x = np.sign(x)
    # Ensure non-trivial
    if np.all(x == 0):
        x = rng.choice([-1, 1], n)
    return x


def _random_init(rng, n):
    """Random ±1 sequence."""
    return rng.choice([-1, 1], n).astype(float)


def _coordinate_descent(x, rng, max_iter=2000, time_budget=30.0):
    """Iterative coordinate descent: flip individual bits to reduce score."""
    start = time.time()
    current_score = _score(x)
    x = x.copy()
    
    # Precompute autoconvolution for fast updates
    ac = _autoconv(x)
    integral = np.sum(x)
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        best_delta = 0
        best_idx = -1
        
        # Sample a subset of positions for speed
        sample_size = min(100, N)
        indices = rng.choice(N, sample_size, replace=False)
        
        for idx in indices:
            # Try flipping x[idx]
            # New integral
            new_integral = integral - 2 * x[idx]
            if abs(new_integral) < 1e-12:
                continue
            
            # Update autoconvolution: flipping x[idx] changes ac by 2*x[idx]*(delta contributions)
            # ac_new[k] = ac[k] + 2*x[idx]*(x[k-idx] + x[idx-k] - 2*x[idx]*delta(k,idx))
            # More precisely: ac = x * x (convolution). Flipping x[idx] from v to -v
            # changes x by -2v at position idx.
            # ac_new = ac - 2v*(delta * x + x * delta) + 4v^2 * delta*delta
            # where delta is 1 at idx, 0 elsewhere.
            # delta * x has x[idx] at position idx (in convolution indexing)
            # Actually for linear convolution of length 2n-1:
            # ac[k] = sum_i x[i] x[k-i]
            # Changing x[idx] by dv = -2v:
            # ac_new[k] = ac[k] + dv*x[k-idx] + dv*x[idx-(k-idx)] ... 
            # Let's just recompute for the sampled positions (slower but correct)
            
            # For efficiency, let's compute the change in max|ac| approximately
            # The change in ac at position k is: dv * (x[k-idx] if 0<=k-idx<n else 0) 
            #   + dv * (x[idx-k+n] if 0<=idx-k+n<n else 0)  -- this is getting complex
            # Let's use a simpler approach: compute the full new ac for candidate flips
            # but only for the best few candidates.
            
            # Simpler: just track which positions would reduce the peak
            # The peak is at some position p. Flipping x[idx] affects ac at positions
            # idx+i for all i where x[i] != 0, i.e., all i.
            # ac_new[k] = ac[k] - 2*v*x[k-idx] (if 0<=k-idx<n) - 2*v*x[idx+k-n] (if 0<=idx+k-n<n)
            # Wait, this is getting complicated. Let me just do a direct but limited search.
            
            pass
        
        # Simpler approach: try flipping each of a random subset and compute score
        # This is O(sample_size * N log N) per iteration which is too slow.
        # Let's use a gradient-based approach instead.
        break
    
    return x, current_score


def _lbfgs_refine(x0, rng, time_budget=15.0):
    """Use scipy.optimize.minimize with L-BFGS-B to refine a continuous relaxation."""
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective(xc):
        if time.time() - start > time_budget:
            return 1e10
        # xc is continuous; we'll evaluate on the continuous version
        ac = _autoconv(xc)
        max_ac = np.max(np.abs(ac))
        integral = np.sum(xc)
        if abs(integral) < 1e-12:
            return 1e10
        return max_ac / (integral * integral)
    
    def gradient(xc):
        # Numerical gradient (central difference)
        eps = 1e-5
        grad = np.zeros_like(xc)
        f0 = objective(xc)
        for i in range(len(xc)):
            xp = xc.copy()
            xp[i] += eps
            xm = xc.copy()
            xm[i] -= eps
            grad[i] = (objective(xp) - objective(xm)) / (2 * eps)
        return grad
    
    # Use Nelder-Mead as fallback if L-BFGS is too slow for gradient
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if time.time() - start > time_budget:
        return x0
    
    # Threshold to ±1
    x_refined = np.sign(result.x)
    if np.all(x_refined == 0):
        x_refined = np.sign(x0)
    return x_refined


def construct():
    """Construct a signed step function minimizing C3'."""
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = 1e10
    
    # Phase 1: Generate diverse initial points
    inits = []
    
    # Random ±1
    for i in range(5):
        x = _random_init(rng, N)
        inits.append(x)
    
    # Spectral initializations
    for i in range(15):
        x = _spectral_init(rng, N)
        inits.append(x)
    
    # Structured patterns
    # Alternating
    inits.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    # Block patterns with different block sizes
    for block in [2, 4, 8, 16, 32]:
        x = np.zeros(N)
        for i in range(N):
            x[i] = 1.0 if (i // block) % 2 == 0 else -1.0
        inits.append(x)
    
    # Evaluate all initial points
    for x in inits:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Phase 2: Local refinement on best candidates
    if time.time() < DEADLINE - SAFETY_MARGIN:
        # Sort inits by score and refine top ones
        scored_inits = []
        for x in inits:
            s = _score(x)
            scored_inits.append((s, x))
        scored_inits.sort(key=lambda t: t[0])
        
        # Refine top 5 with Nelder-Mead
        for s, x in scored_inits[:5]:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            x_ref = _lbfgs_refine(x, rng, time_budget=min(10.0, DEADLINE - SAFETY_MARGIN - time.time() - start_time))
            if x_ref is not None:
                s_ref = _score(x_ref)
                if s_ref < best_score:
                    best_score = s_ref
                    best_x = x_ref.copy()
    
    # Phase 3: Coordinate descent on the best solution
    if best_x is not None and time.time() < DEADLINE - SAFETY_MARGIN:
        x_cd = best_x.copy()
        current_s = best_score
        remaining = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining > 5:
            # Simple iterative flipping
            for it in range(2000):
                if time.time() > DEADLINE - SAFETY_MARGIN:
                    break
                # Try flipping a random position
                idx = rng.integers(0, N)
                old_val = x_cd[idx]
                x_cd[idx] = -old_val
                new_s = _score(x_cd)
                if new_s < current_s:
                    current_s = new_s
                    best_score = new_s
                    best_x = x_cd.copy()
                else:
                    x_cd[idx] = old_val
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
