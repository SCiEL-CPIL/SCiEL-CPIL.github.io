import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 150s kill

def _supnorm_grid(coeffs, n_grid=2048):
    """Fast sup-norm estimate using FFT on a grid."""
    padded = np.zeros(n_grid, dtype=complex)
    padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _refine_peak(coeffs, ang, iters=80):
    """Golden-section refine max of |P(e^{i theta})| near ang."""
    n = len(coeffs)
    ks = np.arange(n)
    
    def f(theta):
        z = np.exp(1j * theta)
        return float(np.abs(np.dot(coeffs, z ** ks)))
    
    # Start with a small window around ang
    a, b = ang - 0.05, ang + 0.05
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    
    return max(f((a + b) / 2), fc, fd)

def _get_top_peaks(coeffs, n_grid=4096, n_peaks=20):
    """Find the angles of the top n_peaks maxima."""
    padded = np.zeros(n_grid, dtype=complex)
    padded[:len(coeffs)] = coeffs
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    idx = np.argsort(mags)[-n_peaks:][::-1]
    
    angles = []
    values = []
    for i in idx:
        ang = 2 * np.pi * i / n_grid
        v = _refine_peak(coeffs, ang, iters=60)
        angles.append(ang)
        values.append(v)
        
    return np.array(angles), np.array(values)

def _supnorm_precise(coeffs, n_grid=4096, n_peaks=20):
    """Sup-norm with golden-section refinement of top peaks."""
    angles, values = _get_top_peaks(coeffs, n_grid, n_peaks)
    return float(np.max(values))

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def construct():
    n = 24
    
    # Generate candidate seeds
    gr = np.array(_gr_sequence(n), dtype=float)
    seeds = [
        gr,
        -gr,
        np.ones(n),
        -np.ones(n),
        np.array([1 if (k % 4 in (0, 1)) else -1 for k in range(n)], dtype=float),
        np.array([1 if (k % 3 in (0, 1)) else -1 for k in range(n)], dtype=float),
    ]
    
    # Add some random seeds
    rng = np.random.default_rng(42)
    for _ in range(20):
        seeds.append(rng.choice([-1, 1], size=n).astype(float))
    
    best = None
    best_val = float('inf')
    
    def try_vec(v):
        nonlocal best, best_val
        val = _supnorm_grid(v, n_grid=1024)
        if val < best_val * 1.01:  # Only refine if promising
            valr = _supnorm_precise(v)
            if valr < best_val:
                best_val = valr
                best = v.copy()
    
    for s in seeds:
        try_vec(s)
    
    if best is None:
        best = seeds[0]
        best_val = _supnorm_precise(best)
        
    # Local search: Simulated Annealing with Critical Angles
    cur = best.copy()
    cur_val = best_val
    
    # Initialize critical angles
    cur_angles, cur_values = _get_top_peaks(cur, n_grid=4096, n_peaks=20)
    cur_max_val = float(np.max(cur_values))
    
    T = 0.5
    n_iter = 0
    last_refine_time = time.time()
    ks = np.arange(n)
    
    while time.time() < _DEADLINE:
        n_iter += 1
        if n_iter % 100 == 0:
            T *= 0.999
            
        # Pick a random index to flip
        k = rng.integers(0, n)
        cand = cur.copy()
        cand[k] *= -1
        
        # Evaluate candidate at current critical angles
        # Vectorized evaluation
        zs = np.exp(1j * cur_angles)
        powers = zs[:, None] ** ks
        vals = powers @ cand
        max_val = float(np.max(np.abs(vals)))
        
        # Acceptance criterion
        if max_val < cur_max_val:
            # Accept improvement
            cur = cand
            cur_val = max_val
            
            # Periodically do a full refinement to update global best and angles
            if time.time() - last_refine_time > 0.5:
                new_angles, new_values = _get_top_peaks(cur, n_grid=4096, n_peaks=20)
                new_max = float(np.max(new_values))
                
                if new_max < best_val:
                    best_val = new_max
                    best = cur.copy()
                    
                # Update critical angles for the search
                cur_angles = new_angles
                cur_values = new_values
                cur_max_val = new_max
                last_refine_time = time.time()
        else:
            # Accept with probability exp(-(max_val - cur_max_val)/T)
            if T > 1e-6:
                delta = max_val - cur_max_val
                if rng.random() < np.exp(-delta / T):
                    cur = cand
                    cur_val = max_val
                    # Update angles occasionally for worsening moves
                    if n_iter % 10 == 0:
                         new_angles, new_values = _get_top_peaks(cur, n_grid=2048, n_peaks=10)
                         cur_angles = new_angles
                         cur_values = new_values
                         cur_max_val = float(np.max(new_values))
    
    # Final refinement
    if best is not None:
        final_val = _supnorm_precise(best)
        if final_val < best_val:
            best_val = final_val
    
    return [int(x) for x in best]
