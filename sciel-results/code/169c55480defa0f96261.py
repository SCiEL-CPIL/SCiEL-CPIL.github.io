import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=30.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=30.0):
    """Gradient of the smooth objective using FFT for speed."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    # d(soft_max)/d(h[j]) = sum_k weights[k] * 2*h[k-j] for valid k
    # This is a correlation: d_soft_max[j] = 2 * sum_{k} weights[k] * h[k-j] (valid)
    # = 2 * (weights * h)[j] in the appropriate convolution sense
    
    # Use FFT to compute the correlation efficiently
    # We need: for each j, sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    # Let m = k-j, then sum_{m=0}^{N-1} weights[m+j] * h[m]
    # This is like a cross-correlation
    
    # Pad to avoid circular convolution issues
    size = 2 * N - 1
    w_padded = np.zeros(size)
    w_padded[:N] = weights[:N]  # weights has length 2N-1, but we only need first N for this
    
    # Actually, weights has length 2N-1 (same as conv)
    # For j in [0, N-1], we need weights[j:j+N] dot h[0:N]
    # This is a sliding dot product
    
    # Use FFT: correlate weights with h
    # result[j] = sum_{m} weights[m+j] * h[m] for m=0..N-1, j=0..N-1
    # This is equivalent to: flip h, convolve with weights, take appropriate slice
    
    h_flipped = h[::-1]
    # Full convolution of weights and h_flipped
    # conv_full[k] = sum_i weights[i] * h_flipped[k-i]
    # We want: for each j, sum_{m=0}^{N-1} weights[m+j] * h[m]
    # = sum_{m=0}^{N-1} weights[m+j] * h_flipped[N-1-m]
    # Let i = m+j, then = sum_{i=j}^{j+N-1} weights[i] * h_flipped[N-1-(i-j)]
    # = sum_{i=j}^{j+N-1} weights[i] * h_flipped[N-1-j+i-j]... this is getting complex
    
    # Simpler: just use the direct formula with numpy broadcasting for N=400
    # d_soft_max[j] = 2 * sum_{k=j}^{min(j+N-1, 2N-2)} weights[k] * h[k-j]
    
    d_soft_max = np.zeros(N)
    for j in range(N):
        k_start = j
        k_end = min(j + N - 1, 2 * N - 2)
        d_soft_max[j] = 2.0 * np.dot(weights[k_start:k_end+1], h[0:k_end-k_start+1])
    
    # Now compute gradient of obj = |2N * soft_max / s^2|
    # Assume 2N * soft_max / s^2 > 0 (we can handle sign separately)
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = np.zeros(N)
    # d(obj)/d(h[j]) = sign * 2N * (d_soft_max[j]/s^2 - 2*soft_max/s^3)
    grad[:] = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(2.0 * N * soft_max / (s * s))
    return grad, obj





def _smooth_objective(h, beta=30.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=30.0):
    """Gradient of the smooth objective."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    obj = abs(2.0 * N * soft_max / (s * s))
    
    # Gradient of soft_max w.r.t. h
    # soft_max = m + (1/beta) * log(sum(exp(beta*(conv_m - m))))
    # d(soft_max)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = h[k-i] + h[k+N-1-i] (with appropriate bounds)
    
    # conv[k] = sum_i h[i]*h[k-i] for valid i
    # d(conv[k])/d(h_j) = sum_{i: i + (k-i) = k} ... 
    # Actually conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
    # d(conv[k])/d(h_j) = h[k-j] + h[j-(k-N+1)] if in range... 
    
    # More carefully: conv[k] = sum_{i} h[i]*h[k-i] where both indices in [0,N-1]
    # So d(conv[k])/d(h_j) = h[k-j] (if 0<=k-j<N) + h[j-(k-(N-1))]... 
    # Let me just use: d(conv[k])/d(h_j) = h[k-j] if 0<=k-j<=N-1, plus 
    # if k-j < 0 or k-j >= N, check the other term.
    
    # Actually for full convolution: conv[k] = sum_{i=0}^{N-1} h[i] * h[k-i] where h[x]=0 if x not in [0,N-1]
    # So d(conv[k])/d(h_j) = h[k-j] if 0<=k-j<N, else 0. Plus by symmetry h[j-(k)]... 
    # No: conv[k] = sum_i h[i]*h[k-i]. d/dh_j = h[k-j] (from i=j term) + h[j] * d(h[k-i])/dh_j (from i where k-i=j, i.e., i=k-j)
    # = h[k-j] + h[j]*h[k-(k-j)]... I'm confusing myself.
    
    # Let's just compute it numerically or use the fact that:
    # conv = h * h (full convolution)
    # d(conv[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, else 0
    # Wait, that's not right either. Let me think again.
    # conv[k] = sum_{i} h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = sum_i [delta(i,j)*h[k-i] + h[i]*delta(k-i,j)]
    # = h[k-j] + h[j] (if k-j is valid) + h[k-j] (if k-j is valid)... 
    # No: first term: i=j gives h[k-j]. Second term: k-i=j gives i=k-j, so h[k-j]*h[j]... 
    # Wait no. d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N? 
    # That's only if the two terms are distinct, which they are when j != k-j, i.e., 2j != k.
    # When 2j = k, we get h[j]^2, and the derivative is 2*h[j].
    # So in all cases: d(conv[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, plus h[j] * (1 if k-j is a valid index for the second factor)... 
    
    # I think the cleanest way: conv[k] = sum_i h[i] h[k-i]
    # d/dh[j] = sum_i [delta_{ij} h[k-i] + h[i] delta_{k-i,j}]
    # = h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N) ... no.
    # First sum: i=j, term is h[k-j] (valid if 0<=k-j<N)
    # Second sum: k-i=j => i=k-j, term is h[k-j] (valid if 0<=k-j<N)
    # So d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N
    # But wait, when i=j AND k-i=j (i.e., k=2j), these are the same term h[j]*h[j], 
    # and the derivative of h[j]^2 is 2*h[j], which matches 2*h[k-j]=2*h[j]. Good.
    
    # So: d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0.
    # Hmm, but that seems off. Let me verify with a small example.
    # h = [a, b], conv = [a^2, ab, ab, b^2] (length 3 for N=2? No, full conv of length 2 is length 3)
    # Actually for N=2: conv = [a*a, a*b+b*a, b*b] = [a^2, 2ab, b^2]
    # d(conv[1])/d(a) = 2b. Formula: 2*h[1-0] = 2*h[1] = 2b. Correct.
    # d(conv[1])/d(b) = 2a. Formula: 2*h[1-1] = 2*h[0] = 2a. Correct.
    # d(conv[0])/d(a) = 2a. Formula: 2*h[0-0] = 2*h[0] = 2a. Correct.
    
    # Great! So d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0.
    
    # Now gradient of obj = 2N * soft_max / s^2 (assuming positive for now)
    # d(obj)/d(h[j]) = 2N * (d(soft_max)/d(h[j]) * s^2 - soft_max * 2*s * 1) / s^4
    # = 2N * (d(soft_max)/d(h[j]) / s^2 - 2*soft_max / s^3)
    
    # d(soft_max)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # = sum_k weights[k] * 2*h[k-j] (if 0<=k-j<N)
    # = 2 * sum_k weights[k] * h[k-j] (valid k)
    
    # Let's compute this efficiently.
    # For each j, d(soft_max)/d(h[j]) = 2 * sum_{k: 0<=k-j<N} weights[k] * h[k-j]
    # = 2 * sum_{m=0}^{N-1} weights[m+j] * h[m] (where m = k-j, so k = m+j, need 0<=m+j<2N-1)
    
    # This is a correlation-like operation.
    
    # Let's just do it with a loop for clarity, N=400 is small enough.
    d_soft_max = np.zeros(N)
    for j in range(N):
        # d(conv[k])/d(h[j]) = 2*h[k-j] for k in [j, j+N-1] (since 0<=k-j<N => j<=k<j+N)
        # k ranges from j to min(j+N-1, 2N-2)
        k_start = j
        k_end = min(j + N - 1, 2 * N - 2)
        if k_start > k_end:
            continue
        d_soft_max[j] = 2.0 * np.dot(weights[k_start:k_end+1], h[0:k_end-k_start+1])
    
    # Now compute gradient of obj
    # obj = 2N * soft_max / s^2 (we'll handle the abs by assuming sign)
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = np.zeros(N)
    for j in range(N):
        # d(obj)/d(h[j]) = sign * 2N * (d_soft_max[j]/s^2 - 2*soft_max/s^3)
        grad[j] = sign * 2.0 * N * (d_soft_max[j] / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return grad, obj


def _eigen_init():
    """Initialize with diverse structured patterns."""
    x = np.arange(N)
    candidates = []
    
    # Uniform (baseline)
    candidates.append(np.ones(N) / N)
    
    # Multi-frequency cosine combinations
    freqs = [1, 2, 3, 4, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50, 75, 100]
    for k in freqs:
        for amp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                h = h / s
            candidates.append(h)
    
    # Two-frequency products
    for k1 in [1, 2, 3, 5, 7, 10, 15, 20]:
        for k2 in [11, 13, 17, 20, 25, 30, 40, 50]:
            for amp in [0.3, 0.5, 1.0, 1.5, 2.0]:
                h = np.ones(N) / N + amp * (np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N))
                s = np.sum(h)
                if abs(s) > 1e-12:
                    h = h / s
                candidates.append(h)
    
    # Three-frequency combinations
    for k1 in [1, 2, 3, 5]:
        for k2 in [7, 10, 13, 17]:
            for k3 in [20, 25, 30, 40]:
                h = np.ones(N) / N
                h += 0.5 * np.cos(2*np.pi*k1*x/N)
                h += 0.3 * np.cos(2*np.pi*k2*x/N)
                h += 0.2 * np.cos(2*np.pi*k3*x/N)
                s = np.sum(h)
                if abs(s) > 1e-12:
                    h = h / s
                candidates.append(h)
    
    # Polynomial bases
    t = x / (N - 1) - 0.5
    for deg in [2, 3, 4, 5, 6, 8]:
        for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) / N + amp * t ** deg
            s = np.sum(h)
            if abs(s) > 1e-12:
                h = h / s
            candidates.append(h)
    
    # Gaussian mixtures
    for n_gauss in [2, 3, 4]:
        for sigma in [10, 20, 30, 50]:
            for centers in [[N//4, 3*N//4], [N//6, N//2, 5*N//6], [N//8, 3*N//8, 5*N//8, 7*N//8]]:
                if len(centers) != n_gauss:
                    continue
                h = np.ones(N) / N
                for c in centers:
                    h += np.exp(-0.5 * ((x - c) / sigma) ** 2)
                s = np.sum(h)
                if abs(s) > 1e-12:
                    h = h / s
                candidates.append(h)
    
    # Find best candidate
    best_h = None
    best_val = 1e18
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective with multiple beta values."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Try multiple beta values for the smooth surrogate
    for beta in [20.0, 40.0, 80.0, 150.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start_local(h_best, n_starts=8, max_time=40.0):
    """Multi-start local refinement with diverse initializations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        # Generate diverse perturbation with varying scales
        scale = 0.05 + 0.08 * trial
        perturbation = rng.normal(0, scale, N)
        # Keep sum approximately the same
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s  # normalize to sum=1
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=40.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=40.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _structured_search(max_time=20.0):
    """Search over structured patterns more thoroughly."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    x = np.arange(N)
    
    # Single frequency with phase
    for k in range(1, 80):
        if time.time() - t0 > max_time * 0.4:
            break
        for amp in np.linspace(0.2, 4.0, 15):
            for phase in [0, np.pi/4, np.pi/2, np.pi]:
                h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N + phase)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                h = h / s
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Two frequency products
    if time.time() - t0 < max_time * 0.6:
        for k1 in range(1, 25):
            for k2 in range(k1 + 1, min(k1 + 15, 60)):
                if time.time() - t0 > max_time * 0.75:
                    break
                for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                    h = np.ones(N) / N + amp * np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    h = h / s
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    # Polynomial bases
    if time.time() - t0 < max_time * 0.8:
        t = x / (N - 1) - 0.5
        for deg in [2, 3, 4, 5, 6, 8]:
            for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                h = np.ones(N) / N + amp * t ** deg
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                h = h / s
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Windowed patterns (Gaussian)
    if time.time() - t0 < max_time * 0.9:
        for sigma in [10, 20, 30, 50, 80]:
            for center in [N//4, N//2, 3*N//4]:
                for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    h = np.ones(N) / N + amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    h = h / s
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Structured search for good initial candidates
    h_struct, val_struct = _structured_search(max_time=12.0)
    
    # Phase 2: Eigen-based initialization
    h_eigen, val_eigen = _eigen_init()
    
    # Pick best from both
    if val_struct <= val_eigen:
        best_h = h_struct.copy()
        best_val = val_struct
    else:
        best_h = h_eigen.copy()
        best_val = val_eigen
    
    # Also include the seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 3: Local refinement from best candidate
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(40.0, time_left() - 15.0))
    
    # Phase 4: Multi-start local refinement
    if time_left() > 30:
        best_h, best_val = _multi_start_local(best_h, n_starts=10, max_time=min(45.0, time_left() - 15.0))
    
    # Phase 5: Another round of structured search with better parameters
    if time_left() > 20:
        h_struct2, val_struct2 = _structured_search(max_time=min(15.0, time_left() - 8.0))
        if val_struct2 < best_val:
            best_val = val_struct2
            best_h = h_struct2.copy()
            # Quick local refine
            if time_left() > 12:
                best_h, best_val = _local_refine(best_h, max_time=min(12.0, time_left() - 5.0))
    
    # Phase 6: Final high-precision polish with very high beta
    if time_left() > 10:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=200.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=200.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 800, 'maxfun': 2000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 7: Nelder-Mead polish if time permits
    if time_left() > 6:
        try:
            res = minimize(
                _value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
