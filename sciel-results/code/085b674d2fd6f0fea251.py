import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import eigh_tridiagonal

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _build_conv_matrix():
    """Build the N x N matrix C such that conv(h,h)[k] = h^T C_k h for each lag k.
    
    For full convolution of length 2N-1:
    conv(h,h)[k] = sum_i h[i]*h[i+k-N+1] for appropriate ranges.
    
    More precisely, conv(h,h)[m] = sum_{i} h[i] * h[m-i] where valid indices.
    For m = 0..2N-2:
      conv[m] = sum_{i=max(0,m-N+1)}^{min(N-1,m)} h[i]*h[m-i]
    
    We can write conv[m] = h^T Q_m h where Q_m is a symmetric matrix.
    """
    # We'll build the matrix for the maximum lag.
    # Actually, let's think differently.
    # conv(h,h) has 2N-1 entries. Each entry is a quadratic form in h.
    # Let's build a 3D tensor or just work with the fact that:
    # conv(h,h)[m] = sum_{i,j: i+j=m} h[i]*h[j]  (with i,j in [0,N-1])
    
    # For the optimization, we need to handle max over m of these quadratic forms.
    # Let's use a different approach: parameterize h and use the structure.
    
    # Actually, for the trust-region approach, let's define:
    # f(h) = max_m (h^T Q_m h)
    # constraint: (sum h)^2 = 1, i.e., 1^T h = 1
    # minimize: ||h||^2
    
    # Q_m[i,j] = 1 if i+j = m, else 0 (for i,j in [0,N-1])
    # This is a Hankel matrix structure.
    
    # For efficiency, let's not build all Q_m explicitly.
    # Instead, note that conv(h,h) can be computed efficiently.
    pass
    return None


def _trust_region_solve(A, b, c, d, max_iter=50):
    """Solve min h^T A h + b^T h + c s.t. h^T A h + b^T h + c <= d, 1^T h = 1.
    
    Actually, let's use a cleaner formulation.
    Minimize ||h||^2 subject to max_m conv(h,h)[m] <= t, sum(h) = 1.
    
    For a given t, this is: min ||h||^2 s.t. h^T Q_m h <= t for all m, 1^T h = 1.
    This is a semidefinite program in disguise, but for small N we can use bisection on t.
    
    Alternatively, use the KKT conditions with a single active constraint.
    """
    pass


def _qp_solve_direct(h0):
    """Try to solve the QP: min ||h||^2 s.t. sum(h)=1, max(conv(h,h)) <= t.
    
    Use bisection on t and solve the resulting QP at each step.
    For a fixed t, the QP is: min h^T h s.t. 1^T h = 1, h^T Q_m h <= t for all m.
    
    We can solve this using scipy's QP solver or by reducing to a trust region problem.
    
    Actually, let's use a simpler approach: 
    The problem min max_m (h^T Q_m h) / (1^T h)^2 is equivalent to 
    finding the smallest t such that min_{h: 1^T h = 1} max_m (h^T Q_m h) <= t.
    
    For a fixed h direction, the optimal scaling is determined by the constraint.
    Let's parameterize h = v + alpha * u where 1^T v = 1 and 1^T u = 0.
    """
    pass


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
    return abs(2.0 * N * soft_max / (s * s))


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
    """Initialize using eigendecomposition of the autoconvolution structure.
    
    The key insight: conv(h,h)[k] for k near N-1 (the peak) involves the 
    "center" of the convolution. The maximum of the autoconvolution is 
    typically at or near the center (k=N-1).
    
    For k=N-1: conv[N-1] = sum_{i=0}^{N-1} h[i]*h[N-1-i]
    This is the "anti-diagonal" sum.
    
    If we want to minimize the max autoconvolution while keeping sum(h)=1,
    we want h to be spread out. The optimal structure for minimizing the 
    peak of the autoconvolution is related to the eigenvectors of the 
    Hankel matrix.
    """
    # Build the Hankel matrix H where H[i,j] = 1 if i+j = N-1, else 0
    # This corresponds to the central autoconvolution term.
    # Actually, let's build the matrix for conv[N-1]:
    # conv[N-1] = sum_i h[i]*h[N-1-i]
    # This is h^T H h where H[i,j] = 1 if i+j=N-1
    
    # But we need to minimize max over ALL k, not just k=N-1.
    # Let's use a different approach: random initialization with structure.
    
    # A good initial guess: h proportional to the inverse of the 
    # autoconvolution peak. Start with a spread-out distribution.
    
    # Try: h[i] = 1/N + epsilon * cos(2*pi*i/N) type perturbations
    # Or: minimize ||h||^2 subject to sum(h)=1 gives h = 1/N * ones(N)
    # But that gives value = 2.0 (the seed).
    
    # To beat 2.0, we need to make the autoconvolution peak smaller 
    # relative to the sum squared.
    
    # One approach: make h have zero mean in a weighted sense.
    # If sum(h) = 1 but h has large variance, the autoconvolution 
    # can have smaller peak.
    
    # Let's try: h = (1/N)*ones + alpha * (v - mean(v)) where v is 
    # an eigenvector of some structured matrix.
    
    # For now, use a simple structured initialization.
    x = np.arange(N)
    # Try several structured candidates
    candidates = []
    
    # Uniform (baseline)
    candidates.append(np.ones(N) / N)
    
    # Cosine modes
    for k in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N)
            # Normalize to sum=1
            s = np.sum(h)
            if abs(s) > 1e-12:
                h = h / s
            candidates.append(h)
    
    # Sine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        for amp in [0.5, 1.0, 1.5, 2.0]:
            h = np.ones(N) / N + amp * np.sin(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                h = h / s
            candidates.append(h)
    
    # Quadratic
    for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
        t = x / (N - 1) - 0.5
        h = np.ones(N) / N + amp * t * t
        s = np.sum(h)
        if abs(s) > 1e-12:
            h = h / s
        candidates.append(h)
    
    # Bilinear (two-frequency)
    for k1 in [1, 2, 3, 5, 7, 10]:
        for k2 in [11, 13, 17, 20, 25, 30, 40]:
            for amp in [0.5, 1.0, 1.5]:
                h = np.ones(N) / N + amp * (np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N))
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
    """Local refinement using smooth objective."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # L-BFGS-B on smooth objective
    try:
        def obj_fn(h):
            return _smooth_objective(h, beta=30.0)
        
        def grad_fn(h):
            g, _ = _smooth_grad(h, beta=30.0)
            return g
        
        res = minimize(
            obj_fn,
            best_h,
            jac=grad_fn,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = _value(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 > max_time * 0.7:
        return best_h, best_val
    
    # Try higher beta for sharper max approximation
    try:
        def obj_fn2(h):
            return _smooth_objective(h, beta=60.0)
        
        def grad_fn2(h):
            g, _ = _smooth_grad(h, beta=60.0)
            return g
        
        res = minimize(
            obj_fn2,
            best_h,
            jac=grad_fn2,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = _value(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass
    
    return best_h, best_val


def _multi_start_local(h_best, n_starts=6, max_time=40.0):
    """Multi-start local refinement with diverse initializations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.8:
            break
        
        # Generate diverse perturbation
        scale = 0.1 + 0.1 * trial
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
                lambda h: _smooth_objective(h, beta=30.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=30.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
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
    for k in range(1, 100):
        if time.time() - t0 > max_time * 0.5:
            break
        for amp in np.linspace(0.1, 5.0, 20):
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
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
    if time.time() - t0 < max_time * 0.7:
        for k1 in range(1, 30):
            for k2 in range(k1 + 1, k1 + 20):
                if time.time() - t0 > max_time * 0.85:
                    break
                for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
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
    if time.time() - t0 < max_time * 0.9:
        t = x / (N - 1) - 0.5
        for deg in [2, 3, 4, 5, 6]:
            for amp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                h = np.ones(N) / N + amp * t ** deg
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                h = h / s
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Windowed patterns (Gaussian, triangular)
    if time.time() - t0 < max_time * 0.95:
        for sigma in [5, 10, 20, 30, 50, 80, 100]:
            for center in [0, N//4, N//2, 3*N//4, N-1]:
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
    h_struct, val_struct = _structured_search(max_time=15.0)
    
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
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(30.0, time_left() - 10.0))
    
    # Phase 4: Multi-start local refinement
    if time_left() > 25:
        best_h, best_val = _multi_start_local(best_h, n_starts=8, max_time=min(35.0, time_left() - 10.0))
    
    # Phase 5: Another round of structured search with better parameters
    if time_left() > 15:
        h_struct2, val_struct2 = _structured_search(max_time=min(15.0, time_left() - 5.0))
        if val_struct2 < best_val:
            best_val = val_struct2
            best_h = h_struct2.copy()
            # Quick local refine
            if time_left() > 10:
                best_h, best_val = _local_refine(best_h, max_time=min(10.0, time_left() - 3.0))
    
    # Phase 6: Final high-precision polish
    if time_left() > 8:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=80.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=80.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 7: Nelder-Mead polish if time permits
    if time_left() > 5:
        try:
            res = minimize(
                _value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
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
