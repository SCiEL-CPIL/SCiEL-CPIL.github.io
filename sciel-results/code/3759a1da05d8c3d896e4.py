"""
Structural change: Represent step heights as a low-rank superposition of
discrete cosine basis functions (DCT-II), optimizing only the basis
coefficients and amplitudes to enforce global smoothness and reduce the
dimensionality of the search space. This structural constraint prevents
the formation of high-frequency local spikes that degrade the L2-to-L1
ratio, allowing gradient-based methods to converge to sharper optima.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

# Number of DCT basis components to use (low-rank representation)
K = 8

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _dct_encode(h):
    """Encode heights into DCT-II coefficients (first K components)."""
    # Use scipy's dct with type 2
    coeffs = opt._dct(h, type=2, norm='ortho') if hasattr(opt, '_dct') else None
    if coeffs is None:
        # Manual DCT-II implementation
        n = len(h)
        k = np.arange(n)
        m = np.arange(n)
        # DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)
        # With orthonormal normalization
        coeffs = np.zeros(n)
        for k_idx in range(n):
            coeffs[k_idx] = np.sqrt(1.0 / n) if k_idx == 0 else np.sqrt(2.0 / n)
            coeffs[k_idx] *= np.sum(h * np.cos(np.pi / n * (m + 0.5) * k_idx))
    return coeffs[:K]

def _dct_decode(coeffs):
    """Decode DCT-II coefficients back to heights."""
    n = N
    m = np.arange(n)
    h = np.zeros(n)
    for k_idx in range(len(coeffs)):
        scale = np.sqrt(1.0 / n) if k_idx == 0 else np.sqrt(2.0 / n)
        h += coeffs[k_idx] * scale * np.cos(np.pi / n * (m + 0.5) * k_idx)
    return h

def _smooth_value_from_coeffs(coeffs, beta=50.0):
    """
    Smooth surrogate for the value function, operating on DCT coefficients.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    Returns negative value for minimization.
    """
    h = _dct_decode(coeffs)
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max: (1/beta) * log(sum(exp(beta * g)))
    # Shift g to prevent overflow
    g_max = np.max(g)
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _generate_starts():
    """Generate diverse starting points in DCT coefficient space."""
    starts = []
    
    # Uniform heights -> only DC component
    h_uniform = np.ones(N)
    starts.append(_dct_encode(h_uniform))
    
    # Gaussian bump
    x = np.linspace(0, N - 1, N)
    h_gauss = np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2)
    starts.append(_dct_encode(h_gauss))
    
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(_dct_encode(p1 + p2))
    
    # Semicircle
    h_semi = np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2))
    starts.append(_dct_encode(h_semi))
    
    # Cosine bump
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    starts.append(_dct_encode(h_cos))
    
    # Triangle
    h_tri = 1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1)
    starts.append(_dct_encode(h_tri))
    
    # Random smooth profiles via DCT coefficients
    for _ in range(20):
        coeffs = np.zeros(K)
        coeffs[0] = RNG.uniform(0.5, 1.5)  # DC component
        for k in range(1, K):
            coeffs[k] = RNG.uniform(-0.5, 0.5)
        starts.append(coeffs)
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin

    best_val = 0.0
    best_h = np.ones(N)
    best_coeffs = _dct_encode(best_h)

    def track(h):
        nonlocal best_val, best_h, best_coeffs
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
            best_coeffs = _dct_encode(h)

    def track_from_coeffs(coeffs):
        nonlocal best_val, best_h, best_coeffs
        h = _dct_decode(coeffs)
        track(h)
        if compute_value(h) >= best_val - 1e-12:
            best_coeffs = coeffs.copy()

    starts = _generate_starts()
    
    # Initial evaluation
    for s in starts:
        h = _dct_decode(s)
        track(h)

    # Optimization Phase in DCT coefficient space
    # No bounds needed since DCT coefficients can be any real number,
    # but we clamp h to be non-negative after decoding
    
    betas = [20.0, 50.0, 100.0]
    
    for s in starts:
        if time.time() > deadline:
            break
        
        x0 = np.asarray(s, dtype=float).copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                res = opt.minimize(
                    lambda x: _smooth_value_from_coeffs(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 0:  # -value, so lower is better
                    track_from_coeffs(res.x)
            except Exception:
                pass

    # Polish phase: Perturb best coefficients and re-optimize with high beta
    for _ in range(30):
        if time.time() > deadline:
            break
        pert = best_coeffs + RNG.normal(0.0, 0.1, K)
        
        # Use high beta for sharpness
        try:
            res = opt.minimize(
                lambda x: _smooth_value_from_coeffs(x, 200.0),
                pert,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track_from_coeffs(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
