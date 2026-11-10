# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv

def _precompute_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics at quadrature points."""
    k = 10
    
    # 70-point Gauss-Legendre quadrature in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl in [-1, 1], these are cos(theta) values
    cos_theta = x_gl  # shape (70,)
    
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)  # shape (130,)
    
    # Create meshgrid
    CT, PH = np.meshgrid(cos_theta, phi, indexing='ij')  # shape (70, 130)
    
    # Surface measure weights: dOmega = sin(theta) dtheta dphi
    # Since we're integrating over cos_theta = x, dtheta = -dx/sin(theta)
    # So dOmega = sin(theta) * (1/sin(theta)) dx dphi = dx dphi
    # Wait, let me reconsider. The surface element on unit sphere is dOmega = sin(theta) dtheta dphi
    # We're using Gauss-Legendre in x = cos(theta), so dx = -sin(theta) dtheta
    # Thus dOmega = sin(theta) dtheta dphi = -dx dphi
    # The quadrature weight for the x-integral already accounts for this.
    # Actually, with leggauss, we integrate f(x) dx where x = cos(theta).
    # The surface integral is ∫∫ f(cos(theta), phi) sin(theta) dtheta dphi
    # = ∫∫ f(x, phi) dx dphi (since sin(theta) dtheta = -dx, and limits flip)
    # So the weights are just w_gl * (2*pi/n_phi) for each (i,j)
    
    # Total surface area = 4*pi
    # ∫ dOmega = ∫_{-1}^{1} dx ∫_0^{2π} dphi = 2 * 2π = 4π ✓
    
    # Weight for each quadrature point
    w_theta = w_gl  # shape (70,)
    w_phi = 2 * np.pi / n_phi  # scalar
    
    # Full weight matrix
    W = np.outer(w_theta, np.full(n_phi, w_phi))  # shape (70, 130)
    
    # Compute normalization constants N_m
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # For m >= 0: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # For m < 0: use |m|
    
    def N_m(m):
        m_abs = abs(m)
        # (10 - m_abs)! / (10 + m_abs)!
        # Compute using log to avoid overflow
        log_num = np.sum(np.log(np.arange(1, 11 - m_abs + 1))) if m_abs < 10 else 0.0
        log_den = np.sum(np.log(np.arange(11 - m_abs, 11 + m_abs + 1))) if m_abs > 0 else 0.0
        # Actually: (10-m)! / (10+m)! = 1 / [(10-m+1)*(10-m+2)*...*(10+m)]
        if m_abs == 0:
            ratio = 1.0
        else:
            # (10-m)! / (10+m)! = 1 / prod_{j=11-m}^{10+m} j
            start = 11 - m_abs
            end = 10 + m_abs
            prod = np.prod(np.arange(start, end + 1))
            ratio = 1.0 / prod
        return np.sqrt(21.0 / (4.0 * np.pi) * ratio)
    
    # Precompute P_10^m(cos_theta) with Condon-Shortley phase
    # lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase (-1)^m
    
    # For each m from -10 to 10, compute Y_m at all points
    # Y_0 = N_0 * P_10^0(cos_theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos_theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^{|m|}(cos_theta) * sin(|m|*phi) for m < 0
    
    Y = np.zeros((21, 70, 130), dtype=np.float64)
    
    for idx, m in enumerate(range(-10, 11)):
        m_abs = abs(m)
        Nm = N_m(m)
        
        # Compute P_10^{m_abs}(cos_theta)
        P = lpmv(m_abs, 10, cos_theta)  # shape (70,)
        
        if m == 0:
            Y[idx] = Nm * np.outer(P, np.ones(n_phi))
        elif m > 0:
            cos_part = np.cos(m * phi)  # shape (130,)
            Y[idx] = np.sqrt(2.0) * Nm * np.outer(P, cos_part)
        else:  # m < 0
            sin_part = np.sin(m_abs * phi)  # shape (130,)
            Y[idx] = np.sqrt(2.0) * Nm * np.outer(P, sin_part)
    
    return Y, W

def _compute_ratio(coeffs, Y, W):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics."""
    # f = sum_m c_m * Y_m
    # Y has shape (21, 70, 130), coeffs has shape (21,)
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (70, 130)
    
    # ||f||_2^2 = ∫ f^2 dOmega = sum W * f^2
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    
    # ||f||_4^4 = ∫ f^4 dOmega = sum W * f^4
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.power(norm4_4, 0.25)
    
    return norm4 / norm2

def _objective(coeffs, Y, W):
    """Objective function for minimization."""
    return _compute_ratio(coeffs, Y, W)

def _gradient(coeffs, Y, W):
    """Compute gradient of R with respect to coeffs."""
    # R = ||f||_4 / ||f||_2
    # dR/dc = (1/||f||_2) * d||f||_4/dc - (||f||_4/||f||_2^2) * d||f||_2/dc
    
    # Simpler: use numerical gradient or analytical
    # Let's use the chain rule properly
    # ||f||_2 = (sum W f^2)^{1/2}
    # ||f||_4 = (sum W f^4)^{1/4}
    
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(W * f2)
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.power(norm4_4, 0.25)
    
    # d||f||_2/dc = (1/(2*norm2)) * 2 * sum W * f * df/dc
    # df/dc_i = Y_i
    # d||f||_2/dc_i = (1/norm2) * sum W * f * Y_i
    
    # d||f||_4/dc = (1/4) * (norm4_4)^{-3/4} * 4 * sum W * f^3 * df/dc
    # d||f||_4/dc_i = norm4^{-3} * sum W * f^3 * Y_i
    
    # dR/dc_i = (1/norm2) * norm4^{-3} * sum W * f^3 * Y_i - (norm4/norm2^2) * (1/norm2) * sum W * f * Y_i
    # = norm4^{-3} / norm2 * sum W * f^3 * Y_i - norm4 / norm2^3 * sum W * f * Y_i
    
    # Let's compute this efficiently
    # sum W * f^3 * Y_i for each i: this is Y^T * (W * f^3)
    Wf3 = W * f * f2  # W * f^3
    Wf = W * f
    
    grad = np.zeros(21)
    for i in range(21):
        grad[i] = (np.power(norm4_4, -0.75) / norm2) * np.sum(Wf3 * Y[i]) \
                - (norm4 / (norm2 * norm2 * norm2)) * np.sum(Wf * Y[i])
    
    return grad

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_{L^4} / ||f||_{L^2} on the unit sphere.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients c_m,
            ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    start_time = time.time()
    deadline = 630.0  # seconds, with safety margin
    
    # Precompute harmonics and weights
    Y, W = _precompute_harmonics()
    
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    # Baseline: all ones
    baseline_coeffs = np.ones(n_coeffs, dtype=np.float64)
    baseline_ratio = _compute_ratio(baseline_coeffs, Y, W)
    
    best_coeffs = baseline_coeffs.copy()
    best_ratio = baseline_ratio
    
    # Helper to check if we should stop
    def should_stop():
        return (time.time() - start_time) > (deadline - 30.0)  # 30s safety margin
    
    # Define diverse initializations
    inits = []
    
    # 1. All ones (baseline)
    inits.append(baseline_coeffs.copy())
    
    # 2. Single modes
    for m in range(-10, 11):
        c = np.zeros(n_coeffs)
        c[m + 10] = 1.0
        inits.append(c)
    
    # 3. Pairs (m, -m) with various weightings
    for m in range(1, 11):
        for ratio_w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = np.zeros(n_coeffs)
            c[m + 10] = ratio_w
            c[-m + 10] = 1.0 - ratio_w
            inits.append(c)
    
    # 4. Symmetric cos combinations (only positive m with cos)
    for m in range(1, 11):
        c = np.zeros(n_coeffs)
        c[m + 10] = 1.0
        c[-m + 10] = 1.0
        inits.append(c)
    
    # 5. Alternating signs
    c = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        c[i] = 1.0 if (i % 2 == 0) else -1.0
    inits.append(c)
    
    # 6. Gaussian-like distribution over m
    m_vals = np.arange(-10, 11)
    sigma = 3.0
    c = np.exp(-m_vals**2 / (2 * sigma**2))
    inits.append(c)
    
    # 7. Uniform over even m only
    c = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        if (i - 10) % 2 == 0:
            c[i] = 1.0
    inits.append(c)
    
    # 8. Uniform over odd m only
    c = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        if (i - 10) % 2 != 0:
            c[i] = 1.0
    inits.append(c)
    
    # 9. High-frequency emphasis
    c = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        m = abs(i - 10)
        c[i] = m / 10.0
    inits.append(c)
    
    # 10. Low-frequency emphasis
    c = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        m = abs(i - 10)
        c[i] = 1.0 - m / 10.0
    inits.append(c)
    
    # 11. Random initializations
    np.random.seed(42)
    for _ in range(20):
        c = np.random.randn(n_coeffs)
        inits.append(c)
    
    # 12. Structured: alternating cos/sin weights
    for scale in [0.5, 1.0, 2.0]:
        c = np.zeros(n_coeffs)
        for m in range(1, 11):
            c[m + 10] = scale
            c[-m + 10] = scale * 0.5
        inits.append(c)
    
    # Evaluate all initializations and pick top 10
    init_ratios = []
    for i, c in enumerate(inits):
        r = _compute_ratio(c, Y, W)
        init_ratios.append((r, i, c))
    
    init_ratios.sort(key=lambda x: x[0])
    
    # Track best so far
    for r, idx, c in init_ratios[:10]:
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    # Optimize from top 10 initializations
    top_inits = [c for r, idx, c in init_ratios[:10]]
    
    for init_idx, c0 in enumerate(top_inits):
        if should_stop():
            break
        
        try:
            result = minimize(
                _objective,
                c0,
                args=(Y, W),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Additional random restarts if time permits
    np.random.seed(123)
    for _ in range(15):
        if should_stop():
            break
        
        c0 = np.random.randn(n_coeffs)
        # Scale to have similar magnitude as best
        c0 = c0 / np.linalg.norm(c0) * np.linalg.norm(best_coeffs)
        
        try:
            result = minimize(
                _objective,
                c0,
                args=(Y, W),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Validate the result
    c = best_coeffs
    
    # Check size
    if len(c) != 21:
        c = baseline_coeffs
    
    # Check for NaN or inf
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        c = baseline_coeffs
    
    # Check max|c| constraints
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale
        if max_abs > 0:
            c = c / max_abs * 1.0
        else:
            c = baseline_coeffs
    
    return c.astype(np.float64)

# EVOLVE-BLOCK-END
