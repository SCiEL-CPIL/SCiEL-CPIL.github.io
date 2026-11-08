"""
Low-rank DCT-based optimization for the second autocorrelation inequality.

The step heights are represented as a superposition of a small number of
discrete cosine basis functions with non-negative coefficients. This structural
constraint enforces global smoothness, preventing high-frequency spikes that
degrade the L2-to-L1 ratio, and reduces the effective dimensionality of the
search space, allowing gradient-based methods to converge to sharper optima.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

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

def _dct_basis(n_basis):
    """
    Precompute DCT-II basis vectors for N points.
    Returns an array of shape (n_basis, N).
    """
    x = np.arange(N)
    basis = np.zeros((n_basis, N))
    for k in range(n_basis):
        if k == 0:
            basis[k] = 1.0 / np.sqrt(N)
        else:
            basis[k] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2 * x + 1) / (2 * N))
    return basis

def _heights_from_coeffs(coeffs, basis):
    """
    Reconstruct heights from non-negative DCT coefficients.
    coeffs: shape (n_basis,)
    basis: shape (n_basis, N)
    Returns: shape (N,) non-negative heights.
    """
    h = coeffs @ basis
    return np.maximum(0.0, h)

def _smooth_value_from_coeffs(coeffs, basis, beta=50.0):
    """
    Smooth surrogate for the value function, operating on DCT coefficients.
    Returns negative value for minimization.
    """
    h = _heights_from_coeffs(coeffs, basis)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _init_coeffs_from_profile(profile, basis):
    """
    Project a given height profile onto the DCT basis to get initial coefficients.
    """
    # Least squares projection
    coeffs, _, _, _ = np.linalg.lstsq(basis, profile, rcond=None)
    # Ensure non-negative
    return np.maximum(0.0, coeffs)

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Precompute DCT basis for different ranks
    ranks = [5, 8, 12, 16]
    bases = {r: _dct_basis(r) for r in ranks}

    # Generate diverse starting profiles
    x = np.linspace(0, N - 1, N)
    profiles = []
    # Uniform
    profiles.append(np.ones(N))
    # Gaussian
    profiles.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    profiles.append(p1 + p2)
    # Semicircle
    profiles.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    profiles.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    profiles.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Random smooth profiles
    for _ in range(15):
        freqs = RNG.randint(1, 6, size=4)
        phases = RNG.uniform(0, 2 * np.pi, size=4)
        amps = RNG.uniform(0.5, 1.0, size=4)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        profiles.append(np.maximum(0.1, prof))

    # Evaluate initial profiles
    for p in profiles:
        track(p)

    # Optimization Phase: Optimize DCT coefficients for each rank and profile
    betas = [30.0, 80.0, 150.0]

    for rank in ranks:
        if time.time() > deadline:
            break
        basis = bases[rank]
        n_basis = rank

        for p in profiles:
            if time.time() > deadline:
                break

            x0 = _init_coeffs_from_profile(p, basis)

            for beta in betas:
                if time.time() > deadline:
                    break

                try:
                    res = opt.minimize(
                        lambda c: _smooth_value_from_coeffs(c, basis, beta),
                        x0,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * n_basis,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.fun < 0:  # -value, so lower is better
                        h_opt = _heights_from_coeffs(res.x, basis)
                        track(h_opt)
                except Exception:
                    pass

    # Polish phase: Perturb best heights, re-project, and re-optimize with high beta
    for _ in range(20):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(0.0, pert)

        # Use the highest rank for polishing
        rank = max(ranks)
        basis = bases[rank]
        x0 = _init_coeffs_from_profile(pert, basis)

        try:
            res = opt.minimize(
                lambda c: _smooth_value_from_coeffs(c, basis, 200.0),
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * rank,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = _heights_from_coeffs(res.x, basis)
            track(h_opt)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
