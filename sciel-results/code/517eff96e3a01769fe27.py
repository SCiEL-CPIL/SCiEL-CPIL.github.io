"""
Optimized program using a global, derivative-free Nelder-Mead simplex
optimization (with multi-start and a final high-precision polish) to minimize
the uncertainty bound for the Hermite-Gaussian test function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    This mirrors the seed program's metric definition exactly.
    """
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0

    # Coefficients of P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12

    p = np.array([a0, a2, a4, a6, a8, a10, a12])

    # Convolution to get coefficients of P(x)^2
    q = np.convolve(p, p)

    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)

    I = q * factors
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10

    x2 = I[1] / I[0]

    # Calculate R(x) for Fourier side
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        if j >= 1 and j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val

    s = np.convolve(r, r)

    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)

    I_R = s * factors_R
    norm_R_sq = I_R[0]

    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)

    product = math.sqrt(x2 * xi2)

    return product

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # safety margin before the 210s kill

    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()

    def objective(x):
        try:
            return _compute_bound(x[0], x[1], x[2])
        except Exception:
            return 1e10

    # Global, derivative-free Nelder-Mead simplex with many multi-starts.
    # Bounds (soft) for generating starting points.
    bounds = [(-8.0, 8.0), (-2.0, 2.0), (-0.3, 0.3)]

    rng = np.random.default_rng(12345)

    # Build a diverse set of starting points:
    #   - the seed
    #   - a Sobol-like low-discrepancy grid (simple Halton-ish via rng)
    #   - random points
    starting_points = [seed_coeffs.copy()]

    # Add structured points on a small grid
    grid_c0 = np.linspace(bounds[0][0], bounds[0][1], 9)
    grid_c4 = np.linspace(bounds[1][0], bounds[1][1], 5)
    grid_c8 = np.linspace(bounds[2][0], bounds[2][1], 5)
    for c0 in grid_c0:
        for c4 in grid_c4:
            for c8 in grid_c8:
                starting_points.append(np.array([c0, c4, c8]))

    # Add random points
    for _ in range(40):
        pt = np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
        ])
        starting_points.append(pt)

    # Sort starting points by initial objective (cheap: just evaluate each once)
    scored = []
    for pt in starting_points:
        v = objective(pt)
        scored.append((v, pt))
    scored.sort(key=lambda t: t[0])

    # Run Nelder-Mead from the best starting points, with tight tolerances.
    # Limit total number of NM runs by time budget.
    max_nm_runs = 60
    nm_runs = 0
    for v0, pt in scored:
        if nm_runs >= max_nm_runs:
            break
        if time.time() > deadline - 15.0:
            break
        if v0 >= best_val and nm_runs > 5:
            # stop early if we're far past the best and have done some runs
            continue
        try:
            res = scipy.optimize.minimize(
                objective,
                pt,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'maxiter': 3000,
                    'maxfev': 5000,
                    'adaptive': True,
                }
            )
            nm_runs += 1
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish pass on the best point
    if time.time() < deadline - 5.0:
        try:
            res_final = scipy.optimize.minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-14,
                    'fatol': 1e-14,
                    'maxiter': 10000,
                    'maxfev': 20000,
                    'adaptive': True,
                }
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs

# Need scipy for minimize
import scipy.optimize
# EVOLVE-BLOCK-END
