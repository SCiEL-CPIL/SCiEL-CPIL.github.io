"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin below 120s limit

    def score(h):
        """Compute ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step function."""
        h = np.asarray(h, dtype=float)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        ac = np.correlate(h, h, mode='full')
        max_ac = np.max(ac)
        if max_ac < 1e-30:
            return 0.0
        return sum_sq / (d * sum_h * max_ac)

    # Parametric shape families with continuous parameters
    # Key insight: optimal shape is a concentrated bump; width controls trade-off

    def make_gaussian(center, width, amplitude=1.0):
        """Gaussian bump on grid."""
        t = np.arange(N)
        h = amplitude * np.exp(-0.5 * ((t - center) / width)**2)
        return np.maximum(0, h)

    def make_lorentzian(center, width, amplitude=1.0):
        """Lorentzian (Cauchy) bump on grid."""
        t = np.arange(N)
        h = amplitude / (1.0 + ((t - center) / width)**2)
        return np.maximum(0, h)

    def make_power(center, width, exponent, amplitude=1.0):
        """Power-law bump: (1 + |t-c|/w)^(-p)"""
        t = np.arange(N)
        h = amplitude * (1.0 + np.abs(t - center) / width) ** (-exponent)
        return np.maximum(0, h)

    def make_plateau(center, half_width, amplitude=1.0):
        """Flat-top with smooth edges."""
        t = np.arange(N)
        dist = np.abs(t - center)
        h = np.where(dist <= half_width, amplitude, 0.0)
        return h

    def make_bell(center, width, sharpness, amplitude=1.0):
        """Generalized bell: exp(-(x/w)^p)"""
        t = np.arange(N)
        h = amplitude * np.exp(-np.abs(t - center) / width) ** sharpness
        return np.maximum(0, h)

    best_h = np.ones(N)
    best_score = score(best_h)

    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        if np.sum(h) < 1e-15:
            return
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Coarse grid search over parametric families
    centers = np.linspace(0, N-1, 25)
    widths = np.logspace(-1.5, 2.5, 40)
    exponents = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    sharpnesses = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    for center in centers:
        if time.time() > deadline - 30:
            break
        for width in widths:
            if time.time() > deadline - 30:
                break
            # Gaussian
            try_candidate(make_gaussian(center, width))
            # Lorentzian
            try_candidate(make_lorentzian(center, width))
            # Power law
            for exp_val in exponents:
                if time.time() > deadline - 30:
                    break
                try_candidate(make_power(center, width, exp_val))
            # Generalized bell
            for sh in sharpnesses:
                if time.time() > deadline - 30:
                    break
                try_candidate(make_bell(center, width, sh))

    # Phase 2: Fine-tune best parametric shape with local optimization
    if time.time() < deadline - 20:
        # Extract approximate parameters from best_h
        # Find center of mass
        total = np.sum(best_h)
        if total > 1e-15:
            com = np.sum(np.arange(N) * best_h) / total
            # Estimate width from second moment
            var = np.sum((np.arange(N) - com)**2 * best_h) / total
            est_width = np.sqrt(var) if var > 0 else 1.0

            # Local optimization around the best shape
            # Use the shape as initial guess and do Nelder-Mead
            h0 = best_h.copy()
            if np.sum(h0) > 1e-15:
                h0 = h0 / np.sqrt(np.sum(h0**2))  # normalize

                def nm_obj(h):
                    h = np.maximum(0, h)
                    s = score(h)
                    return -s if s > 0 else 1e10

                try:
                    res = minimize(
                        nm_obj,
                        h0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    s = -res.fun
                    if s > best_score and s > 0:
                        best_score = s
                        best_h = np.maximum(0, res.x).copy()
                except Exception:
                    pass

    # Phase 3: Multi-start with refined parametric search
    if time.time() < deadline - 15:
        np.random.seed(123)
        # More refined centers and widths around the best
        best_center = np.sum(np.arange(N) * best_h) / max(np.sum(best_h), 1e-15)
        best_width = np.sqrt(np.sum((np.arange(N) - best_center)**2 * best_h) / max(np.sum(best_h), 1e-15))

        for _ in range(50):
            if time.time() > deadline - 10:
                break
            c = best_center + np.random.randn() * best_width * 0.5
            w = best_width * np.exp(np.random.randn() * 0.3)
            c = np.clip(c, 0, N-1)
            w = np.clip(w, 0.5, N)

            # Try multiple shape types
            for shape_fn in [make_gaussian, make_lorentzian, make_bell]:
                try:
                    if shape_fn == make_bell:
                        h = shape_fn(c, w, 2.0 + np.random.rand() * 2.0)
                    else:
                        h = shape_fn(c, w)
                    try_candidate(h)
                except Exception:
                    pass

    # Phase 4: Coordinate descent polish on the best solution
    if time.time() < deadline - 5:
        h = best_h.copy()
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for mult in [0.3, 0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 2.0, 0.0]:
                    h[i] = max(0.0, old * mult)
                    s = score(h)
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break

    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return list(best_h)

# EVOLVE-BLOCK-END
