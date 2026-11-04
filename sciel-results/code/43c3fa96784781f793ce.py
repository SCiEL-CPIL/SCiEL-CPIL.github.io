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

    def try_candidate(h, best_h, best_score):
        """Try a candidate and return updated best."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        if np.sum(h) < 1e-15:
            return h, best_score
        s = score(h)
        if s > best_score:
            return h.copy(), s
        return best_h, best_score

    # Key insight: concentrated spikes maximize the ratio.
    # For a single unit spike: ratio = 1/d
    # For uniform: ratio = 1/(N*d)
    # So we want to concentrate mass as much as possible.
    # But multi-point shapes with specific correlations may do even better.
    
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = score(best_h)

    # Phase 1: Try all single-point spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 2: Try two-point spikes at various separations
    for i in range(N):
        if time.time() > deadline - 40:
            break
        for j in range(i + 1, N):
            if time.time() > deadline - 40:
                break
            for ratio in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = ratio
                best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 3: Try small concentrated shapes (width 1-3)
    for center in range(N):
        if time.time() > deadline - 30:
            break
        for width in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            # Gaussian
            t = np.arange(N)
            h = np.exp(-0.5 * ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Triangular
            h = np.maximum(0, 1 - np.abs(t - center) / width)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Box
            h = np.where(np.abs(t - center) <= width, 1.0, 0.0)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Exponential
            h = np.exp(-np.abs(t - center) / width)
            best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 4: Broader parametric search
    centers = np.linspace(0, N-1, 20)
    widths = np.logspace(-1, 2, 30)
    for center in centers:
        if time.time() > deadline - 20:
            break
        for width in widths:
            if time.time() > deadline - 20:
                break
            t = np.arange(N)
            # Gaussian
            h = np.exp(-0.5 * ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Lorentzian
            h = 1.0 / (1.0 + ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            # Power law
            for p in [1.0, 2.0, 3.0, 5.0]:
                h = (1.0 + np.abs(t - center) / width) ** (-p)
                best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 5: Local optimization from best shape
    if time.time() < deadline - 15:
        h0 = best_h.copy()
        if np.sum(h0**2) > 1e-15:
            h0 = h0 / np.sqrt(np.sum(h0**2))

            def nm_obj(h):
                h = np.maximum(0, h)
                s = score(h)
                return -s if s > 0 else 1e10

            try:
                res = minimize(
                    nm_obj, h0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                s = -res.fun
                if s > best_score and s > 0:
                    best_score = s
                    best_h = np.maximum(0, res.x).copy()
            except Exception:
                pass

    # Phase 6: Coordinate descent polish
    if time.time() < deadline - 5:
        h = best_h.copy()
        for iteration in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try scaling this coordinate
                for mult in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 5.0]:
                    h[i] = max(0.0, old * mult) if old > 0 else mult
                    s = score(h)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break

    # Phase 7: Random restarts with concentrated shapes
    if time.time() < deadline - 3:
        np.random.seed(42)
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            n_active = np.random.choice([1, 2, 3, 4, 5])
            indices = np.random.choice(N, size=n_active, replace=False)
            h = np.zeros(N)
            for idx in indices:
                h[idx] = np.random.lognormal(0, 1)
            best_h, best_score = try_candidate(h, best_h, best_score)

    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)

# EVOLVE-BLOCK-END
