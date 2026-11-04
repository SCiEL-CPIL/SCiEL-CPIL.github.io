"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Parametric search over diverse structures
    candidates = []

    # Cosine modes
    for k in range(1, 30):
        for t in np.linspace(-1.5, 1.5, 25):
            candidates.append(1.0 + t * np.cos(2.0 * np.pi * k * idx / N))
    # Sine modes
    for k in range(1, 30):
        for t in np.linspace(-1.5, 1.5, 25):
            candidates.append(1.0 + t * np.sin(2.0 * np.pi * k * idx / N))
    # Quadratic phase
    for t in np.linspace(-1.5, 1.5, 25):
        candidates.append(1.0 + t * np.sin(2.0 * np.pi * idx * idx / N))
    # Linear ramp
    x = idx / (N - 1)
    for t in np.linspace(-1.5, 1.5, 25):
        candidates.append(1.0 + t * (2.0 * x - 1.0))
    # Quadratic
    for t in np.linspace(-1.5, 1.5, 25):
        candidates.append(1.0 + t * (2.0 * x - 1.0) ** 2)
    # Alternating with decay
    for t in np.linspace(-1.5, 1.5, 25):
        candidates.append(1.0 + t * (-1.0) ** idx * np.exp(-idx / 100.0))
    # Two-frequency mix
    for k1 in [1, 2, 3, 5, 7]:
        for k2 in [2, 3, 4, 6, 8, 10]:
            for t1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for t2 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    candidates.append(1.0 + t1 * np.cos(2.0 * np.pi * k1 * idx / N) + t2 * np.sin(2.0 * np.pi * k2 * idx / N))

    for h in candidates:
        if time.time() > deadline - 30:
            break
        try_candidate(h)

    # Phase 2: Local refinement using soft-max surrogate
    if time.time() < deadline - 15:
        def soft_max(x, beta=300.0):
            m = np.max(x)
            return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0  # normalize sum to 1

        def obj(x):
            conv = np.convolve(x, x)
            sm = soft_max(conv, beta=300.0)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty

        def obj_neg(x):
            # We want to minimize max(conv), which is equivalent to making it as negative as possible
            # But we need sum(x) = 1 for normalization
            conv = np.convolve(x, x)
            sm = soft_max(conv, beta=300.0)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty

        # Try multiple optimization methods and starting points
        start_points = [best_h.copy() / np.sum(best_h)]
        
        # Add some perturbed starting points
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            pert = start_points[0].copy() + 0.1 * np.random.randn(N)
            s_p = np.sum(pert)
            if abs(s_p) > 1e-10:
                start_points.append(pert / s_p)

        for sp in start_points:
            if time.time() > deadline - 8:
                break
            try:
                res = minimize(obj, sp, method='L-BFGS-B', 
                             options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-6})
                if time.time() > deadline - 3:
                    break
                h_new = res.x
                s_new = np.sum(h_new)
                if abs(s_new) > 1e-10:
                    # Scale back to original sum for evaluation
                    h_eval = h_new * (np.sum(best_h) / s_new) if abs(s_new) > 1e-10 else h_new
                    v_new = compute_value(h_new)
                    if v_new < best_val:
                        best_val = v_new
                        best_h = h_new.copy()
            except Exception:
                pass

    # Phase 3: Additional refinement with different beta values
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 3:
                break

            def obj_b(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 500.0 * (s - 1.0) ** 2

            try:
                res = minimize(obj_b, h0, method='L-BFGS-B',
                             options={'maxiter': 150, 'maxfun': 400, 'gtol': 1e-7})
                if time.time() > deadline - 2:
                    break
                h_new = res.x
                v_new = compute_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
                    h0 = h_new.copy() / np.sum(h_new)
            except Exception:
                pass

    return best_h
# EVOLVE-BLOCK-END
