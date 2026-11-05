import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400


def compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    idx = np.arange(N)
    # Square wave: +1 for first N/2, -1 for last N/2
    sq = np.ones(N)
    sq[N // 2:] = -1.0

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # --- Parametric family: h[i] = sq[i] * P(t_i) where t_i = 2*i/N - 1 in [-1,1]
    # P(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    # We optimize [a0..a5] to minimize max(conv(h,h)) subject to sum(h) != 0.
    # By scaling, we can fix sum(h)=1, so we minimize max(conv(h/s, h/s)).

    t = 2.0 * idx / N - 1.0  # in [-1, 1]
    t = t + 1e-12  # avoid exact -1 boundary issues

    def make_h(coeffs):
        """Given polynomial coefficients, return h = sq * P(t)."""
        P = np.ones(N)
        for k in range(len(coeffs)):
            P = P * t + coeffs[k]  # Horner-like (but actually just polynomial eval)
        # Actually let's do proper polynomial evaluation
        P = np.zeros(N)
        for k in range(len(coeffs) - 1, -1, -1):
            P = P * t + coeffs[k]
        h = sq * P
        s = np.sum(h)
        if abs(s) < 1e-10:
            return None
        return h / s

    def objective(coeffs):
        h = make_h(coeffs)
        if h is None:
            return 1e10
        conv = np.convolve(h, h)
        return np.max(conv)

    # --- Phase 1: Evaluate a grid of polynomial coefficients
    # Start with reasonable ranges
    # For a symmetric window, odd coefficients should be ~0 (since sq is antisymmetric about center)
    # Actually sq is symmetric: sq[i] = sq[N-1-i] for i < N/2... no.
    # sq = [+1,...,+1, -1,...,-1]. This is NOT symmetric about the center in a simple way.
    # Let's just search broadly.

    grid_candidates = []

    # Constant polynomial (P=1): h = sq, sum = 0. Bad.
    # P = 1 + c*t: sum(h) = sum(sq) + c*sum(sq*t) = 0 + c*sum(sq*t)
    # sum(sq*t): t goes from -1 to 1, sq is +1 then -1
    # Let's just compute numerically for various coefficients

    # Try various constant + linear + quadratic + ...
    for a0 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for a1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for a2 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                coeffs = [a0, a1, a2]
                h = make_h(coeffs)
                if h is not None:
                    grid_candidates.append((coeffs, h))

    # Add cubic, quartic, quintic terms
    for a0 in [1.0, 2.0, 3.0]:
        for a1 in [-1.0, 0.0, 1.0]:
            for a2 in [-1.0, 0.0, 1.0]:
                for a3 in [-1.0, 0.0, 1.0]:
                    coeffs = [a0, a1, a2, a3]
                    h = make_h(coeffs)
                    if h is not None:
                        grid_candidates.append((coeffs, h))

    # Evaluate grid
    for coeffs, h in grid_candidates:
        if time.time() > deadline - 60:
            break
        try_candidate(h)

    # --- Phase 2: Differential evolution on polynomial coefficients
    # Use 6 coefficients for flexibility
    def de_objective(params):
        h = make_h(params)
        if h is None:
            return 1e10
        conv = np.convolve(h, h)
        return np.max(conv)

    # DE with 6 coefficients
    bounds_de = [(-3.0, 3.0)] * 6
    try:
        res = differential_evolution(
            de_objective,
            bounds=bounds_de,
            maxiter=100,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            workers=1
        )
        if time.time() < deadline - 20:
            h_de = make_h(res.x)
            if h_de is not None:
                try_candidate(h_de)
    except Exception:
        pass

    # DE with 4 coefficients (more constrained)
    bounds_de4 = [(-3.0, 3.0)] * 4
    try:
        res4 = differential_evolution(
            de_objective,
            bounds=bounds_de4,
            maxiter=100,
            popsize=30,
            tol=1e-8,
            seed=123,
            polish=False,
            workers=1
        )
        if time.time() < deadline - 15:
            h_de4 = make_h(res4.x)
            if h_de4 is not None:
                try_candidate(h_de4)
    except Exception:
        pass

    # --- Phase 3: Local refinement on the 400-dim space starting from best
    def soft_max(x, beta=200.0):
        m = np.max(x)
        return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

    def local_refine(h0, max_iter=300, beta=200.0):
        if time.time() > deadline - 10:
            return None

        def obj(x):
            conv = np.convolve(x, x)
            sm = soft_max(conv, beta=beta)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty

        try:
            res = minimize(
                obj,
                h0,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            return res.x
        except Exception:
            return None

    if time.time() < deadline - 25:
        h_best = best_h.copy()
        s_best = np.sum(h_best)
        if abs(s_best) > 1e-10:
            h_best = h_best / s_best

        h_refined = local_refine(h_best, max_iter=400, beta=200.0)
        if h_refined is not None:
            try_candidate(h_refined)

        # Perturbations
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            h_pert = best_h.copy() + 0.05 * np.random.randn(N)
            s_pert = np.sum(h_pert)
            if abs(s_pert) > 1e-10:
                h_pert = h_pert / s_pert
                h_refined = local_refine(h_pert, max_iter=250, beta=200.0)
                if h_refined is not None:
                    try_candidate(h_refined)

    # --- Phase 4: L-BFGS-B polish
    if time.time() < deadline - 8:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0

        for beta in [150.0, 300.0, 500.0]:
            if time.time() > deadline - 3:
                break

            def obj_b(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 500.0 * (s - 1.0) ** 2

            try:
                res = minimize(
                    obj_b,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 500, 'gtol': 1e-8}
                )
                if time.time() > deadline - 2:
                    break
                h_new = res.x
                try_candidate(h_new)
                s_new = np.sum(h_new)
                if abs(s_new) > 1e-10:
                    h0 = h_new.copy() / s_new
            except Exception:
                pass

    # --- Phase 5: Try more polynomial windows with higher order
    if time.time() < deadline - 10:
        # 8-coefficient polynomial
        def de_objective8(params):
            h = make_h(params)
            if h is None:
                return 1e10
            conv = np.convolve(h, h)
            return np.max(conv)

        bounds8 = [(-3.0, 3.0)] * 8
        try:
            res8 = differential_evolution(
                de_objective8,
                bounds=bounds8,
                maxiter=80,
                popsize=25,
                tol=1e-8,
                seed=777,
                polish=False,
                workers=1
            )
            if time.time() < deadline - 5:
                h8 = make_h(res8.x)
                if h8 is not None:
                    try_candidate(h8)
        except Exception:
            pass

    return best_h
