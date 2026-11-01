"""Uncertainty bound minimization using unit-sphere reparameterization with multi-start CMA-ES-like search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _eval_on_sphere(u, v):
    """
    Evaluate the bound at the point on the unit sphere given by (u, v).
    u in [0, 2*pi), v in [0, pi]
    c = (u, v) maps to c0 = sin(v)*cos(u), c4 = sin(v)*sin(u), c8 = cos(v)
    Then scale by r to get (a0, a4, a8).
    """
    cv = np.cos(v)
    sv = np.sin(v)
    c0 = sv * np.cos(u)
    c4 = sv * np.sin(u)
    c8 = cv
    return c0, c4, c8


def _bound_from_uv(u, v, r):
    """Compute bound at spherical point (u, v) with radius r."""
    c0, c4, c8 = _eval_on_sphere(u, v)
    a0 = r * c0
    a4 = r * c4
    a8 = r * c8
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization with multi-start stochastic local search.
    """
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(2024)

    # Phase 1: Grid-based exploration on the sphere
    # The bound is scale-invariant in the sense that we can explore different radii
    # but the angular distribution matters most.
    # Try a coarse grid of (u, v) with several radii.
    radii = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    n_u = 40
    n_v = 25

    for r in radii:
        if time.time() > _DEADLINE - 60:
            break
        for i in range(n_u):
            if time.time() > _DEADLINE - 60:
                break
            u = i * 2.0 * np.pi / n_u
            for j in range(n_v):
                v = (j + 0.5) * np.pi / n_v
                b = _bound_from_uv(u, v, r)
                if b < best_bound:
                    best_bound = b
                    c0, c4, c8 = _eval_on_sphere(u, v)
                    best_params = np.array([r * c0, r * c4, r * c8])

    # Phase 2: Multi-start local search from promising points
    # Collect a set of starting points: the best found, plus some random and seed-neighbor points
    start_points = []

    # Add seed neighborhood
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        u_seed = np.arctan2(seed[1], seed[0])
        v_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
        for du in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            for dv in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                u = (u_seed + du + 2 * np.pi) % (2 * np.pi)
                v = np.clip(v_seed + dv, 0.01, np.pi - 0.01)
                start_points.append((u, v, r_seed))

    # Add best found
    r_best = np.linalg.norm(best_params)
    if r_best > 1e-15:
        u_best = np.arctan2(best_params[1], best_params[0])
        v_best = np.arccos(np.clip(best_params[2] / r_best, -1.0, 1.0))
        start_points.append((u_best, v_best, r_best))

    # Add random starts
    for _ in range(30):
        u = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0.05, np.pi - 0.05)
        r = rng.uniform(0.5, 8.0)
        start_points.append((u, v, r))

    # Phase 2: Stochastic local search with adaptive step sizes
    # For each starting point, run a local search
    for u0, v0, r0 in start_points:
        if time.time() > _DEADLINE - 20:
            break

        u, v, r = u0, v0, r0
        cur_bound = _bound_from_uv(u, v, r)
        if not np.isfinite(cur_bound):
            cur_bound = 1e18
        if cur_bound < best_bound:
            best_bound = cur_bound
            c0, c4, c8 = _eval_on_sphere(u, v)
            best_params = np.array([r * c0, r * c4, r * c8])

        # Adaptive step sizes
        step_u = 0.1
        step_v = 0.1
        step_r = 0.1
        temperature = 0.5
        cooling = 0.999

        max_local_iters = 5000
        for it in range(max_local_iters):
            if time.time() > _DEADLINE - 10:
                break

            # Generate perturbation
            du = rng.normal(0, 1) * step_u
            dv = rng.normal(0, 1) * step_v
            dr = rng.normal(0, 1) * step_r

            new_u = (u + du + np.pi) % (2 * np.pi) - np.pi
            new_u = (new_u + 2 * np.pi) % (2 * np.pi)
            new_v = np.clip(v + dv, 0.005, np.pi - 0.005)
            new_r = np.clip(r + dr, 0.1, 20.0)

            new_bound = _bound_from_uv(new_u, new_v, new_r)
            if not np.isfinite(new_bound):
                new_bound = 1e18

            delta = new_bound - cur_bound

            if new_bound < cur_bound:
                u, v, r = new_u, new_v, new_r
                cur_bound = new_bound
                step_u *= 0.97
                step_v *= 0.97
                step_r *= 0.97
            else:
                # Metropolis accept
                if np.exp(-delta / max(temperature, 1e-10)) > rng.random():
                    u, v, r = new_u, new_v, new_r
                    cur_bound = new_bound
                    step_u *= 1.02
                    step_v *= 1.02
                    step_r *= 1.02
                else:
                    step_u *= 1.03
                    step_v *= 1.03
                    step_r *= 1.03

            temperature *= cooling
            if temperature < 1e-8:
                temperature = 1e-8

            if cur_bound < best_bound:
                best_bound = cur_bound
                c0, c4, c8 = _eval_on_sphere(u, v)
                best_params = np.array([r * c0, r * c4, r * c8])

    # Phase 3: Fine-tuning around the best point
    if best_bound < 1e18:
        r_final = np.linalg.norm(best_params)
        if r_final > 1e-15:
            u_final = np.arctan2(best_params[1], best_params[0])
            v_final = np.arccos(np.clip(best_params[2] / r_final, -1.0, 1.0))

            # Run a finer local search
            step_u = 0.01
            step_v = 0.01
            step_r = 0.01
            temperature = 1e-4
            cooling = 0.998

            for _ in range(10000):
                if time.time() > _DEADLINE - 5:
                    break

                du = rng.normal(0, 1) * step_u
                dv = rng.normal(0, 1) * step_v
                dr = rng.normal(0, 1) * step_r

                new_u = (u_final + du + np.pi) % (2 * np.pi) - np.pi
                new_u = (new_u + 2 * np.pi) % (2 * np.pi)
                new_v = np.clip(v_final + dv, 0.001, np.pi - 0.001)
                new_r = np.clip(r_final + dr, 0.1, 20.0)

                new_bound = _bound_from_uv(new_u, new_v, new_r)
                if not np.isfinite(new_bound):
                    new_bound = 1e18

                if new_bound < best_bound:
                    best_bound = new_bound
                    u_final = new_u
                    v_final = new_v
                    r_final = new_r
                    step_u *= 0.96
                    step_v *= 0.96
                    step_r *= 0.96
                else:
                    if np.exp(-(new_bound - best_bound) / max(temperature, 1e-10)) > rng.random():
                        u_final = new_u
                        v_final = new_v
                        r_final = new_r
                        step_u *= 1.01
                        step_v *= 1.01
                        step_r *= 1.01
                    else:
                        step_u *= 1.02
                        step_v *= 1.02
                        step_r *= 1.02

                temperature *= cooling
                if temperature < 1e-10:
                    temperature = 1e-10

            c0, c4, c8 = _eval_on_sphere(u_final, v_final)
            candidate = np.array([r_final * c0, r_final * c4, r_final * c8])

            # Verify
            final_bound = _compute_bound(*candidate)
            if np.isfinite(final_bound) and final_bound <= best_bound + 1e-12:
                best_params = candidate
                best_bound = final_bound

    # Final fallback: if nothing better than seed was found, return seed
    final_bound_check = _compute_bound(*best_params)
    if not np.isfinite(final_bound_check) or final_bound_check > seed_bound + 1e-12:
        return seed.copy()

    return best_params
# EVOLVE-BLOCK-END
