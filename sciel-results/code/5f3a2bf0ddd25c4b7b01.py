"""Uncertainty bound minimization via unit-sphere reparameterization and geodesic local search."""
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

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier coefficients
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

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


def _sphere_step(current, step_size, rng):
    """Generate a new point on the sphere near current via geodesic step."""
    v = rng.normal(0, 1, 3)
    v = v - np.dot(v, current) * current
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return current.copy()
    v = v / norm_v
    new_point = np.cos(step_size) * current + np.sin(step_size) * v
    return new_point / np.linalg.norm(new_point)


def _local_search(start, max_iters, initial_step, rng, deadline):
    """Perform geodesic local search on the unit sphere."""
    current = start.copy()
    current_bound = _compute_bound(*current)
    best_local = current.copy()
    best_local_bound = current_bound
    step = initial_step
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        new_point = _sphere_step(current, step, rng)
        new_bound = _compute_bound(*new_point)
        
        if not np.isfinite(new_bound):
            new_bound = 1e18
        
        if new_bound < current_bound:
            current = new_point
            current_bound = new_bound
            if new_bound < best_local_bound:
                best_local_bound = new_bound
                best_local = new_point.copy()
            step *= 0.95
        else:
            if np.exp(-(new_bound - current_bound) / 0.01) > rng.random():
                current = new_point
                current_bound = new_bound
                step *= 1.02
            else:
                step *= 0.98
    
    return best_local, best_local_bound


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit sphere reparameterization with multi-start geodesic local search.
    """
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(12345)

    # Phase 1: Fibonacci sphere exploration
    n_points = 400
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n_points):
        if time.time() > _DEADLINE - 80:
            break
        z = 1.0 - 2.0 * i / (n_points - 1)
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        u0 = r * np.cos(theta)
        u4 = r * np.sin(theta)
        u8 = z
        
        b = _compute_bound(u0, u4, u8)
        if b < best_bound and np.isfinite(b):
            best_bound = b
            best_params = np.array([u0, u4, u8])

    # Try seed direction
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        seed_unit = seed / r_seed
        b = _compute_bound(*seed_unit)
        if b < best_bound and np.isfinite(b):
            best_bound = b
            best_params = seed_unit.copy()

    # Phase 2: Multi-start local search
    local, local_bound = _local_search(best_params, 5000, 0.1, rng, _DEADLINE - 30)
    if local_bound < best_bound:
        best_bound = local_bound
        best_params = local.copy()

    for _ in range(15):
        if time.time() > _DEADLINE - 40:
            break
        v = rng.normal(0, 1, 3)
        start = v / np.linalg.norm(v)
        local, local_bound = _local_search(start, 3000, 0.1, rng, _DEADLINE - 40)
        if local_bound < best_bound:
            best_bound = local_bound
            best_params = local.copy()

    # Phase 3: Fine-tuning
    local, local_bound = _local_search(best_params, 10000, 0.01, rng, _DEADLINE - 15)
    if local_bound < best_bound:
        best_bound = local_bound
        best_params = local.copy()

    local, local_bound = _local_search(best_params, 10000, 0.001, rng, _DEADLINE - 5)
    if local_bound < best_bound:
        best_bound = local_bound
        best_params = local.copy()

    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()

    return best_params
# EVOLVE-BLOCK-END
