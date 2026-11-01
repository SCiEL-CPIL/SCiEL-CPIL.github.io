"""Uncertainty bound minimization via unit-sphere reparameterization and adaptive stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    
    Key insight: The uncertainty product is homogeneous of degree 0 in (a0,a4,a8),
    so we can normalize to the unit sphere and search only over directions.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

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


def _bound_from_unit_sphere(c0, c4, c8):
    """
    Compute bound from a unit vector (c0, c4, c8) on the unit sphere.
    Since the uncertainty product is homogeneous of degree 0, the radius
    doesn't matter - only the direction.
    """
    # Normalize to unit length (should already be ~1, but be safe)
    norm = np.sqrt(c0*c0 + c4*c4 + c8*c8)
    if norm < 1e-15:
        return 1e18
    a0 = c0 / norm
    a4 = c4 / norm
    a8 = c8 / norm
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    
    Uses unit-sphere reparameterization: since the uncertainty product is
    homogeneous of degree 0, we search only over directions on the unit
    sphere S^2 using adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_norm = np.linalg.norm(seed)
    seed_unit = seed / seed_norm
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_unit = seed_unit.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration - sample many random points on the unit sphere
    n_samples = 500
    for _ in range(n_samples):
        if time.time() > _DEADLINE - 80:
            break
        # Uniform sampling on S^2 using spherical coordinates
        u = rng.uniform(-1, 1)  # cos(theta2)
        phi = rng.uniform(0, 2 * np.pi)  # theta1
        st2 = np.sqrt(1 - u*u)
        c0 = st2 * np.cos(phi)
        c4 = st2 * np.sin(phi)
        c8 = u
        b = _bound_from_unit_sphere(c0, c4, c8)
        if b < best_bound:
            best_bound = b
            best_unit = np.array([c0, c4, c8])

    # Also explore neighborhood of seed direction
    for _ in range(200):
        if time.time() > _DEADLINE - 80:
            break
        # Random perturbation in tangent space
        # Generate random vector, project onto tangent plane of sphere at best_unit
        v = rng.normal(0, 1, 3)
        v = v - np.dot(v, best_unit) * best_unit  # project to tangent
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            continue
        v = v / v_norm
        # Small step in tangent direction
        step = rng.uniform(0.05, 0.5)
        new_unit = best_unit + step * v
        new_unit = new_unit / np.linalg.norm(new_unit)
        b = _bound_from_unit_sphere(*new_unit)
        if b < best_bound:
            best_bound = b
            best_unit = new_unit

    # Phase 2: Adaptive stochastic local search on the unit sphere
    # State: point on unit sphere (3D vector)
    state = best_unit.copy()
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step size (scalar, applied to tangent-space perturbations)
    step_size = 0.1
    accept_count = 0
    total_count = 0
    temperature = 0.01
    cooling = 0.9995

    max_iters = 50000
    iter_count = 0

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 15:
            break

        # Generate perturbation in tangent space
        v = rng.normal(0, 1, 3)
        v = v - np.dot(v, state) * state  # project to tangent plane
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            iter_count += 1
            continue
        v = v / v_norm

        # Move in tangent direction
        new_state = state + step_size * v
        new_state = new_state / np.linalg.norm(new_state)

        new_val = _bound_from_unit_sphere(*new_state)
        if not np.isfinite(new_val):
            new_val = 1e18

        delta_val = new_val - best_val
        total_count += 1

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            # Shrink step on improvement
            step_size *= 0.97
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-12 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                step_size *= 1.02
            else:
                step_size *= 1.05

        # Adaptive step size adjustment
        if total_count % 200 == 0:
            ratio = accept_count / total_count
            if ratio > 0.5:
                step_size *= 1.15
            elif ratio < 0.2:
                step_size *= 0.7
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-8:
            temperature = 1e-8

        iter_count += 1

    # Phase 3: Fine-tuning with much smaller steps
    step_size = 0.005
    state = best_state.copy()
    temperature = 1e-5
    cooling = 0.998

    for _ in range(30000):
        if time.time() > _DEADLINE - 5:
            break

        # Generate perturbation in tangent space
        v = rng.normal(0, 1, 3)
        v = v - np.dot(v, state) * state
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            continue
        v = v / v_norm

        new_state = state + step_size * v
        new_state = new_state / np.linalg.norm(new_state)

        new_val = _bound_from_unit_sphere(*new_state)
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            step_size *= 0.95
        else:
            if new_val - best_val < 1e-14 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                step_size *= 1.01
            else:
                step_size *= 1.03

        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

    # Return the unit-sphere direction (the actual coefficients can be any scalar multiple)
    # We return the unit vector itself, which is a valid choice of (a0, a4, a8)
    result = best_state.copy()

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound * 1.001:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
