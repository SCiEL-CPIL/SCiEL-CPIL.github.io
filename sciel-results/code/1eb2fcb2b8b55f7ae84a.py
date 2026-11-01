"""Uncertainty bound minimization via spherical reparameterization and adaptive stochastic local search."""
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
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
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
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
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


def _bound_from_sphere(theta1, theta2):
    """
    Convert spherical coordinates (unit sphere) to (a0, a4, a8) and compute bound.
    a0 = sin(theta2) * cos(theta1)
    a4 = sin(theta2) * sin(theta1)
    a8 = cos(theta2)
    """
    st2 = np.sin(theta2)
    a0 = st2 * np.cos(theta1)
    a4 = st2 * np.sin(theta1)
    a8 = np.cos(theta2)
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses spherical reparameterization with adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    # Convert seed to spherical coordinates (normalize first)
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        theta1_seed = np.arctan2(seed[1], seed[0])
        theta2_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
    else:
        theta1_seed = 0.0
        theta2_seed = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration with multiple starting points on the sphere
    # We fix radius to 1 due to scale invariance.
    # Try different angles.
    
    # Initialize with a grid of starting points + random perturbations
    start_points = []
    # Add seed point
    start_points.append((theta1_seed, theta2_seed))
    
    # Add random points
    for _ in range(200):
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(0, np.pi)
        start_points.append((t1, t2))
        
    # Also add some structured points
    for t2 in np.linspace(0.1, np.pi - 0.1, 10):
        for t1 in np.linspace(-np.pi, np.pi, 10):
             start_points.append((t1, t2))

    for t1, t2 in start_points:
        if time.time() > _DEADLINE - 30:
            break
        b = _bound_from_sphere(t1, t2)
        if b < best_bound:
            best_bound = b
            st2 = np.sin(t2)
            best_params = np.array([
                st2 * np.cos(t1),
                st2 * np.sin(t1),
                np.cos(t2)
            ])

    # Phase 2: Adaptive stochastic local search on the best point found
    # Work in spherical coordinates (theta1, theta2)
    # We need to handle the periodicity of theta1 and bounds of theta2
    
    # Get best state from Phase 1
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    t1_best = np.arctan2(best_params[1], best_params[0])
    t2_best = np.arccos(np.clip(best_params[2] / r_best, -1.0, 1.0))

    state = np.array([t1_best, t2_best])
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step sizes for theta1 and theta2
    steps = np.array([0.1, 0.1])
    
    max_iters = 20000
    iter_count = 0
    no_improve_count = 0
    patience = 500

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate perturbation
        delta = rng.normal(0, 1, 2) * steps
        new_state = state + delta
        
        # Wrap theta1 to [-pi, pi]
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        # Clamp theta2 to [0, pi]
        new_state[1] = np.clip(new_state[1], 1e-5, np.pi - 1e-5)
        
        new_val = _bound_from_sphere(new_state[0], new_state[1])
        
        if not np.isfinite(new_val):
            new_val = 1e18
            
        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            no_improve_count = 0
            # Shrink steps slightly on improvement to converge
            steps *= 0.95
        else:
            no_improve_count += 1
            # If we haven't improved for a while, shrink steps more aggressively
            # or accept a small fraction of worse moves to escape local minima
            if no_improve_count > patience:
                 # Reset to best state and try a larger random jump
                 state = best_state + rng.normal(0, 0.5, 2)
                 state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi
                 state[1] = np.clip(state[1], 1e-5, np.pi - 1e-5)
                 no_improve_count = 0
            else:
                 # Standard random walk: accept worse moves with some probability or just shrink
                 # Here we just shrink step size to explore locally
                 steps *= 0.99
                 
        # Adaptive step size adjustment based on acceptance rate (simplified)
        # If steps become too small, they might get stuck. 
        # If steps become too large, they might jump over minima.
        
        iter_count += 1
        
        # If we are very close to the boundary, handle carefully
        if best_val < 1.0:
            # Check if we've converged
            if iter_count % 1000 == 0 and no_improve_count > 1000:
                # Try a different restart from a random point if stuck
                # But first, ensure we have a good global optimum
                pass

    # Phase 3: Fine-tuning
    steps = np.array([0.001, 0.001])
    state = best_state.copy()
    best_val = _bound_from_sphere(best_state[0], best_state[1])
    
    for _ in range(5000):
        if time.time() > _DEADLINE - 5:
            break
        delta = rng.normal(0, 1, 2) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 1e-6, np.pi - 1e-6)
        
        new_val = _bound_from_sphere(new_state[0], new_state[1])
        if not np.isfinite(new_val):
            new_val = 1e18
            
        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.98
        else:
            steps *= 0.99

    # Convert back to (a0, a4, a8)
    t1_final = best_state[0]
    t2_final = best_state[1]
    st2 = np.sin(t2_final)
    result = np.array([
        st2 * np.cos(t1_final),
        st2 * np.sin(t1_final),
        np.cos(t2_final)
    ])

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
