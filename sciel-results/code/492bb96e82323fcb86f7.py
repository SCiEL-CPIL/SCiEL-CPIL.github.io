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

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n
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


def _bound_from_normalized(c):
    """
    Compute bound from a normalized coefficient vector c = (c0, c4, c8) with ||c||=1.
    We scale by an optimal radius found via 1D search.
    """
    # The bound is homogeneous of degree 0 in (a0, a4, a8) because:
    # P(x) scales linearly, |P|^2 scales quadratically, so m_k scales quadratically
    # m2/m0 is scale-invariant. Same for n2/n0.
    # So the bound depends only on the direction, not the magnitude!
    # Therefore we just need the direction on the unit sphere.
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return 1e18
    c_norm = c / norm
    return _compute_bound(c_norm[0], c_norm[1], c_norm[2])


def _gradient_bound(c, h=1e-6):
    """
    Compute gradient of the bound with respect to the normalized direction.
    Uses finite differences on the sphere.
    """
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return np.zeros(3)
    c_norm = c / norm
    
    f0 = _bound_from_normalized(c_norm)
    if not np.isfinite(f0):
        return np.zeros(3)
    
    grad = np.zeros(3)
    for i in range(3):
        # Move in direction i, then project back to sphere
        perturbed = c_norm.copy()
        perturbed[i] += h
        # Project to unit sphere
        perturbed = perturbed / np.linalg.norm(perturbed)
        fi_plus = _bound_from_normalized(perturbed)
        
        perturbed = c_norm.copy()
        perturbed[i] -= h
        perturbed = perturbed / np.linalg.norm(perturbed)
        fi_minus = _bound_from_normalized(perturbed)
        
        if np.isfinite(fi_plus) and np.isfinite(fi_minus):
            grad[i] = (fi_plus - fi_minus) / (2 * h)
    
    return grad


def _newton_refine(c, max_iters=100, tol=1e-12):
    """
    Newton's method refinement on the unit sphere using finite-difference gradients
    and the Hessian approximated via finite differences.
    """
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return c
    c = c / norm
    
    best_c = c.copy()
    best_f = _bound_from_normalized(c)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        f = _bound_from_normalized(c)
        if not np.isfinite(f):
            break
        if f < best_f:
            best_f = f
            best_c = c.copy()
        
        # Compute gradient
        grad = _gradient_bound(c, h=1e-5)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute Hessian via finite differences of gradient
        H = np.zeros((3, 3))
        h_h = 1e-4
        for i in range(3):
            for j in range(i, 3):
                # H[i,j] = d(grad_i)/d(c_j)
                # Use central difference
                c_plus = c.copy()
                c_plus[j] += h_h
                c_plus = c_plus / np.linalg.norm(c_plus)
                gp = _gradient_bound(c_plus, h=1e-5)
                
                c_minus = c.copy()
                c_minus[j] -= h_h
                c_minus = c_minus / np.linalg.norm(c_minus)
                gm = _gradient_bound(c_minus, h=1e-5)
                
                H[i, j] = (gp[i] - gm[i]) / (2 * h_h)
                H[j, i] = H[i, j]
        
        # Solve H * delta = -grad, but project delta to tangent space
        # Tangent space: v · c = 0
        # Project: delta_tangent = delta - (delta · c) * c
        try:
            # Add small regularization for stability
            H_reg = H + 1e-8 * np.eye(3)
            delta = np.linalg.solve(H_reg, -grad)
        except np.linalg.LinAlgError:
            delta = -grad / max(grad_norm, 1e-10)
        
        # Project to tangent space
        delta = delta - np.dot(delta, c) * c
        
        # Adaptive step size
        step_size = 1.0
        for _ in range(20):
            new_c = c + step_size * delta
            new_norm = np.linalg.norm(new_c)
            if new_norm < 1e-15:
                step_size *= 0.5
                continue
            new_c = new_c / new_norm
            new_f = _bound_from_normalized(new_c)
            if np.isfinite(new_f) and new_f < f:
                c = new_c
                break
            step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return best_c


def _adaptive_local_search(c0, rng, max_iters=5000, initial_step=0.1):
    """
    Adaptive stochastic local search on the unit sphere.
    """
    norm = np.linalg.norm(c0)
    if norm < 1e-15:
        return c0
    c = c0 / norm
    
    best_c = c.copy()
    best_f = _bound_from_normalized(c)
    if not np.isfinite(best_f):
        best_f = 1e18
    
    step_size = initial_step
    accept_count = 0
    total_count = 0
    temperature = 1.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate random perturbation on the sphere
        # Use a random tangent direction
        random_vec = rng.standard_normal(3)
        # Project to tangent space
        tangent = random_vec - np.dot(random_vec, c) * c
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            tangent = np.array([1.0, 0.0, 0.0]) - c[0] * c
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = tangent / tangent_norm
        
        new_c = c + step_size * rng.standard_normal() * tangent
        new_c = new_c / np.linalg.norm(new_c)
        
        new_f = _bound_from_normalized(new_c)
        if not np.isfinite(new_f):
            new_f = 1e18
        
        total_count += 1
        
        if new_f < best_f:
            c = new_c
            best_f = new_f
            best_c = c.copy()
            accept_count += 1
            step_size *= 0.97
        else:
            delta_f = new_f - best_f
            if delta_f < 1e-10 or np.exp(-delta_f / max(temperature, 1e-15)) > rng.random():
                c = new_c
                accept_count += 1
                step_size *= 1.02
            else:
                step_size *= 1.05
        
        # Adaptive step size
        if total_count % 100 == 0:
            ratio = accept_count / max(total_count, 1)
            if ratio > 0.6:
                step_size *= 1.1
            elif ratio < 0.3:
                step_size *= 0.8
            accept_count = 0
            total_count = 0
        
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10
    
    return best_c


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit-sphere reparameterization with adaptive stochastic local search
    and Newton refinement.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_norm = np.linalg.norm(seed)
    if seed_norm > 1e-15:
        seed_dir = seed / seed_norm
    else:
        seed_dir = np.array([1.0, 0.0, 0.0])
    
    seed_bound = _compute_bound(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Structured exploration on the unit sphere
    # Fibonacci sphere for uniform coverage
    n_points = 500
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n_points):
        if time.time() > _DEADLINE - 60:
            break
        
        z = 1.0 - 2.0 * i / (n_points - 1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * i
        
        c = np.array([r * np.cos(phi), r * np.sin(phi), z])
        f = _bound_from_normalized(c)
        
        if f < best_bound:
            best_bound = f
            # Scale to original magnitude for return
            best_params = c * seed_norm
    
    # Also try the seed direction and its antipode
    for c in [seed_dir, -seed_dir]:
        f = _bound_from_normalized(c)
        if f < best_bound:
            best_bound = f
            best_params = c * seed_norm
    
    # Random additional points
    for _ in range(200):
        if time.time() > _DEADLINE - 60:
            break
        c = rng.standard_normal(3)
        c = c / np.linalg.norm(c)
        f = _bound_from_normalized(c)
        if f < best_bound:
            best_bound = f
            best_params = c * seed_norm
    
    # Phase 2: Local search from top candidates
    # Collect top candidates from exploration
    candidates = [seed_dir, -seed_dir]
    
    # Re-run exploration collecting all points and their bounds
    all_points = []
    for i in range(n_points):
        z = 1.0 - 2.0 * i / (n_points - 1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * i
        c = np.array([r * np.cos(phi), r * np.sin(phi), z])
        all_points.append(c)
    
    for _ in range(200):
        c = rng.standard_normal(3)
        c = c / np.linalg.norm(c)
        all_points.append(c)
    
    # Score all points
    scored = []
    for c in all_points:
        f = _bound_from_normalized(c)
        scored.append((f, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top 20 candidates
    top_candidates = [c for _, c in scored[:20]]
    if seed_dir not in top_candidates:
        top_candidates.append(seed_dir)
    
    # Run local search from each top candidate
    for cand in top_candidates:
        if time.time() > _DEADLINE - 20:
            break
        
        # Adaptive local search
        result_c = _adaptive_local_search(cand, rng, max_iters=3000, initial_step=0.1)
        f = _bound_from_normalized(result_c)
        
        if f < best_bound:
            best_bound = f
            best_params = result_c * seed_norm
    
    # Phase 3: Newton refinement from the best candidate
    best_c = best_params / np.linalg.norm(best_params)
    refined_c = _newton_refine(best_c, max_iters=50)
    refined_f = _bound_from_normalized(refined_c)
    
    if refined_f < best_bound:
        best_bound = refined_f
        best_params = refined_c * seed_norm
    
    # Phase 4: Final fine-tuning local search
    final_c = best_params / np.linalg.norm(best_params)
    final_c = _adaptive_local_search(final_c, rng, max_iters=5000, initial_step=0.01)
    final_f = _bound_from_normalized(final_c)
    
    if final_f < best_bound:
        best_bound = final_f
        best_params = final_c * seed_norm
    
    # One more Newton refinement
    final_c2 = best_params / np.linalg.norm(best_params)
    refined_c2 = _newton_refine(final_c2, max_iters=30)
    refined_f2 = _bound_from_normalized(refined_c2)
    
    if refined_f2 < best_bound:
        best_bound = refined_f2
        best_params = refined_c2 * seed_norm
    
    # Verify the result
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound + 1e-10:
        # Fall back to seed
        return seed.copy()
    
    return best_params
# EVOLVE-BLOCK-END
