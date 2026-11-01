"""Analytic uncertainty bound with sphere-constrained stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 200


def _bound_value(params):
    """
    Compute uncertainty product (scalar only, no gradient).
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) / (2*n) * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _normalize(c):
    """Normalize to unit sphere."""
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return c
    return c / norm


def _sphere_to_params(c, scale):
    """Convert unit sphere vector to raw parameters."""
    return c * scale


def construct():
    rng = np.random.default_rng(42)
    
    # Seed point
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_norm = np.linalg.norm(seed)
    seed_c = seed / seed_norm if seed_norm > 1e-15 else seed
    
    best_val = _bound_value(seed)
    best_params = seed.copy()
    best_c = seed_c.copy()
    best_scale = seed_norm
    
    # Adaptive step size for sphere search
    step_size = 0.05
    no_improve = 0
    max_no_improve = 50
    
    # Initial diverse starting points on sphere
    starts = [seed_c]
    for i in range(20):
        c = rng.normal(0, 1, 3)
        c = _normalize(c)
        # Try a few scales
        for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
            params = _sphere_to_params(c, scale)
            val = _bound_value(params)
            if val < best_val:
                best_val = val
                best_params = params.copy()
                best_c = c.copy()
                best_scale = scale
    
    # Stochastic local search on sphere with adaptive step size
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate random direction on sphere (tangent space)
        rand_vec = rng.normal(0, 1, 3)
        # Project to tangent space of current point
        tangent = rand_vec - np.dot(rand_vec, best_c) * best_c
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            tangent = rand_vec
            tangent_norm = np.linalg.norm(tangent)
        tangent = tangent / tangent_norm
        
        # Try several step sizes (adaptive)
        candidates = []
        for factor in [step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25]:
            new_c = _normalize(best_c + factor * tangent)
            # Also try varying the scale slightly
            for scale_factor in [1.0, 1.1, 0.9, 1.2, 0.8]:
                new_scale = best_scale * scale_factor
                new_params = _sphere_to_params(new_c, new_scale)
                val = _bound_value(new_params)
                candidates.append((val, new_c, new_scale, new_params))
        
        # Pick the best candidate
        candidates.sort(key=lambda x: x[0])
        best_candidate = candidates[0]
        
        if best_candidate[0] < best_val:
            best_val = best_candidate[0]
            best_c = best_candidate[1].copy()
            best_scale = best_candidate[2]
            best_params = best_candidate[3].copy()
            no_improve = 0
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.1, 0.5)
        else:
            no_improve += 1
            # Decrease step size on stagnation
            step_size = max(step_size * 0.95, 0.001)
            
            # Occasionally restart from a random point
            if no_improve > max_no_improve:
                c = rng.normal(0, 1, 3)
                c = _normalize(c)
                for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    params = _sphere_to_params(c, scale)
                    val = _bound_value(params)
                    if val < best_val:
                        best_val = val
                        best_params = params.copy()
                        best_c = c.copy()
                        best_scale = scale
                        no_improve = 0
                        break
                no_improve = 0
    
    # Final local refinement with very small steps
    if time.time() < _DEADLINE - 5:
        small_step = 0.001
        for it in range(1000):
            if time.time() > _DEADLINE - 3:
                break
            rand_vec = rng.normal(0, 1, 3)
            tangent = rand_vec - np.dot(rand_vec, best_c) * best_c
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-15:
                continue
            tangent = tangent / tangent_norm
            new_c = _normalize(best_c + small_step * tangent)
            new_params = _sphere_to_params(new_c, best_scale)
            val = _bound_value(new_params)
            if val < best_val:
                best_val = val
                best_c = new_c.copy()
                best_params = new_params.copy()
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
