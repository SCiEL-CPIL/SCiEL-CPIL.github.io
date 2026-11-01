"""Uncertainty bound minimization via multi-start Nelder-Mead on spherical coordinates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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


def _objective_sphere(params):
    """
    Objective function for optimization in spherical coordinates.
    params: [theta1, theta2, log_r]
    """
    t1, t2, log_r = params
    r = np.exp(log_r)
    st2 = np.sin(t2)
    a0 = r * st2 * np.cos(t1)
    a4 = r * st2 * np.sin(t1)
    a8 = r * np.cos(t2)
    
    # Handle potential numerical issues with sin/cos near 0/pi
    if not (np.isfinite(a0) and np.isfinite(a4) and np.isfinite(a8)):
        return 1e18
        
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses multi-start Nelder-Mead optimization on spherical coordinates.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    # Convert seed to spherical coordinates for initial guess
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        theta1_seed = np.arctan2(seed[1], seed[0])
        theta2_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
        log_r_seed = np.log(r_seed)
    else:
        theta1_seed = 0.0
        theta2_seed = 0.0
        log_r_seed = 0.0
        
    # Generate a set of diverse starting points on the sphere
    # We will sample from a distribution that covers the sphere and a range of radii
    rng = np.random.default_rng(42)
    
    # Radii to explore: log-uniform from 0.1 to 100
    log_radii = np.linspace(-2.3, 4.6, 50) # ~0.1 to ~100
    
    # Starting points: 
    # 1. The seed
    # 2. Random points on the sphere with various radii
    # 3. Points near the seed
    
    start_points = []
    
    # Add seed
    start_points.append([theta1_seed, theta2_seed, log_r_seed])
    
    # Add random points
    for _ in range(200):
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(0, np.pi)
        lr = rng.choice(log_radii)
        start_points.append([t1, t2, lr])
        
    # Add points near seed
    for _ in range(50):
        dt1 = rng.normal(0, 0.5)
        dt2 = rng.normal(0, 0.5)
        dlr = rng.normal(0, 0.5)
        t1 = theta1_seed + dt1
        t2 = np.clip(theta2_seed + dt2, 0.01, np.pi - 0.01)
        lr = log_r_seed + dlr
        start_points.append([t1, t2, lr])

    # Sort start points by initial objective value to prioritize good basins
    # We don't want to sort strictly because we might miss a good basin that is slightly worse initially
    # but leads to a better minimum. However, for time efficiency, we can prioritize.
    # Let's just run them all if time permits, or the first N best.
    
    # To ensure we don't timeout, we'll limit the number of optimizations
    # Nelder-Mead is fast, but let's say 100 optimizations max.
    
    # Shuffle start points to avoid bias
    indices = rng.permutation(len(start_points))
    
    time_limit_per_opt = 0.5 # seconds
    total_time = 0.0
    
    for idx in indices:
        if time.time() > _DEADLINE - 5:
            break
            
        start = start_points[idx]
        
        # Ensure valid bounds for t2
        start[1] = np.clip(start[1], 0.01, np.pi - 0.01)
        
        # Run Nelder-Mead
        # Bounds:
        # t1: (-pi, pi) -> periodic, Nelder-Mead doesn't handle periodicity well directly, 
        # but since we are just minimizing, it should work if the landscape is smooth enough.
        # However, t1 is an angle. We can wrap it.
        # t2: (0, pi)
        # log_r: (-inf, inf) but practically (-5, 5)
        
        res = minimize(_objective_sphere, start, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
        
        if res.success and res.fun < best_bound:
            best_bound = res.fun
            # Convert back to cartesian
            t1_final = res.x[0]
            t2_final = res.x[1]
            log_r_final = res.x[2]
            
            r_final = np.exp(log_r_final)
            st2 = np.sin(t2_final)
            best_params = np.array([
                r_final * st2 * np.cos(t1_final),
                r_final * st2 * np.sin(t1_final),
                r_final * np.cos(t2_final)
            ])

    # Final verification
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
