"""Combined program: rigid-body dynamics initialization + log-sum-exp smooth energy with binary search."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

# EVOLVE-BLOCK-START

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)


def _smooth_energy(P, r, beta=80.0):
    """
    Smooth log-sum-exp energy function for differentiable optimization.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    
    def _lse(v, beta):
        v_scaled = beta * v
        v_max = np.max(v_scaled)
        if not np.isfinite(v_max) or v_max == -np.inf:
            return 0.0
        return (1.0 / beta) * (v_max + np.log(np.sum(np.exp(v_scaled - v_max))))
    
    E_cc = _lse(violations_cc, beta)
    E_cb = _lse(violations_cb, beta)
    
    return E_cc + E_cb


def _smooth_energy_grad(P, r, beta=80.0):
    """
    Gradient of the smooth log-sum-exp energy.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    v_scaled_cc = beta * violations_cc
    v_max_cc = np.max(v_scaled_cc)
    if np.isfinite(v_max_cc) and v_max_cc != -np.inf:
        weights_cc = np.exp(v_scaled_cc - v_max_cc)
        weights_cc /= np.sum(weights_cc)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        
        W_cc = weights_cc * inv_dist
        sum_W = np.sum(W_cc, axis=1)
        term1 = (P.T @ W_cc).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term2 - term1
    
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    v_scaled_cb = beta * violations_cb
    v_max_cb = np.max(v_scaled_cb)
    if np.isfinite(v_max_cb) and v_max_cb != -np.inf:
        weights_cb = np.exp(v_scaled_cb - v_max_cb)
        weights_cb /= np.sum(weights_cb)
        
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        
        grad += (weights_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _optimize_positions(P_init, r_target, time_limit=5.0, beta=80.0):
    """
    Optimize positions P to minimize smooth energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _check_feasible(P, r, tol=1e-7):
    """Check if packing with radius r is feasible."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    dist_origin = np.linalg.norm(P, axis=1)
    max_origin = np.max(dist_origin)
    
    return (min_pair >= 2.0 * r - tol) and (max_origin + r <= 1.0 + tol)


def _get_hex_initial_points():
    """Initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P


def _dynamics_settle(P_init, r_target, time_limit=5.0):
    """
    Rigid-body dynamics with Lennard-Jones-like repulsion and confining well.
    Uses velocity-Verlet with adaptive damping to settle into low-energy config.
    Returns (P_settled, energy).
    """
    P = P_init.copy()
    V = np.zeros_like(P)
    
    # LJ parameters scaled to our problem
    sigma = 2.0 * r_target  # equilibrium distance
    epsilon = 1.0
    k_well = 50.0  # harmonic well strength
    damping = 0.95
    
    dt = 0.001
    max_steps = int(time_limit / dt)
    max_steps = min(max_steps, 50000)
    
    def _forces(pos):
        F = np.zeros_like(pos)
        # Pairwise LJ repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-6:
                    continue
                # LJ: 4*eps*((sig/dist)^12 - (sig/dist)^6)
                sr6 = (sigma / dist) ** 6
                f_mag = 24 * epsilon * (2 * sr6 ** 2 - sr6) / dist
                f_vec = f_mag * d / dist
                F[i] += f_vec
                F[j] -= f_vec
        # Confining well
        for i in range(N):
            r_i = np.linalg.norm(pos[i])
            if r_i > 1.0 - r_target:
                # Push inward
                F[i] -= k_well * pos[i]
            elif r_i > 0.01:
                # Gentle centering
                F[i] -= 0.5 * pos[i]
            else:
                F[i] -= 0.5 * pos[i]
        return F
    
    F = _forces(P)
    
    best_P = P.copy()
    best_score = _compute_score(P)
    
    for step in range(max_steps):
        # Velocity Verlet
        P_half = P + 0.5 * dt * V
        V_half = V + 0.5 * dt * F
        P = P_half + 0.5 * dt * V_half
        F_new = _forces(P)
        V = (V_half + 0.5 * dt * F_new) * damping
        
        # Check for convergence
        if step % 1000 == 0:
            score_now = _compute_score(P)
            if score_now > best_score:
                best_score = score_now
                best_P = P.copy()
            if np.linalg.norm(F_new) < 1e-8:
                break
    
    return best_P, _smooth_energy(best_P, r_target, beta=80.0)


def _optimize_ring_structure(radii0, counts, angles_offset0=None, time_limit=20.0):
    """Optimize ring radii and angles for fixed counts using score maximization."""
    if angles_offset0 is None:
        angles_offset0 = [0.0] * len(counts)
    
    n_rings = len(counts)
    
    def build_and_score(params):
        radii = np.maximum(params[:n_rings], 0.0)
        angles = params[n_rings:]
        P = _get_ring_initialization(np.concatenate([radii, angles]), counts)
        return _compute_score(P)
    
    x0 = np.concatenate([radii0, angles_offset0])
    
    t_start = time.time()
    try:
        res = minimize(
            lambda x: -build_and_score(x),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if res.success or res.fun < -build_and_score(x0) + 1e-10:
            radii_opt = np.maximum(res.x[:n_rings], 0.0)
            angles_opt = res.x[n_rings:]
            return radii_opt, angles_opt
    except Exception:
        pass
    
    return radii0, angles_offset0


def _global_search_ring_params(ring_counts, time_limit=15.0):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    bounds = [(0.0, 0.9)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
        
    start_time = time.time()
    
    try:
        res = differential_evolution(neg_score, bounds, maxiter=30, popsize=12, 
                                     tol=1e-5, seed=42, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        return best_P
    except Exception:
        # Fallback to hex
        return _get_hex_initial_points()


def _perturb(P, sigma=0.01):
    """Add small Gaussian perturbation to escape local minima."""
    return P + sigma * np.random.randn(*P.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Define ring structures for N=50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
        [1, 6, 12, 11, 20],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # Use dynamics to settle from hex init
    r_init = score_best
    P_dyn, E_dyn = _dynamics_settle(P_best, r_init, time_limit=min(8.0, deadline - (time.time() - start_time) - 60))
    score_dyn = _compute_score(P_dyn)
    if score_dyn > score_best:
        P_best = P_dyn.copy()
        score_best = score_dyn
    
    # For each ring structure, optimize and refine
    for counts in ring_structures:
        if time.time() > deadline - 30.0:
            break
        
        n_rings = len(counts)
        
        # Initial radii: evenly spaced
        if n_rings == 1:
            radii0 = np.array([0.0])
        else:
            radii0 = np.linspace(0.05, 0.85, n_rings)
            radii0[0] = 0.0
        
        angles0 = [0.0] * n_rings
        
        # Optimize ring structure parameters
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(10.0, deadline - (time.time() - start_time) - 40)
        )
        
        # Build centers from optimized ring structure
        P_candidate = _get_ring_initialization(
            np.concatenate([radii_opt, angles_opt]), counts
        )
        
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate.copy()
            score_best = score_candidate
        
        # Use dynamics to settle further
        r_target = max(score_candidate, score_best) * 0.99
        P_dyn2, E_dyn2 = _dynamics_settle(P_candidate, r_target, time_limit=min(4.0, deadline - (time.time() - start_time) - 35))
        score_dyn2 = _compute_score(P_dyn2)
        if score_dyn2 > score_best:
            P_best = P_dyn2.copy()
            score_best = score_dyn2
        
        # Binary search on r with energy minimization
        r_low = score_candidate * 0.98
        r_high = min(score_candidate * 1.1, 0.135)
        
        # Ensure r_low is feasible by optimizing
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=3.0)
        if E < 1e-5:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
            r_low = score
        
        # Binary search iterations
        for iteration in range(6):
            if time.time() > deadline - 20.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            
            best_P_for_r = None
            best_energy = np.inf
            
            n_restarts = 2 if iteration < 3 else 3
            
            for restart in range(n_restarts):
                if time.time() > deadline - 20.0:
                    break
                
                if restart == 0:
                    P_init = P_best.copy()
                else:
                    sigma = 0.003 * (restart + 1)
                    P_init = _perturb(P_best, sigma)
                
                P_opt, E = _optimize_positions(P_init, r_mid, time_limit=3.0, beta=80.0)
                
                if E < best_energy:
                    best_energy = E
                    best_P_for_r = P_opt
            
            if best_P_for_r is None:
                break
            
            if _check_feasible(best_P_for_r, r_mid):
                r_low = r_mid
                score = _compute_score(best_P_for_r)
                if score > score_best:
                    P_best = best_P_for_r.copy()
                    score_best = score
            else:
                r_high = r_mid
    
    # Final polish with higher beta
    if time.time() < deadline - 10.0:
        r_target = score_best
        for restart in range(4):
            if time.time() > deadline - 5.0:
                break
            
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = _perturb(P_best, 0.001 * restart)
            
            P_opt, E = _optimize_positions(P_init, r_target, time_limit=2.5, beta=150.0)
            
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
    
    # One more dynamics pass from best
    if time.time() < deadline - 5.0:
        P_final, E_final = _dynamics_settle(P_best, score_best * 0.99, time_limit=min(3.0, deadline - time.time() - 2))
        score_final = _compute_score(P_final)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
    
    return P_best

# EVOLVE-BLOCK-END
