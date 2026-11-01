"""Circle packing in a unit circle - 50 equal circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(pts):
    """Compute the packing score (minimum radius)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    # Container constraint
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(dists_center >= 1.0):
        return 0.0
    r_container = 1.0 - np.max(dists_center)
    if r_container <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_container, r_pair)

def _soft_min_obj_and_grad(pts, beta=50.0):
    """Compute soft-min objective and gradient using log-sum-exp for stability."""
    pts = np.asarray(pts, dtype=np.float64)
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    
    # Container soft constraint: penalize points outside radius 1
    container_violation = np.maximum(0, dists_center - 1.0)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = dists[dists < np.inf]
    if len(min_dists) == 0:
        return 1e6, np.zeros_like(pts)
    
    # Soft min of pairwise distances via log-sum-exp
    m = np.min(min_dists)
    exp_terms = np.exp(-beta * (min_dists - m))
    soft_min_pair = m - np.log(np.sum(exp_terms)) / beta
    r1 = soft_min_pair / 2.0
    
    # Container radius
    r2 = 1.0 - np.max(dists_center)
    
    # Soft min of r1 and r2
    m_final = min(r1, r2)
    arr = np.array([r1, r2])
    exp_final = np.exp(-beta * (arr - m_final))
    soft_min_final = m_final - np.log(np.sum(exp_final)) / beta
    
    # Penalty for container violation
    penalty = 100.0 * np.sum(container_violation**2)
    
    obj = -(soft_min_final - penalty)
    
    # Gradient computation (numerical, central differences for accuracy)
    grad = np.zeros_like(pts)
    eps = 1e-7
    for i in range(N):
        for j in range(2):
            pts_p = pts.copy()
            pts_p[i, j] += eps
            obj_p = _soft_min_obj_and_grad(pts_p, beta)[0]
            pts_m = pts.copy()
            pts_m[i, j] -= eps
            obj_m = _soft_min_obj_and_grad(pts_m, beta)[0]
            grad[i, j] = (obj_p - obj_m) / (2 * eps)
    
    return obj, grad

def _force_directed_relax(pts, n_steps=5000, dt=0.001, deadline=None):
    """
    Rigid-body dynamics with velocity-Verlet integration.
    Strong short-range repulsive forces + confining harmonic potential.
    Adaptive exponentially decaying damping.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Initial velocities zero
    v = np.zeros_like(pts)
    
    # Force parameters
    repulsion_strength = 50.0
    repulsion_range = 0.5  # only interact within this range
    container_k = 50.0  # harmonic confinement stiffness
    damping_init = 0.98
    damping_decay = 0.99999  # exponential decay rate per step
    
    damping = damping_init
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsive forces (short-range)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random nudge for coincident points
                    d = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(d)
                if dist < repulsion_range:
                    # Strong short-range repulsion: F ~ 1/dist^2 capped
                    f_mag = repulsion_strength / (dist * dist + 1e-6)
                    f_mag = min(f_mag, 100.0)  # cap force magnitude
                    f_dir = d / dist
                    forces[i] += f_mag * f_dir
                    forces[j] -= f_mag * f_dir
        
        # Container confinement: harmonic potential pushing inward
        dists_center = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if dists_center[i] > 0.8:
                # Force proportional to distance beyond 0.8, directed inward
                excess = dists_center[i] - 0.8
                f_mag = container_k * excess
                if dists_center[i] > 1e-10:
                    f_dir = -pts[i] / dists_center[i]
                    forces[i] += f_mag * f_dir
                else:
                    forces[i] += f_mag * np.array([1.0, 0.0])
        
        # Velocity-Verlet integration
        # Step 1: Update position by half step with current velocity
        pts_half = pts + 0.5 * dt * v
        
        # Compute forces at half-step positions
        forces_half = np.zeros_like(pts_half)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts_half[i] - pts_half[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(d)
                if dist < repulsion_range:
                    f_mag = repulsion_strength / (dist * dist + 1e-6)
                    f_mag = min(f_mag, 100.0)
                    f_dir = d / dist
                    forces_half[i] += f_mag * f_dir
                    forces_half[j] -= f_mag * f_dir
        
        dists_half = np.linalg.norm(pts_half, axis=1)
        for i in range(n):
            if dists_half[i] > 0.8:
                excess = dists_half[i] - 0.8
                f_mag = container_k * excess
                if dists_half[i] > 1e-10:
                    f_dir = -pts_half[i] / dists_half[i]
                    forces_half[i] += f_mag * f_dir
                else:
                    forces_half[i] += f_mag * np.array([1.0, 0.0])
        
        # Step 2: Update velocity
        v_new = v + dt * forces_half
        v_new *= damping  # Apply damping
        
        # Step 3: Update position by second half step
        pts_new = pts_half + 0.5 * dt * v_new
        
        # Project to stay inside unit circle (hard constraint)
        dists_new = np.linalg.norm(pts_new, axis=1)
        mask = dists_new > 0.999
        if np.any(mask):
            pts_new[mask] *= 0.999 / dists_new[mask]
            # Zero out radial velocity component for reflected points
            for i in np.where(mask)[0]:
                radial_dir = pts[i] / dists_center[i] if dists_center[i] > 1e-10 else np.array([1.0, 0.0])
                v_radial = np.dot(v_new[i], radial_dir)
                if v_radial > 0:
                    v_new[i] -= v_radial * radial_dir
        
        pts = pts_new
        v = v_new
        damping *= damping_decay
        
        # Track best
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
    
    return best_pts, best_score

def _hex_lattice_init():
    """Generate initial points using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _ring_init():
    """Generate initial points on concentric rings matching known optimal structure."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Known structure: rings with approximately 6, 12, 18, 13 points (total 50)
    ring_configs = [
        (6, 0.25),
        (12, 0.45),
        (18, 0.65),
        (13, 0.80),
    ]
    for n_on_ring, ring_radius in ring_configs:
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        ring_pts = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        pts.append(ring_pts)
    pts = np.vstack(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        extra = N - len(pts)
        angles = np.random.uniform(0, 2 * np.pi, extra)
        radii = np.sqrt(np.random.uniform(0.5, 0.95, extra))
        extra_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = np.vstack([pts, extra_pts])
    return pts

def _random_init():
    """Generate random initial points inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _soft_min_optimize(pts, deadline, beta_schedule=None):
    """
    Multi-stage soft-min optimization with beta annealing.
    Uses L-BFGS-B with increasing beta for precision.
    """
    if beta_schedule is None:
        beta_schedule = [50, 200, 500, 1000, 5000, 10000, 50000, 100000]
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        def neg_obj(x, b=beta):
            obj, _ = _soft_min_obj_and_grad(x, b)
            return obj
        
        try:
            res = minimize(
                neg_obj,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            candidate = res.x
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def _full_optimize(init_pts, deadline):
    """Full optimization pipeline: force-directed relax -> soft-min optimize."""
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # Stage 1: Force-directed relaxation (dynamics-based, escapes local minima)
    if time.time() < deadline - 30.0:
        fd_pts, fd_score = _force_directed_relax(init_pts, n_steps=3000, dt=0.002, deadline=deadline)
        if fd_score > best_score:
            best_score = fd_score
            best_pts = fd_pts.copy()
    
    # Stage 2: Soft-min optimization with beta annealing
    if time.time() < deadline - 10.0:
        sm_pts, sm_score = _soft_min_optimize(best_pts, deadline)
        if sm_score > best_score:
            best_score = sm_score
            best_pts = sm_pts.copy()
    
    # Stage 3: Final high-precision polish
    if time.time() < deadline - 5.0 and best_score > 0:
        try:
            def neg_obj_final(x):
                obj, _ = _soft_min_obj_and_grad(x, 100000)
                return obj
            
            res = minimize(
                neg_obj_final,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            candidate = res.x
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety (150s limit)
    
    # Generate multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Try to improve each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        optimized_pts, opt_score = _full_optimize(init_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = optimized_pts.copy()
    
    # Final global polish on the best configuration
    if time.time() < deadline - 5.0 and best_score > 0:
        try:
            def neg_obj_final(x):
                obj, _ = _soft_min_obj_and_grad(x, 100000)
                return obj
            
            res = minimize(
                neg_obj_final,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            candidate = res.x
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
        except Exception:
            pass
    
    # Ensure all points are strictly inside the unit circle
    dists_center = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = dists_center >= 0.9999
    if np.any(mask):
        scale = 0.9999 / dists_center[mask]
        best_pts[mask] *= scale[:, np.newaxis]
    
    return best_pts
