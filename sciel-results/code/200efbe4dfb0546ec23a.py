"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius.
    Uses log-sum-exp to approximate the minimum of:
    - distances to boundaries
    - half pairwise distances
    """
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    b_terms = np.concatenate([b_left, b_right, b_bottom, b_top])
    all_terms = np.concatenate([b_terms, p_terms])
    
    m = np.min(all_terms)
    # Clip to prevent underflow in exp for very large beta
    # If all_terms - m is large, exp(-beta * large) -> 0
    # We only care about terms close to the minimum
    shifted = all_terms - m
    # To avoid underflow issues and speed up, we can ignore terms > 100/beta
    # But simple clipping is safer
    shifted = np.clip(shifted, 0, 50.0 / beta) 
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    return soft_min_val


def _objective(c_flat, beta):
    """Objective function for minimization: -soft_min_radius."""
    c = c_flat.reshape(-1, 2)
    # Ensure points are within bounds for the objective to be well-behaved?
    # L-BFGS-B handles bounds, but the objective should be smooth.
    r = _soft_min_radius(c, beta)
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a few spacings
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            def obj(params):
                return _objective(params, beta)
            
            maxiter = 80 if beta < 100 else 150
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    
    T_start = 0.001
    T_end = 1e-8
    n_steps = 5000
    
    def _build_clusters(c, r_target):
        """Group circles into clusters based on proximity."""
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        return list(clusters.values())
    
    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 500
    
    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        if np.random.random() < 0.6 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    
    return best_c, best_r


def _md_relax(c, r_target, time_limit=2.0):
    """
    Relaxes the configuration using Molecular Dynamics to resolve overlaps
    for circles of radius r_target.
    Returns the relaxed configuration.
    """
    c = c.copy()
    v = np.zeros_like(c)
    
    # Parameters
    dt = 0.001
    damping = 0.99
    k_rep = 100.0 # Spring constant for repulsion
    k_wall = 1000.0 # Spring constant for boundaries
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        F = np.zeros_like(c)
        
        # Pairwise forces using broadcasting (N=200 is small enough for 200x200 matrix)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d_mat, 1.0)
        
        # Unit vectors
        unit_vec = diff / d_mat[:, :, np.newaxis]
        
        # Overlap
        overlap = np.maximum(0.0, 2*r_target - d_mat)
        
        # Force contribution from each pair j on i
        force_mat = k_rep * overlap[:, :, np.newaxis] * unit_vec
        F = np.sum(force_mat, axis=1)
        
        # Boundary forces
        # Push back if close to boundary (distance < r)
        F[:, 0] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 0])
        F[:, 0] -= k_wall * np.maximum(0.0, c[:, 0] - (0.5 - r_target))
        F[:, 1] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 1])
        F[:, 1] -= k_wall * np.maximum(0.0, c[:, 1] - (0.5 - r_target))
        
        # Velocity Verlet (simplified: Euler with damping)
        v = v * damping + F * dt
        c = c + v * dt
        
        # Hard clip to ensure they don't fly out
        c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Adaptive time step
        max_v = np.max(np.linalg.norm(v, axis=1))
        if max_v > 0.05:
            dt *= 0.5
        elif max_v < 0.01 and dt < 0.005:
            dt *= 1.5

    return c


def _md_radius_expansion(c0, r0, max_time=30.0):
    """
    Iteratively increase the radius and relax using MD.
    """
    start_time = time.time()
    c = c0.copy()
    r = r0
    
    # We want to expand r. 
    # Start with a small increment.
    # If the config is already valid for r, we can try to increase it.
    # We will try to increase r by small steps, running MD relaxation at each step.
    
    # Determine a max possible r (rough upper bound)
    # Area argument: 200 * pi * r^2 <= 1 => r <= 1/sqrt(200*pi) ~ 0.0398
    # But packing efficiency is ~0.9, so r_max ~ 0.0398 * sqrt(0.9) ~ 0.0378
    # Current best is ~0.0366. We can try to push towards 0.038.
    
    r_max_try = 0.0385
    current_r = r
    
    # First, ensure we are relaxed at current r
    c = _md_relax(c, current_r, time_limit=min(5.0, max_time - 2.0))
    current_r = _compute_radius(c)
    
    # Expansion loop
    # Step size decreases as we get closer to the limit
    step_size = 0.0002 
    max_steps = 200 # Safety limit on number of expansions
    
    for i in range(max_steps):
        if time.time() > start_time + max_time - 5.0:
            break
            
        # Try to increase radius
        new_r_target = current_r + step_size
        
        # If we hit the theoretical max or no improvement, stop
        if new_r_target > r_max_try:
            break
            
        # Relax at new radius
        c = _md_relax(c, new_r_target, time_limit=min(2.0, max_time - (time.time() - start_time) - 2.0))
        
        # Compute actual max radius of this configuration
        actual_r = _compute_radius(c)
        
        # If we couldn't achieve the target radius, we might have made it worse?
        # Actually, if MD fails to resolve overlaps, the circles will overlap, 
        # and _compute_radius will return a smaller value (min pair dist / 2).
        # We should only accept if actual_r >= current_r (or slightly less if we want to explore, 
        # but for expansion we want monotonic improvement or at least non-degradation).
        
        if actual_r > current_r:
            current_r = actual_r
            # Reduce step size for finer control
            step_size *= 0.95
        else:
            # If we didn't improve, maybe try a smaller step?
            # Or just stop if we are stuck.
            # Let's try a few retries with smaller steps
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return c, current_r


def _slsqp_polish(c0, r0, max_time=10.0):
    """
    Final high-precision polish using SLSQP.
    """
    start_time = time.time()
    
    # Define constraints for SLSQP
    # 1. Boundary constraints: 0.5 - |x| >= r, 0.5 - |y| >= r
    # 2. Pairwise constraints: |c_i - c_j| >= 2r
    
    # Since SLSQP works on fixed variables, we fix the radius r to r0 (or slightly less) 
    # and optimize positions to satisfy constraints.
    # Actually, we want to maximize r. 
    # A common trick is to fix r to a value slightly below the current best 
    # and see if we can satisfy the constraints. If yes, we can try a higher r.
    # However, SLSQP is a constrained optimizer. 
    # Let's try to maximize r by using a penalty method or just optimizing positions 
    # for a fixed r_target.
    
    # Better approach: 
    # Use SLSQP to minimize 0 subject to constraints:
    # 1. x_i in [-0.5+r, 0.5-r]
    # 2. ||c_i - c_j|| >= 2r
    # If we can find a feasible solution for a given r, then r is achievable.
    # We can do a binary search on r.
    
    # But SLSQP is slow for N=200 with O(N^2) constraints.
    # Instead, let's use SLSQP to polish the local optimum by minimizing 
    # the sum of squared violations of the constraints for a fixed r_target = r0.
    # If the sum of violations is 0, the config is valid for r0.
    
    r_target = r0
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Sum of squared violations
        # Boundary violations
        b_viol = np.maximum(0, (r_target - (0.5 - np.abs(c[:, 0]))))
        b_viol += np.maximum(0, (r_target - (0.5 - np.abs(c[:, 1]))))
        
        # Pairwise violations
        d = pdist(c, metric='euclidean')
        p_viol = np.maximum(0, 2*r_target - d)
        
        return np.sum(b_viol**2) + np.sum(p_viol**2)
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        
        if r_new > r0:
            return c_new, r_new
        else:
            return c0, r0
            
    except Exception:
        return c0, r0


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 60:
        remaining = deadline - time.time() - 60
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 50:
        remaining = max(10.0, deadline - time.time() - 50)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 4. MD Radius Expansion: The key improvement
    if best_c is not None and time.time() < deadline - 20:
        remaining = max(10.0, deadline - time.time() - 20)
        c_md, r_md = _md_radius_expansion(best_c, best_radius, max_time=min(30.0, remaining))
        if r_md > best_radius:
            best_radius = r_md
            best_c = c_md.copy()
    
    # 5. Final SLSQP Polish
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(10.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
