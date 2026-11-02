"""Combined program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
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


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 80.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, selecting most central points."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.050, 0.080, 40):
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
            # Select most central points to maximize boundary distance
            dist_to_bnd = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
            idx = np.argsort(dist_to_bnd)[::-1]
            c = points[idx[:N]].copy()
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


def _logdet_fista(c0, max_time=40.0):
    """
    Optimize using Accelerated Proximal Gradient (FISTA) with Log-Determinant barrier.
    Objective: Minimize -soft_min(distances) - gamma * log_det(Gram_matrix)
    """
    start_time = time.time()
    c = c0.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # FISTA parameters
    L = 50.0
    t_k = 1.0
    x_k = c.copy()
    x_prev = c.copy()
    
    # Annealing beta and gamma
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for idx, beta in enumerate(betas):
        if time.time() > start_time + max_time - 5:
            break
        
        gamma = gammas[idx]
        x_k = best_c.copy()
        x_prev = best_c.copy()
        t_k = 1.0
        
        # Run FISTA for a fixed number of steps or time
        steps_per_beta = 100
        for k in range(steps_per_beta):
            if time.time() > start_time + max_time - 5:
                break
            
            # 1. Compute Gradient
            # Soft-min part
            D = cdist(x_k, x_k, metric='euclidean')
            np.fill_diagonal(D, 1.0)
            d_min = np.min(D)
            
            # Weights for soft-min
            shifted = D - d_min
            shifted = np.clip(shifted, 0, 100.0/beta)
            weights = np.exp(-beta * shifted)
            np.fill_diagonal(weights, 0.0)
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.zeros_like(weights)
            
            # Gradient of soft-min S: grad_S = - (A X - X (A 1)^T) where A_ij = w_ij / D_ij
            A = weights / D
            np.fill_diagonal(A, 0.0)
            row_sums = np.sum(A, axis=1)
            grad_soft = A @ x_k - x_k * row_sums[:, np.newaxis]
            
            # Log-Det part
            x_c = x_k - np.mean(x_k, axis=0, keepdims=True)
            G = x_c.T @ x_c
            G += 1e-8 * np.eye(2)
            G_inv = np.linalg.inv(G)
            grad_logdet = 2.0 * (x_c @ G_inv)
            
            # Total Gradient for Objective J = -S - gamma * L
            grad_J = -grad_soft - gamma * grad_logdet
            
            # 2. Gradient Step
            step_size = 1.0 / L
            y_k = x_k - step_size * grad_J
            
            # 3. Projection onto box [-0.5, 0.5]
            y_k = np.clip(y_k, -0.5 + 1e-12, 0.5 - 1e-12)
            
            # 4. FISTA Update
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            x_next = y_k + ((t_k - 1.0) / t_next) * (y_k - x_k)
            
            x_prev = x_k
            x_k = x_next
            t_k = t_next
            
            # Evaluate
            r_curr = _compute_radius(x_k)
            if r_curr > best_r:
                best_r = r_curr
                best_c = x_k.copy()
                
        # After each beta, refine with a few standard steps or just keep best
        x_k = best_c.copy()
        x_prev = best_c.copy()
        t_k = 1.0

    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=35.0):
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
    n_steps = 6000

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
    for circles of radius r_target. Uses cKDTree for efficient neighbor search.
    """
    c = c.copy()
    v = np.zeros_like(c)

    dt = 0.001
    damping = 0.99
    k_rep = 100.0
    k_wall = 1000.0
    search_radius = 2.5 * r_target

    start_time = time.time()

    while time.time() - start_time < time_limit:
        F = np.zeros_like(c)

        # Build KDTree for neighbor search
        tree = cKDTree(c)

        for i in range(N):
            neighbors = tree.query_ball_point(c[i], search_radius)
            for j in neighbors:
                if i >= j:
                    continue
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                overlap = 2 * r_target - d
                if overlap > 0:
                    unit_vec = diff / d
                    force = k_rep * overlap * unit_vec
                    F[i] += force
                    F[j] -= force

        # Wall forces
        F[:, 0] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 0])
        F[:, 0] -= k_wall * np.maximum(0.0, c[:, 0] - (0.5 - r_target))
        F[:, 1] += k_wall * np.maximum(0.0, (-0.5 + r_target) - c[:, 1])
        F[:, 1] -= k_wall * np.maximum(0.0, c[:, 1] - (0.5 - r_target))

        v = v * damping + F * dt
        c = c + v * dt
        c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)

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
    current_r = r0

    r_max_try = 0.0395

    c = _md_relax(c, current_r, time_limit=min(5.0, max_time - 2.0))
    current_r = _compute_radius(c)

    step_size = 0.0002
    max_steps = 200

    for i in range(max_steps):
        if time.time() > start_time + max_time - 5.0:
            break

        new_r_target = current_r + step_size

        if new_r_target > r_max_try:
            break

        remaining = max(0.5, max_time - (time.time() - start_time) - 2.0)
        c = _md_relax(c, new_r_target, time_limit=min(2.0, remaining))
        actual_r = _compute_radius(c)

        if actual_r > current_r:
            current_r = actual_r
            step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break

    return c, current_r


def _slsqp_polish(c0, r0, max_time=15.0):
    """
    Final high-precision polish using SLSQP with active-set refinement.
    """
    start_time = time.time()
    c = c0.copy()
    r = r0

    # Iterative active-set refinement with decreasing tolerance
    tolerances = [0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    for tol in tolerances:
        if time.time() > start_time + max_time - 3:
            break

        r_target = r

        def objective(x):
            c_local = x.reshape(-1, 2)
            # Boundary violations
            b_viol = np.maximum(0, (r_target - (0.5 - np.abs(c_local[:, 0]))))
            b_viol += np.maximum(0, (r_target - (0.5 - np.abs(c_local[:, 1]))))

            # Pairwise violations
            d = pdist(c_local, metric='euclidean')
            p_viol = np.maximum(0, 2 * r_target - d)

            return np.sum(b_viol**2) + np.sum(p_viol**2)

        try:
            result = minimize(
                objective,
                c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-18}
            )

            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)

            if r_new > r:
                c = c_new
                r = r_new
        except Exception:
            pass

    return c, r


def _qp_optimize_v2(points, max_iter=8000, lr_init=0.003, lr_decay=0.9995, lr_min=0.00002, r_step=0.00008):
    """
    Improved QP optimizer with:
    - Smaller initial learning rate for stability
    - Slower decay for more iterations at useful learning rates
    - Smaller radius increment for finer convergence
    """
    n = len(points)
    r = _compute_radius(points)
    points = points.copy()
    
    best_points = points.copy()
    best_r = r
    
    for it in range(max_iter):
        lr = max(lr_init * (lr_decay ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        
        # --- Pairwise gradient contribution ---
        # Use a smooth approximation: weight based on proximity to 2r
        # For pairs closer than 2r, push apart with force proportional to violation
        threshold = 2.0 * r
        mask = dists < threshold
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        # Violation amount (how much below 2r)
        viol = np.where(mask, threshold - dists, 0.0)
        
        # Use linear weight for smoother gradients
        weights = np.zeros_like(dists)
        weights[mask] = viol[mask] / safe_dists[mask]
        
        # Gradient from pairwise constraints
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # --- Boundary gradient contribution ---
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Active boundary constraint per point
        active_x = bx <= by
        
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        # Gradient: push away from boundary
        grad_x = viol_x * np.sign(points[:, 0])
        grad_y = viol_y * np.sign(points[:, 1])
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # Update positions
        points = points - lr * total_grad
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check current radius
        current_r = _compute_radius(points)
        
        # Adaptive radius update
        if current_r >= r:
            # Can increase r slightly
            r = current_r + r_step * 0.5
            # But don't overshoot
            r = min(r, current_r + 1e-6)
        else:
            # We lost some radius; reduce r a bit to recover
            r = max(current_r, r - r_step)
            
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 148  # Safety margin (evaluator kills at 150s)

    best_radius = 0.0
    best_c = None

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Primary optimization with Log-Det FISTA
    if time.time() < deadline - 55:
        remaining = deadline - time.time() - 55
        c_opt, r_opt = _logdet_fista(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 45:
        remaining = max(10.0, deadline - time.time() - 45)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 4. MD Radius Expansion
    if best_c is not None and time.time() < deadline - 25:
        remaining = max(10.0, deadline - time.time() - 25)
        c_md, r_md = _md_radius_expansion(best_c, best_radius, max_time=min(25.0, remaining))
        if r_md > best_radius:
            best_radius = r_md
            best_c = c_md.copy()

    # 5. SLSQP Polish with active-set refinement
    if best_c is not None and time.time() < deadline - 14:
        remaining = max(5.0, deadline - time.time() - 14)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(14.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final.copy()

    # 6. QP Polish (Inspired by solution 2)
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(2.0, deadline - time.time() - 5)
        c_qp = _qp_optimize_v2(best_c, max_iter=5000)
        r_qp = _compute_radius(c_qp)
        if r_qp > best_radius:
            best_radius = r_qp
            best_c = c_qp.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
# EVOLVE-BLOCK-END
