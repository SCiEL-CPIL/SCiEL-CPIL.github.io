"""Pack 300 equal circles in the unit disk.
Strategy: Symmetry-enforced ring model (concentric shells) with differential evolution
for global optimization, followed by local refinement using cKDTree-based force relaxation."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution

N = 300
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _score(points):
    """Compute the packing score (minimum of pair distances/2 and boundary margins)."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min() / 2.0
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = (1.0 - radii).min()
    return min(min_pair_dist, min_boundary)


def _ring_to_points(radii, angles):
    """Convert ring radii and angular offsets to point coordinates.
    
    radii: array of shape (n_rings,)
    angles: array of shape (n_rings, 2) where angles[i] = [offset, spacing]
            Actually, for each ring, we have a radius and a set of angles.
            Let's define the parameterization more carefully.
    
    We define n_rings rings. For ring i (0-indexed), we have:
      - radius r_i
      - number of circles n_i
      - angular offset theta_i
    
    The total number of circles is sum(n_i) = N.
    The points in ring i are at angles: theta_i + 2*pi*k/n_i for k=0..n_i-1.
    
    To make this differentiable and amenable to DE, we need to handle the discrete n_i.
    A common trick is to fix the number of rings and the number of circles per ring,
    and only optimize the radii and offsets. But n_i must be integers.
    
    Alternative: Use a fixed ring structure. For N=300, a typical optimal packing has
    rings with counts roughly proportional to the radius. Let's assume a fixed set of
    ring counts that sum to N. We can search over a few plausible configurations.
    
    Let's try a simpler approach: assume a fixed number of rings, say K, and
    distribute the N circles among them. To avoid discrete search, we can use
    a continuous relaxation where the "count" is real-valued, but that's tricky.
    
    Better: Let's fix the ring counts. For N=300, known packings have rings like
    [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, ...] but the sum must be 300.
    Let's try a few candidate ring count vectors and pick the best.
    """
    # This function is a placeholder; the actual conversion is done in _evaluate_ring_config
    pass


def _evaluate_ring_config(ring_params, ring_counts):
    """Evaluate a ring configuration.
    
    ring_params: array of shape (K, 2) where K is number of rings.
                 ring_params[i] = [radius_i, angle_offset_i]
    ring_counts: array of shape (K,) with integer counts per ring.
    
    Returns: (score, points)
    """
    K = len(ring_counts)
    points_list = []
    for i in range(K):
        r_i = ring_params[i, 0]
        theta_i = ring_params[i, 1]
        n_i = ring_counts[i]
        if n_i == 0:
            continue
        if n_i == 1:
            # Single circle at center if r_i is small, or at angle theta_i
            if r_i < 1e-6:
                points_list.append(np.array([0.0, 0.0]))
            else:
                x = r_i * np.cos(theta_i)
                y = r_i * np.sin(theta_i)
                points_list.append(np.array([x, y]))
        else:
            angles = theta_i + 2.0 * np.pi * np.arange(n_i) / n_i
            x = r_i * np.cos(angles)
            y = r_i * np.sin(angles)
            points_list.append(np.column_stack([x, y]))
    
    if not points_list:
        return 0.0, np.zeros((N, 2))
    
    points = np.vstack(points_list)
    # Ensure we have exactly N points (should be by construction)
    if len(points) != N:
        # Pad or trim (should not happen if ring_counts sum to N)
        if len(points) < N:
            pad = np.random.uniform(-0.5, 0.5, (N - len(points), 2))
            points = np.vstack([points, pad])
        else:
            points = points[:N]
    
    score = _score(points)
    return score, points


def _local_refine(points, time_budget=10.0):
    """Local refinement using force-directed relaxation."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    tree = cKDTree(points)
    
    step_size = 0.005
    min_step = 1e-8
    patience = 30
    no_improve = 0
    start_time = time.time()
    
    k_neighbors = 6
    _, indices = tree.query(points, k=k_neighbors + 1)
    
    while time.time() - start_time < time_budget:
        i = np.random.randint(n)
        
        force = np.zeros(2)
        
        # Boundary force
        r_i = np.hypot(points[i, 0], points[i, 1])
        if r_i > 1e-10:
            boundary_margin = 1.0 - r_i
            if boundary_margin < 0.05:
                force -= points[i] / r_i * (0.05 - boundary_margin) * 2.0
        
        # Neighbor forces
        for j in indices[i, 1:]:
            if j == i:
                continue
            diff = points[i] - points[j]
            d = np.hypot(diff[0], diff[1])
            if d < 2.0 * best_score * 1.5:
                target = 2.0 * best_score
                if d < target:
                    push = (target - d) / max(d, 1e-10)
                    force += diff * push
        
        norm_f = np.hypot(force[0], force[1])
        if norm_f > 1e-12:
            move = force / norm_f * step_size
            new_point = points[i] + move
            
            if np.hypot(new_point[0], new_point[1]) < 1.0:
                old_point = points[i].copy()
                points[i] = new_point
                
                # Quick check
                valid = True
                for j in indices[i, 1:]:
                    if j == i:
                        continue
                    d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
                    if d < 2.0 * best_score * 0.99:
                        valid = False
                        break
                
                if valid:
                    no_improve += 1
                    if no_improve % 20 == 0:
                        new_score = _score(points)
                        if new_score > best_score:
                            best_score = new_score
                            no_improve = 0
                            tree = cKDTree(points)
                            _, indices = tree.query(points, k=k_neighbors + 1)
                        else:
                            points[i] = old_point
                else:
                    points[i] = old_point
                    no_improve += 1
            else:
                no_improve += 1
        
        if no_improve > patience:
            step_size *= 0.8
            no_improve = 0
            if step_size < min_step:
                break
    
    return points, best_score


def construct():
    start_time = time.time()
    
    # Try a few ring count configurations
    # Common configurations for N=300
    # We'll try a few patterns and pick the best
    
    candidate_configs = [
        # Pattern 1: roughly proportional to radius
        [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],  # sum=551, too many
    ]
    
    # Let's generate valid configs that sum to N=300
    # Start with a base pattern and adjust
    
    # A simpler approach: try a few fixed ring counts
    # Ring counts should be roughly 1, 6, 12, 18, ... but sum to 300
    
    # Let's try: 1 center + rings of 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90
    # Sum of 6+12+18+...+90 = 6*(1+2+...+15) = 6*120 = 720, too many
    
    # For N=300, let's try:
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 + 60 = 331, still too many
    
    # Let's just use a greedy approach: start with 1, then add rings of size 6, 12, 18, ...
    # until we reach N
    
    def generate_ring_counts(N):
        counts = [1]
        current = 1
        ring_size = 6
        while current < N:
            add = min(ring_size, N - current)
            if add > 0:
                counts.append(add)
                current += add
            ring_size += 6
        # Adjust the last ring to make it sum to N
        if sum(counts) != N:
            counts[-1] += N - sum(counts)
            if counts[-1] <= 0:
                counts.pop()
        return counts
    
    ring_counts = generate_ring_counts(N)
    K = len(ring_counts)
    
    print(f"Ring counts: {ring_counts}, total rings: {K}")
    
    # Define bounds for differential evolution
    # For each ring: radius in [0, 1], angle in [0, 2*pi]
    bounds = []
    for i in range(K):
        # Radius bound: ring i should be roughly at position i/(K-1) * 0.9
        r_min = 0.0 if i == 0 else (i - 1) / max(K - 1, 1) * 0.8
        r_max = 0.95 if i == K - 1 else (i + 1) / max(K - 1, 1) * 0.9
        r_min = max(0.0, r_min)
        r_max = min(1.0, r_max)
        bounds.append((r_min, r_max))
        bounds.append((0, 2 * np.pi))
    
    # Objective function for DE (minimize negative score)
    def objective(params):
        ring_params = params.reshape(K, 2)
        # Ensure radii are non-negative
        ring_params[:, 0] = np.abs(ring_params[:, 0])
        score, _ = _evaluate_ring_config(ring_params, ring_counts)
        return -score
    
    # Run differential evolution
    if time.time() - start_time < DEADLINE - 30:
        try:
            result = differential_evolution(
                objective, bounds, maxiter=50, tol=1e-6, seed=42,
                polish=False, workers=-1
            )
            if result.success or -result.fun > 0:
                best_params = result.x
                best_score_de, points_de = _evaluate_ring_config(best_params.reshape(K, 2), ring_counts)
            else:
                best_score_de = 0.0
                points_de = None
        except Exception as e:
            print(f"DE failed: {e}")
            best_score_de = 0.0
            points_de = None
    else:
        best_score_de = 0.0
        points_de = None
    
    # Fallback: hexagonal lattice
    if points_de is None or best_score_de < 0.05:
        # Generate hex lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 3
            pts = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing * 0.5:
                        pts.append((x, y))
            points_hex = np.array(pts)
            if len(points_hex) >= N:
                radii = np.hypot(points_hex[:, 0], points_hex[:, 1])
                points_hex = points_hex[np.argsort(radii, kind='stable')[:N]]
                break
            spacing *= 0.98
        
        best_score_hex = _score(points_hex)
        if best_score_de < best_score_hex:
            points_de = points_hex
            best_score_de = best_score_hex
    
    # Local refinement
    if time.time() - start_time < DEADLINE - 5:
        time_budget = min(20, DEADLINE - (time.time() - start_time) - 5)
        points_final, best_score_final = _local_refine(points_de, time_budget)
    else:
        points_final = points_de
        best_score_final = best_score_de
    
    # Ensure validity
    points_final = np.array(points_final, dtype=np.float64)
    radii = np.hypot(points_final[:, 0], points_final[:, 1])
    if (radii >= 1.0).any():
        for i in range(len(points_final)):
            if radii[i] >= 1.0:
                points_final[i] *= 0.999 / radii[i]
    
    return points_final
