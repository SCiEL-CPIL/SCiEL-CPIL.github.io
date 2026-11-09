"""Physics-based force-directed relaxation for packing 300 circles in unit disk."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the feasible radius (the actual score)."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(dists)
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pips) if False else len(pairs) == 0:
        min_pair = 1.0
    else:
        diff = points[pairs[:, 0]] - points[pairs[:, 1]]
        min_pair = np.min(np.linalg.norm(diff, axis=1))
    return min(min_boundary, min_pair / 2.0)


def _hex_init(spacing, seed=0):
    """Generate hexagonal lattice points inside unit disk, pick N closest to center."""
    rng = np.random.default_rng(seed)
    k = int(1.0 / spacing) + 4
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.3:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    # Add small jitter to break symmetry
    pts = pts + rng.uniform(-spacing * 0.05, spacing * 0.05, size=pts.shape)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind='stable')[:N]
    return pts[idx].copy()


def _ring_init(n_rings, seed=0):
    """Concentric ring initialization with jitter."""
    rng = np.random.default_rng(seed)
    pts = [[0.0, 0.0]]
    remaining = N - 1
    counts = [1]
    for i in range(n_rings - 1):
        c = remaining // (n_rings - 1 - i)
        counts.append(c)
        remaining -= c
    for ring_idx in range(1, n_rings):
        count = counts[ring_idx]
        r_ring = ring_idx / n_rings * 0.92
        for i in range(count):
            angle = 2 * np.pi * i / count + rng.uniform(-0.05, 0.05)
            r = r_ring * (1 + rng.uniform(-0.03, 0.03))
            pts.append([r * np.cos(angle), r * np.sin(angle)])
    pts = np.array(pts)
    return pts


def _relax(points, deadline, max_iters=3000, T0=0.015, T1=0.0005, seed=42):
    """Force-directed relaxation with adaptive step and annealing."""
    points = points.copy()
    best_points = points.copy()
    best_score = _score(points)
    current_score = best_score
    rng = np.random.default_rng(seed)
    
    # Build tree once; rebuild periodically
    tree = cKDTree(points)
    tree_age = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Adaptive temperature / step
        frac = it / max_iters
        T = T0 * (T1 / T0) ** frac
        step = T * 0.8
        
        # Compute forces
        tree_age += 1
        if tree_age >= 5 or it == 0:
            tree = cKDTree(points)
            tree_age = 0
        
        # Pairwise repulsion: for each point, sum (r_i - r_j) / dist * unit_vec for close pairs
        # Use query_pairs with a cutoff
        cutoff = step * 3 + 0.05
        pairs = tree.query_pairs(cutoff, output_type='ndarray')
        
        forces = np.zeros_like(points)
        if len(pairs) > 0:
            diff = points[pairs[:, 0]] - points[pairs[:, 1]]
            dists = np.linalg.norm(diff, axis=1)
            # Avoid division by zero
            mask = dists > 1e-12
            if np.any(mask):
                diff_m = diff[mask]
                dists_m = dists[mask]
                # Repulsion force: stronger when closer
                # Force magnitude ~ 1/dist^2 for hard-core feel
                mag = 1.0 / (dists_m + 1e-6)
                unit = diff_m / dists_m[:, None]
                f = mag[:, None] * unit
                # Accumulate
                for k in range(len(pairs)):
                    if not mask[k]:
                        continue
                    forces[pairs[k, 0]] += f[k]
                    forces[pairs[k, 1]] -= f[k]
        
        # Boundary force: pull inward if near boundary
        r_vec = np.linalg.norm(points, axis=1)
        # Force proportional to how far past the "safe" radius
        safe = 1.0 - 0.01
        boundary_violation = np.maximum(0, r_vec - safe)
        # Inward force
        boundary_force = -points / np.maximum(r_vec, 1e-6)[:, None] * (boundary_violation * 5.0)[:, None]
        forces += boundary_force
        
        # Apply forces with noise
        noise = rng.normal(0, T * 0.5, size=points.shape)
        displacement = forces * step + noise
        
        # Limit displacement magnitude
        disp_mag = np.linalg.norm(displacement, axis=1)
        max_disp = step * 2
        scale = np.minimum(1.0, max_disp / np.maximum(disp_mag, 1e-6))
        displacement *= scale[:, None]
        
        points_new = points + displacement
        
        # Reject if any point goes outside unit disk
        if np.any(np.linalg.norm(points_new, axis=1) > 1.0):
            # Clamp
            r_new = np.linalg.norm(points_new, axis=1)
            mask_out = r_new > 1.0
            points_new[mask_out] *= (1.0 / r_new[mask_out])[:, None]
        
        new_score = _score(points_new)
        
        # Accept if better, or with probability based on temperature
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / (T + 1e-10)):
            points = points_new
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            # Revert
            pass
    
    return best_points, best_score


def _local_refine(points, deadline, max_iters=2000, seed=99):
    """Targeted local refinement: move the most constrained points."""
    points = points.copy()
    best_points = points.copy()
    best_score = _score(points)
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find the point with smallest clearance (most constrained)
        dists = np.linalg.norm(points, axis=1)
        clearance = 1.0 - dists
        
        # Also estimate local crowding
        tree = cKDTree(points)
        # For each point, find nearest neighbor distance
        nn_dists, _ = tree.query(points, k=2)
        local_density = nn_dists[:, 1]  # nearest neighbor distance
        
        # Priority: low clearance OR low local distance
        priority = clearance / 0.1 + local_density / 0.1
        # Move the point with lowest priority (most constrained)
        idx = np.argmin(priority)
        
        original = points[idx].copy()
        best_local = original
        best_local_score = best_score
        
        # Try several perturbations
        for _ in range(8):
            if time.time() > deadline:
                break
            # Direction: away from nearest neighbors and toward center if near boundary
            nn, _ = tree.query(original, k=2)
            # Simple random perturbation with bias toward center if near boundary
            if dists[idx] > 0.9:
                # Bias toward center
                direction = -original / np.linalg.norm(original)
                magnitude = rng.uniform(0.005, 0.03)
                dx = direction[0] * magnitude + rng.uniform(-0.005, 0.005)
                dy = direction[1] * magnitude + rng.uniform(-0.005, 0.005)
            else:
                dx = rng.uniform(-0.015, 0.015)
                dy = rng.uniform(-0.015, 0.015)
            
            points[idx] = original + np.array([dx, dy])
            s = _score(points)
            if s > best_local_score:
                best_local_score = s
                best_local = points[idx].copy()
        
        points[idx] = best_local
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = points.copy()
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    candidates = []
    
    # Generate diverse initializations
    # 1. Hexagonal lattices at various spacings
    for trial in range(15):
        spacing = 0.055 + trial * 0.003
        pts = _hex_init(spacing, seed=trial)
        if pts is not None:
            s = _score(pts)
            if s > 0:
                candidates.append((pts, s))
    
    # 2. Ring initializations
    for n_rings in [8, 10, 12, 15, 18]:
        pts = _ring_init(n_rings, seed=n_rings)
        if pts is not None:
            s = _score(pts)
            if s > 0:
                candidates.append((pts, s))
    
    if not candidates:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_score = candidates[0]
    
    # Relax the top candidates with force-directed method
    top_k = min(4, len(candidates))
    for i in range(top_k):
        if time.time() > deadline - 30.0:
            break
        cand_pts, cand_s = candidates[i]
        refined_pts, refined_s = _relax(cand_pts, deadline - 20.0, 
                                         max_iters=2000, T0=0.012, T1=0.0003, seed=100+i)
        if refined_pts is not None and refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts
    
    # Local refinement on the best result
    if time.time() < deadline - 15.0:
        refined_pts, refined_s = _local_refine(best_points, deadline - 10.0,
                                                max_iters=1500, seed=200)
        if refined_pts is not None and refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts
    
    # Final polish: very small steps
    if time.time() < deadline - 5.0:
        refined_pts, refined_s = _relax(best_points, deadline,
                                         max_iters=1000, T0=0.003, T1=0.0001, seed=300)
        if refined_pts is not None and refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts
    
    return best_points

# EVOLVE-BLOCK-END
