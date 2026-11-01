"""Physics-based relaxation for 12 unit hexagons inside a regular hexagon."""
import numpy as np
import time
import random

N = 12
DEADLINE = 145.0

_SQRT3 = np.sqrt(3.0)
_HALF_SQRT3 = _SQRT3 / 2.0
_60DEG = np.radians(60.0)


def _hex_verts(cx, cy, angle_rad):
    """Vertices of a unit regular hexagon."""
    thetas = angle_rad + _60DEG * np.arange(6)
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])


def _hex_normals(angle_rad):
    """6 outward edge normals."""
    thetas = angle_rad + np.radians(30) + _60DEG * np.arange(6)
    return np.column_stack([np.cos(thetas), np.sin(thetas)])


def _axial_to_cartesian(q, r):
    x = _SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    return [x, y]


def _make_lattice_config():
    """Standard hexagonal lattice: center + 6 ring + 5 more."""
    coords = [(0, 0)]
    coords.extend([(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1)])
    coords.extend([(2, 0), (0, 2), (-2, 1), (-2, -1), (1, -2)])
    centers = np.array([_axial_to_cartesian(q, r) for q, r in coords])
    angles = np.zeros(N)
    return centers, angles


def _make_random_config():
    """Random initial configuration."""
    centers = np.random.uniform(-3, 3, size=(N, 2))
    angles = np.random.uniform(0, 2 * np.pi, size=N)
    return centers, angles


def _pairwise_min_distance(centers, angles):
    """Compute minimum distance between all pairs of hexagons using SAT.
    Returns array of min separations (negative = overlap)."""
    n = len(centers)
    min_seps = np.full((n, n), np.inf)
    for i in range(n):
        v1 = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
        n1 = _hex_normals(angles[i])
        for j in range(i + 1, n):
            v2 = _hex_verts(centers[j, 0], centers[j, 1], angles[j])
            n2 = _hex_normals(angles[j])
            axes = np.vstack([n1, n2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1),
                              p2.min(axis=1) - p1.max(axis=1))
            min_seps[i, j] = min_seps[j, i] = seps.min()
    return min_seps


def _outer_side(centers, angles, outer_center, outer_angle_rad):
    """Compute required outer hexagon side length."""
    outer_n = _hex_normals(outer_angle_rad)
    max_d = 0.0
    for i in range(len(centers)):
        v = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
        rel = v - np.array(outer_center)
        d = np.abs(rel @ outer_n.T)
        max_d = max(max_d, d.max())
    return max_d / _HALF_SQRT3


def _objective(centers, angles, outer_center, outer_angle_rad, penalty_w=100.0):
    """Objective: outer side + penalty for overlaps."""
    side = _outer_side(centers, angles, outer_center, outer_angle_rad)
    # Overlap penalty
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        v1 = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
        n1 = _hex_normals(angles[i])
        for j in range(i + 1, n):
            v2 = _hex_verts(centers[j, 0], centers[j, 1], angles[j])
            n2 = _hex_normals(angles[j])
            axes = np.vstack([n1, n2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1),
                              p2.min(axis=1) - p1.max(axis=1))
            min_sep = seps.min()
            if min_sep < 0:
                penalty += min_sep * min_sep
    return side + penalty_w * penalty


def _relax_lj(centers, angles, outer_center, outer_angle_rad, 
              n_steps=2000, lr=0.01, temp=1.0, t0=None):
    """Lennard-Jones style relaxation with containment constraint."""
    centers = centers.copy()
    angles = angles.copy()
    n = len(centers)
    
    best_obj = float('inf')
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    for step in range(n_steps):
        if t0 is not None and time.time() - t0 > DEADLINE:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        torques = np.zeros(n)
        
        for i in range(n):
            v1 = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
            n1 = _hex_normals(angles[i])
            for j in range(i + 1, n):
                v2 = _hex_verts(centers[j, 0], centers[j, 1], angles[j])
                n2 = _hex_normals(angles[j])
                axes = np.vstack([n1, n2])
                p1 = axes @ v1.T
                p2 = axes @ v2.T
                seps = np.maximum(p1.min(axis=1) - p2.max(axis=1),
                                  p2.min(axis=1) - p1.max(axis=1))
                min_sep = seps.min()
                
                if min_sep < 0.5:
                    # Repulsive force
                    r_vec = centers[j] - centers[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-10:
                        r_hat = r_vec / r_mag
                    else:
                        r_hat = np.array([np.random.randn(), np.random.randn()])
                        r_hat /= np.linalg.norm(r_hat)
                    
                    # LJ-like repulsion
                    f = 10.0 / (min_sep + 0.1) ** 3
                    forces[i] -= f * r_hat
                    forces[j] += f * r_hat
        
        # Containment force: push vertices back inside outer hexagon
        outer_n = _hex_normals(outer_angle_rad)
        for i in range(n):
            v = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
            rel = v - np.array(outer_center)
            d = np.abs(rel @ outer_n.T)
            max_d = d.max()
            current_side = max_d / _HALF_SQRT3
            # Gentle pull toward center
            center_vec = np.array(outer_center) - centers[i]
            forces[i] += 0.1 * center_vec
        
        # Update positions
        for i in range(n):
            centers[i] += lr * forces[i]
        
        # Update angles with small random perturbation
        angles += np.random.normal(0, 0.01, size=n)
        
        # Evaluate
        obj = _objective(centers, angles, outer_center, outer_angle_rad)
        if obj < best_obj:
            best_obj = obj
            best_centers = centers.copy()
            best_angles = angles.copy()
    
    return best_centers, best_angles, best_obj


def _optimize_rotation(centers, angles, outer_center, outer_angle_rad, t0):
    """Optimize global rotation of the entire configuration."""
    best_angle = 0.0
    best_side = float('inf')
    
    for trial in range(36):
        if time.time() - t0 > DEADLINE:
            break
        angle = trial * (2 * np.pi / 36)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        rotated = centers @ rot.T
        side = _outer_side(rotated, angles, outer_center, outer_angle_rad)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    # Refine
    if best_side < float('inf'):
        for _ in range(100):
            if time.time() - t0 > DEADLINE:
                break
            angle = best_angle + np.random.uniform(-0.05, 0.05)
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
            rotated = centers @ rot.T
            side = _outer_side(rotated, angles, outer_center, outer_angle_rad)
            if side < best_side:
                best_side = side
                best_angle = angle
    
    return best_angle, best_side


def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    outer_center = (0.0, 0.0)
    outer_angle_rad = 0.0
    
    best_overall = None
    best_overall_side = float('inf')
    
    # Strategy 1: Lattice-based relaxation
    centers0, angles0 = _make_lattice_config()
    
    # Try multiple relaxation runs with different parameters
    for lr in [0.005, 0.01, 0.02]:
        for n_steps in [1000, 2000, 3000]:
            if time.time() - t0 > DEADLINE - 10:
                break
            c, a, obj = _relax_lj(centers0, angles0, outer_center, outer_angle_rad,
                                  n_steps=n_steps, lr=lr, t0=t0)
            # Optimize rotation
            rot_angle, side = _optimize_rotation(c, a, outer_center, outer_angle_rad, t0)
            if side < best_overall_side:
                rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
                rotated_c = c @ rot.T
                best_overall = (rotated_c, a, side)
                best_overall_side = side
    
    # Strategy 2: Random restarts with relaxation
    for restart in range(5):
        if time.time() - t0 > DEADLINE - 10:
            break
        centers_r, angles_r = _make_random_config()
        c, a, obj = _relax_lj(centers_r, angles_r, outer_center, outer_angle_rad,
                              n_steps=1500, lr=0.01, t0=t0)
        rot_angle, side = _optimize_rotation(c, a, outer_center, outer_angle_rad, t0)
        if side < best_overall_side:
            rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                            [np.sin(rot_angle), np.cos(rot_angle)]])
            rotated_c = c @ rot.T
            best_overall = (rotated_c, a, side)
            best_overall_side = side
    
    # Use best result
    if best_overall is None:
        # Fallback
        centers0, angles0 = _make_lattice_config()
        inner = []
        for i in range(N):
            inner.append([centers0[i, 0], centers0[i, 1], np.degrees(angles0[i])])
        side = _outer_side(centers0, angles0, outer_center, outer_angle_rad)
        return inner, outer_center, side, 0.0
    
    centers_best, angles_best, side_best = best_overall
    
    # Validate: check for overlaps
    valid = True
    for i in range(N):
        v1 = _hex_verts(centers_best[i, 0], centers_best[i, 1], angles_best[i])
        n1 = _hex_normals(angles_best[i])
        for j in range(i + 1, N):
            v2 = _hex_verts(centers_best[j, 0], centers_best[j, 1], angles_best[j])
            n2 = _hex_normals(angles_best[j])
            axes = np.vstack([n1, n2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1),
                              p2.min(axis=1) - p1.max(axis=1))
            if seps.min() < -1e-7:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to lattice
        centers0, angles0 = _make_lattice_config()
        inner = []
        for i in range(N):
            inner.append([centers0[i, 0], centers0[i, 1], np.degrees(angles0[i])])
        side = _outer_side(centers0, angles0, outer_center, outer_angle_rad)
        return inner, outer_center, side, 0.0
    
    inner = []
    for i in range(N):
        inner.append([centers_best[i, 0], centers_best[i, 1], np.degrees(angles_best[i])])
    
    final_side = _outer_side(centers_best, angles_best, outer_center, outer_angle_rad)
    return inner, outer_center, final_side, 0.0
