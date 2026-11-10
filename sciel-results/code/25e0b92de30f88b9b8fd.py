# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _sat_gap(vi, vj):
    """Return minimum SAT gap between two hexagons. Positive = separated, negative = overlapping."""
    all_normals = []
    for v in [vi, vj]:
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-15:
                n = n / norm
                all_normals.append(n)
    
    min_gap = np.inf
    for n in all_normals:
        proj_i = vi @ n
        proj_j = vj @ n
        gap = max(proj_j.min() - proj_i.max(), proj_i.min() - proj_j.max())
        if gap > 0:
            min_gap = min(min_gap, gap)
        else:
            return gap  # Overlapping, return immediately
    return min_gap


def _check_validity(inner, outer, s):
    """
    Check all validity conditions. Returns (is_valid, min_violation).
    inner: (12,3) array (x,y,angle_deg)
    outer: (3,) array (x,y,angle_deg)
    s: float side length
    """
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False, -1.0

    if s < np.sqrt(12.0) - 1e-9:
        return False, -1.0

    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-6:
                return False, -1.0

    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _sat_gap(vi, vj)
            if gap < 1e-6:
                return False, -1.0

    for k in range(12):
        vi = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for v in vi:
            for e in range(6):
                p1 = ov[e]
                p2 = ov[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, -1.0

    return True, 0.0


def _compute_s_for_inner(inner, outer_angle_deg):
    """
    Given inner hexagon positions and a fixed outer hexagon angle,
    find the optimal center and minimal side length s.
    """
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    a = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                         np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
    
    dots = all_verts @ normals.T  # (72, 6)
    M = dots.max(axis=0)  # (6,)
    
    # Solve 2D minimax: minimize max_i(M_i - c·n_i)
    # This is a small LP. Use grid + refinement.
    c0 = all_verts.mean(axis=0)
    
    # Coarse grid
    best_s = np.inf
    best_c = c0
    for dx in np.linspace(-5, 5, 21):
        for dy in np.linspace(-5, 5, 21):
            c = c0 + np.array([dx, dy])
            vals = M - c @ normals.T
            s_val = vals.max() / (np.sqrt(3) / 2)
            if s_val < best_s:
                best_s = s_val
                best_c = c
    
    # Refine
    for _ in range(30):
        for dx in np.linspace(-0.1, 0.1, 11):
            for dy in np.linspace(-0.1, 0.1, 11):
                c = best_c + np.array([dx, dy])
                vals = M - c @ normals.T
                s_val = vals.max() / (np.sqrt(3) / 2)
                if s_val < best_s:
                    best_s = s_val
                    best_c = c
    
    return best_s, np.array([best_c[0], best_c[1], outer_angle_deg])


def _compute_outer_s(inner, cx, cy, ang):
    """Compute outer hexagon side length given inner positions and outer center/angle."""
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    a = np.deg2rad(ang)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                         np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
    
    dots = all_verts @ normals.T
    M = dots.max(axis=0)
    vals = M - np.array([cx, cy]) @ normals.T
    return vals.max() / (np.sqrt(3) / 2)


def _overlap_penalty(inner, threshold=0.01):
    """Compute penalty for overlapping inner hexagons. Returns 0 if all separated by threshold."""
    penalty = 0.0
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _sat_gap(vi, vj)
            if gap < threshold:
                penalty += (threshold - gap) ** 2
    return penalty


def _seed_config_hex_cluster():
    """Hexagonal cluster: center + 6 ring1 + 5 ring2"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2, 0, 0],
        [-2, 0, 0],
        [1, sqrt3, 0],
        [-1, sqrt3, 0],
        [1, -sqrt3, 0],
        [-1, -sqrt3, 0],
        [3, sqrt3, 0],
        [-3, sqrt3, 0],
        [3, -sqrt3, 0],
        [-3, -sqrt3, 0],
        [0, 2 * sqrt3, 0],
    ])


def _seed_config_spread():
    """More spread out arrangement"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2.2, 0, 0],
        [-2.2, 0, 0],
        [1.1, 1.9, 0],
        [-1.1, 1.9, 0],
        [1.1, -1.9, 0],
        [-1.1, -1.9, 0],
        [3.3, 1.9, 0],
        [-3.3, 1.9, 0],
        [3.3, -1.9, 0],
        [-3.3, -1.9, 0],
        [0, 3.8, 0],
    ])


def _seed_config_rotated():
    """Cluster with slight rotation of outer hexagons"""
    sqrt3 = np.sqrt(3)
    base = _seed_config_hex_cluster()
    base[6:, 2] = 10.0
    return base


def _sa_optimize(inner, deadline, rng):
    """
    Simulated annealing to optimize inner hexagon positions (and their angles)
    while minimizing the required outer hexagon side length.
    The outer hexagon center is kept at origin; only angle is optimized.
    """
    sqrt3 = np.sqrt(3)
    best_inner = inner.copy()
    best_s = np.inf
    
    # Initial: find best outer angle for current inner
    for ang in np.arange(0, 60, 5.0):
        s_val = _compute_outer_s(inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
    
    best_ang = 0.0
    for ang in np.arange(0, 60, 0.5):
        s_val = _compute_outer_s(inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
            best_ang = ang
    
    # SA state
    current_inner = inner.copy()
    current_ang = best_ang
    current_s = best_s
    current_penalty = _overlap_penalty(current_inner)
    current_obj = current_s + current_penalty * 10.0
    
    T = 1.0  # Initial temperature
    T_min = 0.001
    step_size = 0.3
    n_iter = 0
    max_iter = 200000
    accept_count = 0
    
    while n_iter < max_iter and time.time() < deadline - 5:
        n_iter += 1
        
        # Generate perturbation
        candidate = current_inner.copy()
        # Move 1-3 random hexagons
        n_move = rng.integers(1, 4)
        indices = rng.choice(12, size=n_move, replace=False)
        for idx in indices:
            candidate[idx, 0] += rng.normal(0, step_size)
            candidate[idx, 1] += rng.normal(0, step_size)
            candidate[idx, 2] += rng.normal(0, step_size * 10)  # Angle in degrees
            # Keep angle in [0, 60)
            candidate[idx, 2] = candidate[idx, 2] % 60.0
        
        # Small chance to perturb outer angle
        if rng.random() < 0.1:
            cand_ang = (current_ang + rng.normal(0, step_size * 5)) % 60.0
        else:
            cand_ang = current_ang
        
        # Evaluate
        cand_s = _compute_outer_s(candidate, 0, 0, cand_ang)
        cand_penalty = _overlap_penalty(candidate)
        cand_obj = cand_s + cand_penalty * 10.0
        
        delta = cand_obj - current_obj
        
        # Accept?
        if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_inner = candidate
            current_ang = cand_ang
            current_s = cand_s
            current_penalty = cand_penalty
            current_obj = cand_obj
            accept_count += 1
            
            if current_penalty < 0.001 and current_s < best_s:
                best_s = current_s
                best_inner = current_inner.copy()
        
        # Adaptive temperature and step size
        if n_iter % 500 == 0:
            accept_rate = accept_count / 500.0
            if accept_rate > 0.5:
                T *= 0.9
                step_size *= 0.95
            elif accept_rate < 0.1:
                T = min(T * 1.2, 1.0)
                step_size = min(step_size * 1.1, 1.0)
            accept_count = 0
        
        # Periodically re-optimize outer angle
        if n_iter % 2000 == 0:
            for ang in np.arange(max(0, current_ang - 5), min(60, current_ang + 5.1), 0.5):
                s_val = _compute_outer_s(current_inner, 0, 0, ang)
                if s_val < current_s:
                    current_s = s_val
                    current_ang = ang
    
    # Final: re-optimize outer angle finely
    for ang in np.arange(max(0, best_ang - 2), min(60, best_ang + 2.1), 0.1):
        s_val = _compute_outer_s(best_inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
            best_ang = ang
    
    return best_inner, np.array([0.0, 0.0, best_ang]), best_s


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer hexagon side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple seed configurations
    seeds = [
        _seed_config_hex_cluster(),
        _seed_config_spread(),
        _seed_config_rotated(),
    ]
    
    best_inner = None
    best_outer = None
    best_s = np.inf
    
    # Phase 1: SA from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 120:
            break
        
        # Allocate remaining time among seeds
        remaining_seeds = len(seeds) - i
        per_seed_time = (deadline - 120 - time.time()) / max(remaining_seeds, 1)
        sa_deadline = time.time() + min(per_seed_time, deadline - 120)
        
        inner, outer, s = _sa_optimize(seed, sa_deadline, rng)
        
        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Phase 2: Fine-tune best result with more SA iterations
    if best_inner is not None and time.time() < deadline - 60:
        inner, outer, s = _sa_optimize(best_inner, deadline - 60, rng)
        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Phase 3: Final outer angle optimization
    if best_inner is not None and time.time() < deadline - 10:
        for ang in np.arange(0, 60, 0.25):
            s_val = _compute_outer_s(best_inner, 0, 0, ang)
            if s_val < best_s:
                best_s = s_val
                best_outer = np.array([0.0, 0.0, ang])
        
        # Fine refine
        if best_outer is not None:
            for ang in np.arange(max(0, best_outer[2] - 0.25), min(60, best_outer[2] + 0.26), 0.025):
                s_val = _compute_outer_s(best_inner, 0, 0, ang)
                if s_val < best_s:
                    best_s = s_val
                    best_outer = np.array([0.0, 0.0, ang])
    
    # Final verification
    if best_inner is None:
        inner = _seed_config_hex_cluster()
        s, outer = _compute_s_for_inner(inner, 0.0)
        best_inner = inner
        best_outer = outer
        best_s = s
    
    valid, _ = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        # Fallback to a known valid configuration
        inner = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [1.25, -2.17, 0],
            [-1.25, -2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, -2.17, 0],
            [-3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
