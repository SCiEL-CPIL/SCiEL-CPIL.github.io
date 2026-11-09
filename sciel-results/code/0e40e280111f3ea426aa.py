# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """Return 6 outward unit edge normals for a hexagon given its 6 vertices in order."""
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separated(v1, n1, v2, n2):
    """
    Check if two hexagons with vertices v1 (6x2) and v2 (6x2) and normals n1, n2 (6x2)
    are separated. Returns True if separated by > 1e-6 on at least one of 12 normals.
    """
    all_normals = np.vstack([n1, n2])  # 12 x 2
    for k in range(12):
        n = all_normals[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if (max1 < min2 - 1e-6) or (max2 < min1 - 1e-6):
            return True
    return False


def _all_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _min_outer_side(inner_hexes, angle_deg=0.0):
    """
    Find the minimum side length S such that all inner hexagons fit inside
    a regular hexagon centered at origin with side S and given angle.
    Returns S or inf if infeasible.
    """
    def _fits(S):
        outer_verts = _hex_vertices(0, 0, S, angle_deg)
        for hex_verts, normals in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    # Compute bounding radius
    max_r = 0
    for hex_verts, normals in inner_hexes:
        r = np.max(np.linalg.norm(hex_verts, axis=1))
        max_r = max(max_r, r)

    # Binary search for minimum S
    S_low = max_r
    S_high = S_low * 2.0

    for _ in range(60):
        S_mid = (S_low + S_high) / 2.0
        if _fits(S_mid):
            S_high = S_mid
        else:
            S_low = S_mid

    return S_high


def _find_min_outer_side_all_angles(inner_hexes):
    """Find the minimum outer side length over all angles."""
    best_S = float('inf')
    best_angle = 0.0
    
    # Try many angles
    for angle_deg in np.linspace(0, 60, 61):
        S = _min_outer_side(inner_hexes, angle_deg)
        if S < best_S:
            best_S = S
            best_angle = angle_deg
    
    # Refine around best angle
    if best_S < float('inf'):
        for angle_deg in np.linspace(best_angle - 5, best_angle + 5, 21):
            S = _min_outer_side(inner_hexes, angle_deg)
            if S < best_S:
                best_S = S
                best_angle = angle_deg
    
    return best_S, best_angle


def _hex_lattice_positions(max_radius=5):
    """Generate hexagonal lattice positions within a radius."""
    positions = []
    # Hexagonal lattice: positions are (a*sqrt(3)/2 + b*sqrt(3)/2, a*1.5 + b*0) for integers a, b
    # Or use axial coordinates: (q, r) -> (q + r/2, r*sqrt(3)/2)
    sqrt3 = np.sqrt(3)
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            x = (q + r / 2.0) * sqrt3
            y = r * 1.5
            if np.sqrt(x**2 + y**2) <= max_radius * 2.0:
                positions.append((x, y))
    return np.array(positions)


def _generate_lattice_subsets(n_select=11, max_radius=4):
    """
    Generate all subsets of size n_select from hexagonal lattice positions.
    Uses the fact that the optimal packing is likely on the hexagonal lattice.
    """
    lattice = _hex_lattice_positions(max_radius)
    n = len(lattice)
    
    # We need to choose 11 positions from the lattice
    # Use itertools.combinations but this can be very large
    # Instead, we'll use a smarter approach: start with known good configurations
    # and perturb them
    
    # Known good configuration: center + 6 neighbors + 4 more
    # In hexagonal lattice coordinates:
    # Center: (0, 0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more: (2,0), (1,1), (-1,2), (-2,1) or similar
    
    # Let's generate candidate configurations
    candidates = []
    
    # Configuration 1: compact hexagonal arrangement
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,2), (-2,1)
    config1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,2), (-2,1)]
    candidates.append(config1)
    
    # Configuration 2: slightly different
    config2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,0), (2,-1)]
    candidates.append(config2)
    
    # Configuration 3: ring + center + 4
    config3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (0,-2), (-2,1)]
    candidates.append(config3)
    
    # Convert to Cartesian coordinates
    sqrt3 = np.sqrt(3)
    all_positions = []
    for config in candidates:
        pos = np.array([(q + r/2.0)*sqrt3, r*1.5] for q, r in config)
        all_positions.append(pos)
    
    return all_positions


def _evaluate_configuration(positions, angles=None):
    """Evaluate a configuration: return (S, angle) or (inf, 0)."""
    if angles is None:
        angles = np.zeros(11)
    
    inner_hexes = []
    for i in range(11):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        norms = _edge_normals(verts)
        inner_hexes.append((verts, norms))
    
    if not _all_separated(inner_hexes):
        return float('inf'), 0.0
    
    S, angle = _find_min_outer_side_all_angles(inner_hexes)
    return S, angle


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    t_start = time.time()
    deadline = 600.0
    
    def _elapsed():
        return time.time() - t_start
    
    def _time_left():
        return deadline - _elapsed()
    
    # Generate candidate configurations from hexagonal lattice
    candidates = _generate_lattice_subsets()
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Evaluate each candidate
    for i, pos in enumerate(candidates):
        if _time_left() < 30:
            break
        
        S, angle = _evaluate_configuration(pos)
        if S < best_S:
            best_S = S
            best_positions = pos.copy()
            best_angles = np.zeros(11)
            best_outer_angle = angle
    
    # If we found a good configuration, try to improve it with local optimization
    if best_S < float('inf') and _time_left() > 60:
        # Try small perturbations of positions
        rng = np.random.default_rng(42)
        for trial in range(20):
            if _time_left() < 30:
                break
            
            perturbed = best_positions + rng.normal(0, 0.1, size=(11, 2))
            S, angle = _evaluate_configuration(perturbed)
            if S < best_S:
                best_S = S
                best_positions = perturbed.copy()
                best_angles = np.zeros(11)
                best_outer_angle = angle
        
        # Try optimizing angles
        if _time_left() > 30:
            def _angle_objective(angle_params):
                angles = angle_params
                inner_hexes = []
                for i in range(11):
                    verts = _hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, angles[i])
                    norms = _edge_normals(verts)
                    inner_hexes.append((verts, norms))
                
                if not _all_separated(inner_hexes):
                    return 1e10
                
                S, _ = _find_min_outer_side_all_angles(inner_hexes)
                return S
            
            # Start from zero angles
            x0 = np.zeros(11)
            try:
                res = minimize(_angle_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_S:
                    best_S = res.fun
                    best_angles = res.x.copy()
                    # Recompute outer angle
                    inner_hexes = []
                    for i in range(11):
                        verts = _hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
                        norms = _edge_normals(verts)
                        inner_hexes.append((verts, norms))
                    _, best_outer_angle = _find_min_outer_side_all_angles(inner_hexes)
            except Exception:
                pass
    
    # Final fallback: use the seed arrangement if nothing better found
    if best_S >= float('inf'):
        seed_positions = np.array(
            [
                [0, 0],
                [-2.5, 0],
                [2.5, 0],
                [-1.25, 2.165],
                [1.25, 2.165],
                [-1.25, -2.165],
                [1.25, -2.165],
                [-3.75, 2.165],
                [3.75, 2.165],
                [-3.75, -2.165],
                [3.75, -2.165],
            ]
        )
        best_positions = seed_positions
        best_angles = np.zeros(11)
        S, angle = _evaluate_configuration(best_positions, best_angles)
        best_S = S
        best_outer_angle = angle
    
    # Build final result
    inner_hexes = []
    for i in range(11):
        verts = _hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
        norms = _edge_normals(verts)
        inner_hexes.append((verts, norms))
    
    final_S, final_angle = _find_min_outer_side_all_angles(inner_hexes)
    
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    outer_hex_side_length = float(final_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
