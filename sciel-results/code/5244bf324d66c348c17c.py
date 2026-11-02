"""Discrete lattice topology search + continuous refinement for 12 hexagons in a hexagon.
Enumerates distinct hexagonal lattice packings, then optimizes global rotation/translation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
INV_SQRT3 = 2.0 / SQRT3  # apothem to side factor


def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Minimum outer hexagon side to contain all inner hexagons (support function)."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * INV_SQRT3
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def sat_disjoint(poly1, poly2, tol=1e-8):
        """Check if two polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    def validate_disjoint(inner_hexes):
        """Check all pairs are disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                    return False
        return True

    # --- Generate discrete lattice topologies ---
    def generate_lattice_topologies():
        """Generate distinct hexagonal lattice packings for 12 hexagons."""
        topologies = []
        
        # Topology 1: 3x4 honeycomb (staggered)
        # Rows of 4, 4, 4 with stagger
        for stagger in [0.0, 0.5]:
            positions = []
            for r in range(3):
                for c in range(4):
                    x = c * SQRT3 + (r % 2) * SQRT3 * stagger
                    y = r * 1.5
                    positions.append((x, y, 0.0))
            topologies.append(positions)
        
        # Topology 2: 4x3 honeycomb (transposed)
        positions = []
        for r in range(4):
            for c in range(3):
                x = c * SQRT3 + (r % 2) * SQRT3 * 0.5
                y = r * 1.5
                positions.append((x, y, 0.0))
        topologies.append(positions)
        
        # Topology 3: 2x6 honeycomb
        positions = []
        for r in range(2):
            for c in range(6):
                x = c * SQRT3 + (r % 2) * SQRT3 * 0.5
                y = r * 1.5
                positions.append((x, y, 0.0))
        topologies.append(positions)
        
        # Topology 4: Hexagonal cluster (1 center + 6 ring + 5 outer)
        positions = [(0, 0, 0)]
        for i in range(6):
            angle = i * np.pi / 3
            positions.append((SQRT3 * np.cos(angle), 1.5 * np.sin(angle), 0.0))
        # Add 5 more in a ring
        for i in range(5):
            angle = i * 2 * np.pi / 5 + np.pi / 10
            r = 2 * SQRT3
            positions.append((r * np.cos(angle), r * np.sin(angle), 0.0))
        topologies.append(positions)
        
        # Topology 5: 3x4 with 30-degree rotation of individual hexagons
        positions = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3 + (r % 2) * SQRT3 * 0.5
                y = r * 1.5
                positions.append((x, y, np.pi / 6))
        topologies.append(positions)
        
        # Topology 6: 4x3 with 30-degree rotation
        positions = []
        for r in range(4):
            for c in range(3):
                x = c * SQRT3 + (r % 2) * SQRT3 * 0.5
                y = r * 1.5
                positions.append((x, y, np.pi / 6))
        topologies.append(positions)
        
        # Topology 7: Compact 3x4 with slight compression
        positions = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3 * 0.98 + (r % 2) * SQRT3 * 0.49
                y = r * 1.48
                positions.append((x, y, 0.0))
        topologies.append(positions)
        
        # Topology 8: 2x6 with 30-degree rotation
        positions = []
        for r in range(2):
            for c in range(6):
                x = c * SQRT3 + (r % 2) * SQRT3 * 0.5
                y = r * 1.5
                positions.append((x, y, np.pi / 6))
        topologies.append(positions)
        
        return topologies

    def make_hexes_from_positions(positions, outer_cx, outer_cy, outer_a):
        """Create hexagon vertices from positions."""
        inner_hexes = []
        for (x, y, angle) in positions:
            verts = hexagon_vertices(x, y, 1.0, angle)
            inner_hexes.append(verts)
        return inner_hexes

    def objective_for_topology(positions, outer_cx, outer_cy, outer_a):
        """Objective for a fixed topology: minimize outer side."""
        inner_hexes = make_hexes_from_positions(positions, outer_cx, outer_cy, outer_a)
        
        # Check disjointness
        if not validate_disjoint(inner_hexes):
            return 1e6
        
        side = outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_a)
        return side

    def optimize_topology(positions):
        """Optimize global rotation and translation for a fixed topology."""
        # Parameters: outer_cx, outer_cy, outer_a
        # We can fix the cluster center at origin by symmetry, so optimize outer position and angle
        
        def obj(params):
            outer_cx = params[0]
            outer_cy = params[1]
            outer_a = params[2]
            return objective_for_topology(positions, outer_cx, outer_cy, outer_a)
        
        # Initial guess: cluster centered, outer hexagon aligned
        x0 = np.array([0.0, 0.0, 0.0])
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            return result.fun, result.x
        except Exception:
            return 1e6, x0

    def optimize_with_individual_rotations(positions_base):
        """Allow individual hexagon rotations for more flexibility."""
        # Parameters: outer_cx, outer_cy, outer_a, plus 12 individual angle offsets
        def obj(params):
            outer_cx = params[0]
            outer_cy = params[1]
            outer_a = params[2]
            angle_offsets = params[3:15]
            
            positions = []
            for i in range(N):
                x, y, base_angle = positions_base[i]
                angle = base_angle + angle_offsets[i]
                positions.append((x, y, angle))
            
            inner_hexes = make_hexes_from_positions(positions, outer_cx, outer_cy, outer_a)
            
            if not validate_disjoint(inner_hexes):
                return 1e6
            
            side = outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_a)
            return side
        
        x0 = np.zeros(15)
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            return result.fun, result.x
        except Exception:
            return 1e6, x0

    # --- Main search ---
    topologies = generate_lattice_topologies()
    
    best_score = 1e6
    best_params = None
    best_topology_idx = 0
    
    # Phase 1: Optimize each topology with global rotation only
    for idx, positions in enumerate(topologies):
        if time.time() > deadline - 20:
            break
        
        score, params = optimize_topology(positions)
        if score < best_score:
            best_score = score
            best_params = params
            best_topology_idx = idx
    
    # Phase 2: For the best topology, allow individual rotations
    if time.time() < deadline - 15:
        score2, params2 = optimize_with_individual_rotations(topologies[best_topology_idx])
        if score2 < best_score:
            best_score = score2
            best_params = params2
    
    # Phase 3: Try all topologies with individual rotations (if time permits)
    if time.time() < deadline - 10:
        for idx in range(len(topologies)):
            if time.time() > deadline - 8:
                break
            if idx == best_topology_idx:
                continue
            score3, params3 = optimize_with_individual_rotations(topologies[idx])
            if score3 < best_score:
                best_score = score3
                best_params = params3
                best_topology_idx = idx
    
    # --- Extract best result ---
    if best_params is None:
        # Fallback: widely spaced valid configuration
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Determine if we used individual rotations
    if len(best_params) == 15:
        outer_cx = best_params[0]
        outer_cy = best_params[1]
        outer_a = best_params[2]
        angle_offsets = best_params[3:15]
        positions = []
        for i in range(N):
            x, y, base_angle = topologies[best_topology_idx][i]
            angle = base_angle + angle_offsets[i]
            positions.append((x, y, angle))
    else:
        outer_cx = best_params[0]
        outer_cy = best_params[1]
        outer_a = best_params[2]
        positions = topologies[best_topology_idx]
    
    # Create final hexagons
    inner_hexes = []
    for (x, y, angle) in positions:
        verts = hexagon_vertices(x, y, 1.0, angle)
        inner_hexes.append(verts)
    
    # Final validation
    if not validate_disjoint(inner_hexes):
        # Fallback
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    exact_side = outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_a)
    
    inner = [[positions[i][0], positions[i][1], np.degrees(positions[i][2])] for i in range(N)]
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
