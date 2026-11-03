"""Packing 12 unit hexagons in a regular hexagon - minimize outer side."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Hex lattice basis vectors (for unit hexagons, center-to-center distance = 2)
    # Using axial coordinates: a1 = (2, 0), a2 = (1, sqrt(3))
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, math.sqrt(3)])

    # Generate a compact 12-hexagon cluster in hex lattice
    # Use a roughly hexagonal shape: center + 6 neighbors + 5 more
    # Axial coordinates for a compact 12-hexagon cluster
    # Let's use: a 4-4-4 strip or better, a hexagonal-ish cluster
    # Best compact: rows in axial coords
    # A good 12-hexagon shape: (0,0), (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),
    # plus (2,0),(-2,0),(0,2),(0,-2) -- that's 12 but not super compact
    # Better: a 3x4 parallelogram or hexagonal cluster
    # Let me use axial coords for a compact hexagonal shape of 12:
    # Center (0,0)
    # Ring 1: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1) = 6
    # Ring 2 partial: (2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2) -- pick 5 of these
    # Actually let me just use a rectangular grid in axial coords
    # Rows: y=0: x=-2,-1,0,1,2 (5), y=1: x=-1,0,1 (3), y=-1: x=-1,0,1 (3), y=2: x=0 (1) = 12
    # Or simpler: 4 rows of 3: y=-1,0,1,2 with x=-1,0,1 each = 12
    # Let me use axial coords: (q, r) where position = q*a1 + r*a2
    axial_coords = []
    # 4 rows of 3 hexagons
    for r in range(-1, 3):
        for q in range(-1, 2):
            axial_coords.append((q, r))
    # That's 4*3 = 12 hexagons

    # Convert to Cartesian
    positions = []
    for q, r in axial_coords:
        pos = q * a1 + r * a2
        positions.append(pos)
    positions = np.array(positions)

    # Center the cluster
    centroid = np.mean(positions, axis=0)
    positions = positions - centroid

    # Initial angles: all 0
    angles = np.zeros(N)

    # State: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    def state_to_positions(state):
        pos = state[0:2*N].reshape(N, 2)
        ang = state[2*N:3*N]
        return pos, ang

    def compute_objective(state):
        pos, ang = state_to_positions(state)

        # Compute the 6 normal directions of the outer hexagon
        # Outer hexagon is axis-aligned (angle 0), normals at 0, 60, 120, 180, 240, 300 degrees
        normals = []
        for i in range(6):
            theta = i * math.pi / 3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        normals = np.array(normals)  # 6 x 2

        # For each inner hexagon, compute max projection onto any normal
        # A unit hexagon at position p with angle a has vertices at p + R(a) @ base_vertices
        # The max projection onto normal n is p·n + max over vertices of (R(a)@v)·n
        # For a regular hexagon, the support function in direction n is:
        # h(n) = max_i (v_i · n) where v_i are vertices
        # For unit hexagon (side 1), vertices are at distance 1 from center
        # h(n) for a unit hexagon = 1 (the circumradius is 1)
        # Wait, no. For a regular hexagon with side s=1, the circumradius is 1, inradius is sqrt(3)/2
        # The support function h(n) = max of dot products of vertices with n
        # For a regular hexagon, h(n) depends on the angle of n relative to the hexagon

        # Actually, for a regular hexagon with side 1, the vertices are at angles 0, 60, 120, 180, 240, 300 (if unrotated)
        # The support function in direction n at angle phi is:
        # h(phi) = cos(min_k |phi - k*pi/3|) for k=0..5, but that's not quite right
        # Let me just compute it directly

        # Vertices of unit hexagon (side 1) at angle 0:
        base_verts = []
        for i in range(6):
            theta = i * math.pi / 3
            base_verts.append(np.array([math.cos(theta), math.sin(theta)]))
        base_verts = np.array(base_verts)  # 6 x 2

        # For each hexagon, compute max projection onto each normal
        max_proj = 0.0
        for i in range(N):
            p = pos[i]
            a = ang[i]
            # Rotate base vertices
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca, -sa], [sa, ca]])
            verts = R @ base_verts  # 6 x 2
            # For each normal, compute max of (p + v)·n = p·n + max(v·n)
            for j in range(6):
                n = normals[j]
                proj = p @ n + np.max(verts @ n)
                if proj > max_proj:
                    max_proj = proj

        # The outer hexagon side S must satisfy: for all normals n, max_i (p_i + v_i)·n <= S * inradius
        # inradius of outer hexagon with side S is S*sqrt(3)/2
        # So S >= max_proj / (sqrt(3)/2) = max_proj * 2/sqrt(3)
        # But we also need the hexagon to contain all points, so:
        # S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2/sqrt(3)

        S = max_proj * 2.0 / math.sqrt(3)
        return S

    # Check overlap: two hexagons overlap if any edge of one intersects any edge of the other
    # Or use SAT: two convex polygons overlap if no separating axis exists
    def hex_vertices(pos, ang):
        base_verts = []
        for i in range(6):
            theta = i * math.pi / 3
            base_verts.append(np.array([math.cos(theta), math.sin(theta)]))
        base_verts = np.array(base_verts)
        ca, sa = math.cos(ang), math.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]])
        return pos + R @ base_verts.T  # 6 x 2

    def sat_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        for verts in [v1, v2]:
            for i in range(6):
                # Edge from verts[i] to verts[(i+1)%6]
                edge = verts[(i + 1) % 6] - verts[i]
                # Normal to edge
                n = np.array([-edge[1], edge[0]])
                # Project both polygons onto this normal
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                # Check for separation
                if max1 < min2 or max2 < min1:
                    return False  # Separated, no overlap
        return True  # No separating axis, they overlap

    def overlap_penalty(state):
        pos, ang = state_to_positions(state)
        penalty = 0.0
        verts_list = []
        for i in range(N):
            verts_list.append(hex_vertices(pos[i], ang[i]))
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(verts_list[i], verts_list[j]):
                    # Compute overlap depth approximately
                    penalty += 1.0
        return penalty

    def full_objective(state):
        if time.time() > deadline:
            return 1e6
        S = compute_objective(state)
        op = overlap_penalty(state)
        # Penalize overlaps heavily
        return S + 10.0 * op

    # Initial state
    initial_state = np.zeros(3 * N)
    for i in range(N):
        initial_state[2*i] = positions[i][0]
        initial_state[2*i+1] = positions[i][1]
        initial_state[2*i+2] = 0.0

    # Run Nelder-Mead
    best_state = initial_state.copy()
    best_val = full_objective(initial_state)

    try:
        result = minimize(
            full_objective,
            initial_state,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if result.fun < best_val:
            best_state = result.x
            best_val = result.fun

        # Try a few more restarts with small perturbations if time allows
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            perturbed = best_state + np.random.normal(0, 0.1, size=3*N)
            # Keep angles in reasonable range
            perturbed[2*N:3*N] = np.random.uniform(-math.pi/6, math.pi/6, N)
            r2 = minimize(
                full_objective,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if r2.fun < best_val:
                best_state = r2.x
                best_val = r2.fun

    except Exception:
        pass

    # Extract best positions and angles
    pos, ang = state_to_positions(best_state)

    # Re-center to minimize the enclosing hexagon
    # The optimal center is the one that minimizes max projection
    # For simplicity, use centroid
    center = np.mean(pos, axis=0)
    pos = pos - center

    # Recompute the optimal outer hexagon side
    S = compute_objective(best_state)

    # Convert to required format
    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(math.degrees(ang[i]))])

    outer_center = [float(center[0]), float(center[1])]
    outer_side = float(S)
    outer_angle = 0.0  # Outer hexagon is axis-aligned

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
