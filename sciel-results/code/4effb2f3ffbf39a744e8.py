# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _solve_contact_topology(contact_pairs, init_positions):
    """
    Solve for positions such that specified pairs of hexagons are exactly touching
    (center distance = 2 for unit hexagons sharing an edge).
    
    contact_pairs: list of (i, j) pairs that must be at distance 2
    init_positions: initial guess (11, 2)
    
    Returns: (positions, success)
    """
    n = len(init_positions)
    d_target = 2.0  # center-to-center distance for edge-sharing unit hexagons
    
    def equations(vars):
        positions = vars.reshape(n, 2)
        eqs = []
        for (i, j) in contact_pairs:
            d = np.linalg.norm(positions[i] - positions[j])
            eqs.append(d - d_target)
        return np.array(eqs)
    
    init_vars = init_positions.flatten()
    
    try:
        sol, info, ier, msg = fsolve(equations, init_vars, full_output=True, maxfev=50000)
        if ier == 1:
            positions = sol.reshape(n, 2)
            # Verify separation
            angles = np.zeros(n)
            if _check_separation_fast(positions, angles):
                return positions, True
    except Exception:
        pass
    return None, False

def _build_initial_guess(ring_r, outer_r, angle_offset):
    """Build initial guess for 11 hexagons: 1 center + 6 ring + 4 outer."""
    positions = np.zeros((11, 2))
    
    # Center at origin
    positions[0] = [0.0, 0.0]
    
    # 6 ring hexagons at angles 30+60k + angle_offset
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
    
    # 4 outer hexagons at angles 30+120k and 90+120k + angle_offset
    outer_angles = [30, 90, 150, 210]
    for k, va in enumerate(outer_angles):
        a = np.radians(va + angle_offset)
        positions[7 + k] = [outer_r * np.cos(a), outer_r * np.sin(a)]
    
    return positions

def _try_topology(contact_pairs, init_positions):
    """Try to solve a specific contact topology."""
    positions, success = _solve_contact_topology(contact_pairs, init_positions)
    if success:
        angles = np.zeros(len(positions))
        side = _min_enclosing_side(positions, angles)
        return positions, angles, side
    return None, None, None

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Define several contact topologies to try
    # Topology 1: Center touches all 6 ring, ring touches adjacent ring, outer touches ring
    # This is the standard honeycomb-like configuration
    
    # Generate multiple initial guesses and topologies
    topologies = [
        # (contact_pairs, ring_r_init, outer_r_init, angle_offset_init)
        # Topology: center-ring contacts + ring-ring contacts
        ([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
          (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
          (1, 7), (2, 8), (3, 9), (4, 10)], 2.0, 3.5, 0.0),
        
        # Topology: center-ring + ring-outer (different outer placement)
        ([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
          (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
          (1, 7), (3, 8), (5, 9), (6, 10)], 2.0, 3.5, 15.0),
        
        # Topology: sparser contacts, let optimizer find better
        ([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
          (1, 7), (2, 8), (3, 9), (4, 10)], 2.0, 3.5, 0.0),
        
        # Topology: ring-ring + outer-ring (no center-ring)
        ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
          (1, 7), (2, 8), (3, 9), (4, 10)], 2.0, 3.5, 0.0),
        
        # Topology: all ring touches center, outer touches specific ring
        ([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
          (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
          (2, 7), (4, 8), (6, 9), (1, 10)], 2.0, 3.5, 0.0),
    ]
    
    # Try each topology
    for contact_pairs, ring_r_init, outer_r_init, angle_offset_init in topologies:
        if time.time() > deadline - 120:
            break
        
        init_positions = _build_initial_guess(ring_r_init, outer_r_init, angle_offset_init)
        positions, angles, side = _try_topology(contact_pairs, init_positions)
        
        if positions is not None:
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()
    
    # Also try a grid of initial guesses with a simple topology
    if time.time() < deadline - 120:
        simple_topology = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                           (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
        
        for ring_r in [1.9, 2.0, 2.1, 2.2]:
            if time.time() > deadline - 120:
                break
            for outer_r in [3.2, 3.4, 3.6, 3.8]:
                if time.time() > deadline - 120:
                    break
                for angle_offset in [0.0, 10.0, 20.0, 30.0]:
                    if time.time() > deadline - 120:
                        break
                    
                    init_positions = _build_initial_guess(ring_r, outer_r, angle_offset)
                    positions, angles, side = _try_topology(simple_topology, init_positions)
                    
                    if positions is not None:
                        if side < best_side:
                            best_side = side
                            best_positions = positions.copy()
                            best_angles = angles.copy()
    
    # Local refinement using coordinate descent on the best solution
    if best_positions is not None and time.time() < deadline - 30:
        angles = best_angles
        for cd_iter in range(500):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(11):
                for coord in range(2):
                    for step in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]:
                        trial = best_positions.copy()
                        trial[i, coord] += step
                        if not _check_separation_fast(trial, angles):
                            continue
                        trial_side = _min_enclosing_side(trial, angles)
                        if trial_side < best_side - 1e-12:
                            best_positions = trial
                            best_side = trial_side
                            improved = True
                            break
            if not improved:
                break
    
    # Fallback if no solution found
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid config
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
