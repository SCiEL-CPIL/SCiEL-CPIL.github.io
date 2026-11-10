# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    """Compute all 66 vertices for 11 hexagons. positions (11,2), angles (11,)"""
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    """Check all pairs separated using SAT on 12 edge normals per pair."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    
    # Precompute edge normals for each hexagon (6 normals each)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]  # (6,2)
            vj = all_verts[j]  # (6,2)
            # 12 normals: 6 from hex i, 6 from hex j
            normals = np.vstack([hex_normals[i], hex_normals[j]])  # (12,2)
            
            # Project all vertices onto each normal
            proj_i = vi @ normals.T  # (6, 12)
            proj_j = vj @ normals.T  # (6, 12)
            
            max_i = np.max(proj_i, axis=0)  # (12,)
            min_i = np.min(proj_i, axis=0)  # (12,)
            max_j = np.max(proj_j, axis=0)  # (12,)
            min_j = np.min(proj_j, axis=0)  # (12,)
            
            # Check if any normal separates them
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue  # separated
            
            return False  # not separated
    return True

def _min_enclosing_side(positions, angles):
    """Min side length of outer hexagon (centered at origin, angle=0) containing all vertices."""
    all_verts = _all_vertices(positions, angles)  # (11, 6, 2)
    flat = all_verts.reshape(-1, 2)  # (66, 2)
    
    # Outer hexagon at angle 0: vertices at angles 0, 60, 120, 180, 240, 300
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    
    projs = flat @ normals.T  # (66, 6)
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    """Check all inner vertices inside outer hexagon (center=origin, angle=0)."""
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

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    d = np.sqrt(3)
    
    # Symmetric initial arrangement: center + 6 ring + 4 in gaps
    # All angles = 0 for simplicity (exploits outer hexagon symmetry)
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center
    positions[0] = [0.0, 0.0]
    # 6 surrounding at distance sqrt(3)
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k + 1] = [d * np.cos(a), np.sin(a) * d]
    # 4 more in the "valleys" at distance 2*sqrt(3)
    for k in range(0, 6, 2):
        a = np.radians(30 + 60 * k)
        positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    # One more at the remaining valley
    a = np.radians(30 + 60 * 1)
    positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    
    # Verify initial validity
    if not _check_separation_fast(positions, angles):
        # Scale up slightly if needed
        scale = 1.0
        while not _check_separation_fast(positions, angles) and scale < 5.0:
            positions = positions * 1.01
            scale += 1.01
        positions[0] = [0.0, 0.0]  # Keep center
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Phase 1: Symmetry-reduced optimization
    # We exploit 6-fold symmetry: 
    # - Hex 0 is at center (0,0)
    # - Hexes 1-6 are in a ring, symmetric under 60 deg rotation
    # - Hexes 7-10 are in a second ring, symmetric under 60 deg rotation
    # - Hex 11 is free (but we only have 11, so maybe 10 are symmetric and 1 is free? No, 1+6+4=11)
    # Let's define parameters for the symmetric configuration:
    # r1, a1 (radius, angle of ring 1) - but angle is fixed by symmetry?
    # Actually, let's just use SLSQP on the full set but with good initial guess and constraints.
    
    def objective(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        return _min_enclosing_side(pos, ang)
    
    def constraints_sep(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        if not _check_separation_fast(pos, ang, tol=1e-5):
            return 0.0
        return 1.0
        
    # This constraint approach is tricky with SLSQP. 
    # Instead, let's use a penalty method or just Nelder-Mead with many restarts.
    # The current program already does Nelder-Mead. 
    # Let's try a different strategy: 
    # 1. Fix angles to 0 (or 30) for all hexagons to reduce DOF.
    # 2. Optimize positions only.
    # 3. Then optimize angles for the most constrained hexagons.
    
    # Phase 1: Optimize positions with fixed angles (all 0)
    def objective_pos(x):
        pos = x.reshape(11, 2)
        return _min_enclosing_side(pos, best_angles)
    
    # Use SLSQP with bounds
    bounds = [(-10, 10)] * 22
    # Initial guess
    x0 = best_positions.flatten()
    
    if time.time() < deadline - 60.0:
        try:
            res = minimize(
                objective_pos,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_side:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    best_positions = trial_pos
                    best_side = _min_enclosing_side(best_positions, best_angles)
        except Exception:
            pass
    
    # Phase 2: Optimize angles for each hexagon individually
    if time.time() < deadline - 30.0:
        for i in range(11):
            def objective_ang(ang):
                trial_angles = best_angles.copy()
                trial_angles[i] = ang
                return _min_enclosing_side(best_positions, trial_angles)
            
            # SLSQP on angle
            try:
                res = minimize(
                    objective_ang,
                    best_angles[i],
                    method='SLSQP',
                    bounds=[(0, 360)],
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                if res.fun < best_side:
                    trial_angles = best_angles.copy()
                    trial_angles[i] = res.x[0] % 360
                    if _check_separation_fast(best_positions, trial_angles):
                        best_angles = trial_angles
                        best_side = _min_enclosing_side(best_positions, best_angles)
            except Exception:
                pass
    
    # Phase 3: Fine-tune positions again
    if time.time() < deadline - 30.0:
        try:
            res = minimize(
                objective_pos,
                best_positions.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_side:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    best_positions = trial_pos
                    best_side = _min_enclosing_side(best_positions, best_angles)
        except Exception:
            pass
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid configuration
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(0, 6, 2):
            a = np.radians(30 + 60 * k)
            positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        a = np.radians(30 + 60 * 1)
        positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure containment
    if not _check_containment_fast(best_positions, best_angles, final_side):
        # Slightly increase side
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
