"""
Direct 2D Free-Body Optimization for Hexagon Packing.

Strategy:
1. Represent the 11 inner hexagons by their center coordinates (x, y).
2. Assume all inner hexagons have the same orientation as the outer hexagon (angle 0).
   This is optimal for regular hexagons packed in a regular hexagon container due to symmetry.
3. Use scipy.optimize.differential_evolution to minimize the outer side length.
   The objective function calculates the minimal outer side required to contain all
   vertices of the inner hexagons, given their current positions.
4. Enforce non-overlap using a penalty term based on the Separating Axis Theorem (SAT).
   If any two hexagons overlap, the penalty is large, guiding the optimizer away.
5. The outer hexagon is centered at (0,0) with side length S and angle 0.
   The minimal S is determined by the maximum projection of all inner hexagon vertices
   onto the 6 normal directions of the outer hexagon.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # The 6 constraints for a hexagon with side S centered at origin, angle 0:
    # |lx| <= S
    # |lx + ly/sqrt(3)| <= S  (Wait, let's verify the inequalities for a standard hexagon)
    # Standard hexagon vertices at (1,0), (1/2, sqrt(3)/2), ...
    # Edges are vertical? No.
    # Let's use the support function.
    # For a regular hexagon of side S, centered at origin, with vertices at angles 0, 60, 120...
    # The apothem is S * sqrt(3) / 2.
    # The distance from center to any point inside is limited by the support function.
    # The support function h(u) for a hexagon with side S is S * max( u_x, u_y/sqrt(3), ... )?
    # Let's derive the inequalities for the region defined by 6 lines.
    # Line 1: x = S (Normal (1,0)) -> x <= S
    # Line 2: x + y/sqrt(3) = S (Normal (1, 1/sqrt(3)) normalized? No.)
    # Let's use the vertex form to find the bounding box in the rotated frame.
    # Actually, the formula provided in the seed code is:
    # c1 = lx
    # c2 = -lx
    # c3 = lx + ly / SQRT3
    # c4 = -lx + ly / SQRT3
    # c5 = -lx - ly / SQRT3
    # c6 = lx - ly / SQRT3
    # Max of these is the required side length S.
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / SQRT3
    c4 = -lx + ly / SQRT3
    c5 = -lx - ly / SQRT3
    c6 = lx - ly / SQRT3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _check_no_overlap_fast(positions, angles, tol=1e-7):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(positions)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
        
    # SAT Check
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _objective(params):
    """
    Objective function for differential evolution.
    params: [x0, y0, x1, y1, ..., x10, y10] (2*N floats)
    """
    # Check deadline
    if time.time() > DEADLINE:
        return 1000.0

    # Reshape params to (N, 2)
    positions = params.reshape(N, 2)
    angles = [0.0] * N  # Fixed orientation for simplicity and symmetry optimality
    
    # 1. Calculate minimal outer side length
    all_verts = _hex_vertices_batch(positions, angles, side=1.0)
    min_side = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
    
    # 2. Check for overlaps and apply penalty
    # Instead of a boolean check, we can calculate a penalty based on penetration depth,
    # but for DE, a boolean penalty that forces the optimizer to explore valid regions
    # is often sufficient if the initial population is good or the penalty is high enough.
    # However, a smoother penalty helps. Let's use a simple overlap check first.
    # If overlapping, return a large penalty + min_side.
    
    # To make the landscape smoother, we can calculate the minimum distance between pairs
    # and penalize if it's negative (overlap).
    # But SAT is discrete. Let's stick to a high penalty for overlap.
    # We can also add a small regularization to keep centers close to a hex lattice to help convergence?
    # No, let's trust the optimizer.
    
    if not _check_no_overlap_fast(positions, angles):
        # Penalty: Return a value significantly larger than any valid side length
        # Valid side lengths for 11 hexagons are likely around 4-5.
        # So 100.0 is a good penalty.
        return 100.0 + min_side
        
    return min_side

def construct():
    global START_TIME, DEADLINE
    
    # Initial guess: Hexagonal lattice
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 4 hexagons at distance 2*sqrt(3) (aligned with ring 1 gaps? or corners?)
    # Let's start with a valid, tight packing.
    # Standard 19-hex cluster has side length 3*sqrt(3) ~ 5.2 for the outer hex if aligned?
    # Let's generate a standard hex lattice position for 11 hexagons.
    
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    pos_init = []
    # Center
    pos_init.append([0.0, 0.0])
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos_init.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: Pick 4 that form a symmetric subset.
    # In the 19-hex cluster, the second ring has 12 hexagons.
    # 6 are at distance 2*sqrt(3) (aligned with ring 1).
    # 6 are at distance 3 (corners).
    # To be compact, we should probably pick the ones at distance 2*sqrt(3) if they don't overlap?
    # Wait, distance between ring 1 (dist r1) and ring 2 (dist r2) aligned is r2-r1 = sqrt(3).
    # So they touch. This is a valid packing.
    # Let's pick 4 of the 6 aligned ones. To be symmetric, we can't pick 4 out of 6 with 6-fold symmetry.
    # We can pick 0, 90, 180, 270? No, those angles don't exist in the hex lattice.
    # Let's just pick the first 4: 0, 60, 120, 180 degrees.
    # This breaks symmetry, but the optimizer can rotate/shift.
    # Actually, let's pick 0, 120, 240 (3 hexagons) and one more?
    # Or just let the optimizer figure it out from a random-ish start?
    # Better: Start with a valid packing.
    # Let's use the 4 hexagons at angles 0, 90, 180, 270 in a *square* arrangement? No, must be hex lattice compatible for validity.
    # Let's use the 4 hexagons at angles 0, 60, 120, 180 from the 2*sqrt(3) ring.
    for i in range(4):
        a = i * np.pi / 3
        pos_init.append([r2 * np.cos(a), r2 * np.sin(a)])
        
    # This set of 11 positions is valid (no overlaps).
    # Calculate its side length.
    angles_init = [0.0] * N
    pos_init_arr = np.array(pos_init)
    all_verts_init = _hex_vertices_batch(pos_init_arr, angles_init, side=1.0)
    init_side = _min_side_vectorized(all_verts_init, 0.0, 0.0, 0.0)
    
    # Define bounds for DE
    # The centers should be within a reasonable range.
    # Max side length is probably < 10. So centers within +/- 5.
    bounds = [(-5.0, 5.0)] * (2 * N)
    
    # Run Differential Evolution
    best_side = init_side
    best_positions = pos_init_arr.copy()
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-4,
            polish=True,
            init='sobol',
            popsize=15
        )
        
        if result.fun < best_side:
            best_side = result.fun
            best_positions = result.x.reshape(N, 2)
            
    except Exception as e:
        # If DE fails or times out, use the initial packing
        pass
    
    # Final Verification and Construction
    # Ensure the final positions are valid
    final_positions = best_positions
    final_angles = [0.0] * N
    
    # Check validity one last time
    if not _check_no_overlap_fast(final_positions, final_angles):
        # Fallback to initial
        final_positions = pos_init_arr
        final_side = init_side
    else:
        # Recalculate exact side
        all_verts_final = _hex_vertices_batch(final_positions, final_angles, side=1.0)
        final_side = _min_side_vectorized(all_verts_final, 0.0, 0.0, 0.0)
        
        # If the calculated side is worse than initial (due to numerical issues or invalidity), use initial
        if final_side > init_side:
            final_positions = pos_init_arr
            final_side = init_side

    inner_list = [[float(final_positions[i][0]), float(final_positions[i][1]), 0.0] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    # Quick self-test
    inner, center, side, angle = construct()
    print(f"Result: Side={side}, Center={center}, Angle={angle}")
    print(f"Inner: {inner}")
