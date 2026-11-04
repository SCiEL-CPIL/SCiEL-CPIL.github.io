"""Packing 12 unit regular hexagons in a regular hexagon via SAT + multi-start SA."""
import math
import random
import time

N = 12

# Precompute unit hexagon vertices (flat-top, side=1)
# Vertices at angles 0, 60, ..., 300 degrees
_HEX_VERTS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]
# Edge normals (outward) for flat-top hexagon: 6 directions
_HEX_NORMALS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]

def _hex_verts(angle_deg):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(ca * x - sa * y, sa * x + ca * y) for (x, y) in _HEX_VERTS]

def _hex_normals(angle_deg):
    """Return 6 outward edge normals of a unit regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(ca * x - sa * y, sa * x + ca * y) for (x, y) in _HEX_NORMALS]

def _support(verts, nx, ny):
    """Max dot product of vertices with direction (nx,ny)."""
    return max(nx * x + ny * y for (x, y) in verts)

def _min_separation(cx1, cy1, a1, cx2, cy2, a2):
    """Minimum translation distance to separate two unit hexagons (SAT).
    Returns the minimum separation needed (0 if touching/overlapping)."""
    v1 = _hex_verts(a1)
    v2 = _hex_verts(a2)
    # Translate to relative
    v2_rel = [(x - cx1 + cx2 - cx2, y - cy1 + cy2 - cy2) for (x, y) in v2]
    # Actually: v2 relative to v1's center
    dx, dy = cx2 - cx1, cy2 - cy1
    # Project both onto each normal
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    axes = n1 + n2
    min_overlap = 0.0
    for (nx, ny) in axes:
        # Support of hex1 in direction (nx,ny) and (-nx,-ny)
        s1_pos = _support(v1, nx, ny)
        s1_neg = _support(v1, -nx, -ny)
        # Support of hex2 (translated by (dx,dy)) in same directions
        s2_pos = _support(v2, nx, ny) + nx * dx + ny * dy
        s2_neg = _support(v2, -nx, -ny) - nx * dx - ny * dy
        # Gap along this axis: if s1_pos < s2_neg or s2_pos < s1_neg, separated
        # Overlap = min(s1_pos, s2_pos) - max(s1_neg, s2_neg) ... need care
        # Projected intervals: [s1_neg, s1_pos] for hex1, [s2_neg, s2_pos] for hex2
        overlap = min(s1_pos, s2_pos) - max(s1_neg, s2_neg)
        if overlap > min_overlap:
            min_overlap = overlap
    return min_overlap

def _outer_side_length(positions, angles):
    """Compute minimum outer hexagon side length needed to contain all inner hexagons.
    Outer hexagon is centered at origin, flat-top (angle 0), side length S.
    A point (x,y) is inside outer hexagon of side S iff for each of 6 normals n_i:
    n_i . p <= S * apothem_factor... 
    For a regular hexagon with side S, the apothem is S*sqrt(3)/2.
    The support in direction n_i is S*sqrt(3)/2.
    So we need: for each inner hex, for each of 6 outer normals, support <= S*sqrt(3)/2.
    Thus S >= max_support / (sqrt(3)/2) = 2*max_support/sqrt(3).
    """
    apothem_factor = math.sqrt(3) / 2.0
    outer_normals = _HEX_NORMALS  # flat-top outer
    max_support = 0.0
    for (cx, cy), ang in zip(positions, angles):
        verts = _hex_verts(ang)
        # Translate
        tverts = [(x + cx, y + cy) for (x, y) in verts]
        for (nx, ny) in outer_normals:
            s = _support(tverts, nx, ny)
            if s > max_support:
                max_support = s
    return max_support / apothem_factor

def _check_valid(positions, angles):
    """Check no overlaps between inner hexagons."""
    for i in range(N):
        for j in range(i + 1, N):
            sep = _min_separation(positions[i][0], positions[i][1], angles[i],
                                  positions[j][0], positions[j][1], angles[j])
            if sep > 1e-9:
                return False
    return True

def _energy(positions, angles):
    """Energy = outer side length + penalty for overlaps."""
    sl = _outer_side_length(positions, angles)
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = _min_separation(positions[i][0], positions[i][1], angles[i],
                                  positions[j][0], positions[j][1], angles[j])
            if sep > 0:
                penalty += sep * sep * 100.0
    return sl + penalty

def _sa_step(positions, angles, temp, rng):
    """One SA step: perturb a random hexagon's position and angle."""
    idx = rng.randint(0, N - 1)
    dx = rng.gauss(0, 0.3)
    dy = rng.gauss(0, 0.3)
    da = rng.gauss(0, 5.0)
    new_positions = [list(p) for p in positions]
    new_angles = list(angles)
    new_positions[idx][0] += dx
    new_positions[idx][1] += dy
    new_angles[idx] = (new_angles[idx] + da) % 360.0
    return new_positions, new_angles

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    best_result = None
    best_sl = float('inf')
    
    rng = random.Random(42)
    
    # Initial configurations: several multi-starts
    num_starts = 50
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Initialize: random positions in a disk of radius 4, random angles
        positions = []
        angles = []
        for i in range(N):
            r = rng.uniform(0, 4.0)
            a = rng.uniform(0, 2 * math.pi)
            positions.append([r * math.cos(a), r * math.sin(a)])
            angles.append(rng.uniform(0, 360.0))
        
        # SA parameters
        temp = 2.0
        cooling = 0.9995
        min_temp = 0.001
        steps_per_start = 5000
        
        cur_positions = [list(p) for p in positions]
        cur_angles = list(angles)
        cur_energy = _energy(cur_positions, cur_angles)
        
        for step in range(steps_per_start):
            if time.time() > deadline:
                break
            
            new_positions, new_angles = _sa_step(cur_positions, cur_angles, temp, rng)
            new_energy = _energy(new_positions, new_angles)
            
            delta = new_energy - cur_energy
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                cur_positions = new_positions
                cur_angles = new_angles
                cur_energy = new_energy
                
                # Check validity and update best
                if _check_valid(cur_positions, cur_angles):
                    sl = _outer_side_length(cur_positions, cur_angles)
                    if sl < best_sl:
                        best_sl = sl
                        best_result = (cur_positions, list(cur_angles))
            
            temp *= cooling
            if temp < min_temp:
                break
        
        # Final check for this start
        if _check_valid(cur_positions, cur_angles):
            sl = _outer_side_length(cur_positions, cur_angles)
            if sl < best_sl:
                best_sl = sl
                best_result = (cur_positions, list(cur_angles))
    
    # If no valid result found, return a safe fallback
    if best_result is None:
        # Fallback: place hexagons in a grid, large outer
        inner = [[3.5 * (i % 3 - 1), 3.5 * ((i // 3) % 3 - 1), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    positions, angles = best_result
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, (0.0, 0.0), best_sl, 0.0

# EVOLVE-BLOCK-END
