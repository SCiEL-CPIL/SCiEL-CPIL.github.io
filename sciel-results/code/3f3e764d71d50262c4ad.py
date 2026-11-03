"""Structural change: fixed-orientation analytical packing with fast random-restart local search."""
import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3.0)


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


def _min_side_fixed_orientation(pos):
    """
    Compute minimum outer hexagon side length for N unit hexagons (all angle=0)
    centered at positions pos (N,2), with outer hexagon centered at origin, angle=0.
    
    For a unit hexagon with angle=0 (vertices at 0°,60°,120°,180°,240°,300°):
    vertices are at (±1, 0), (±0.5, ±sqrt(3)/2)
    
    For outer hexagon centered at origin with angle=0 and side S:
    A point (px, py) is inside iff:
      px <= S, -px <= S,
      sqrt(3)*px + py <= sqrt(3)*S,
      -sqrt(3)*px + py <= sqrt(3)*S,
      -sqrt(3)*px - py <= sqrt(3)*S,
      sqrt(3)*px - py <= sqrt(3)*S
    
    Required S = max over all inner vertices of:
      px, -px, px + py/sqrt(3), -px + py/sqrt(3), -px - py/sqrt(3), px - py/sqrt(3)
    """
    # All 6 vertices of each unit hexagon (angle=0):
    # (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    offsets = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])  # (6, 2)
    
    # All vertices: (N*6, 2)
    all_verts = pos[:, np.newaxis, :] + offsets[np.newaxis, :, :]  # (N, 6, 2)
    all_verts = all_verts.reshape(-1, 2)  # (N*6, 2)
    
    px = all_verts[:, 0]
    py = all_verts[:, 1]
    
    c1 = px
    c2 = -px
    c3 = px + py / SQRT3
    c4 = -px + py / SQRT3
    c5 = -px - py / SQRT3
    c6 = px - py / SQRT3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))


def _check_no_overlap(pos, tol=1e-7):
    """
    Check no two unit hexagons (angle=0) overlap.
    For two unit hexagons with angle=0, they overlap iff their center distance
    is less than the minimum separation in the direction connecting them.
    Use SAT with 6 axes (3 unique directions due to symmetry).
    """
    n = len(pos)
    # For unit hexagons with angle=0, the support function in direction (dx,dy) is:
    # max over vertices of (v·d). The 6 edge normals are at angles 30,90,150,210,270,330
    # i.e., (sqrt3/2, 1/2), (0,1), (-sqrt3/2, 1/2), (-sqrt3/2,-1/2), (0,-1), (sqrt3/2,-1/2)
    # But for SAT between two identical convex polygons, we need to check all edge normals of both.
    # Since both have the same orientation, we only need 6 axes.
    
    # Edge normals for hexagon with vertices at angles 0,60,...,300:
    # Edges connect consecutive vertices. Normal to edge from vertex i to i+1:
    # The 6 outward normals point at angles 30, 90, 150, 210, 270, 330 degrees
    normals = np.array([
        [SQRT3 / 2.0, 0.5],
        [0.0, 1.0],
        [-SQRT3 / 2.0, 0.5],
        [-SQRT3 / 2.0, -0.5],
        [0.0, -1.0],
        [SQRT3 / 2.0, -0.5]
    ])  # (6, 2)
    
    # For each pair, check SAT
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[i] - pos[j]
            # Project both hexagons onto each normal axis
            # Hex i: center pos[i], vertices pos[i] + offsets
            # Hex j: center pos[j], vertices pos[j] + offsets
            # Projection of hex i onto normal k: pos[i]·n_k + max(offsets·n_k)
            # Projection of hex j onto normal k: pos[j]·n_k + max(offsets·n_k)
            # Since both have same shape, the support in direction n_k is the same for both.
            # So the projections differ only by (pos[i]-pos[j])·n_k
            # SAT: they separate if there exists k such that |diff·n_k| > 2*support(n_k)
            # where support(n_k) = max(offsets·n_k)
            
            # Actually for SAT: project both polygons. 
            # Proj_i on axis k: [pos[i]·n_k + min_off_k, pos[i]·n_k + max_off_k]
            # Proj_j on axis k: [pos[j]·n_k + min_off_k, pos[j]·n_k + max_off_k]
            # They overlap on axis k iff: pos[i]·n_k + max_off_k >= pos[j]·n_k + min_off_k
            # AND pos[j]·n_k + max_off_k >= pos[i]·n_k + min_off_k
            # i.e., (pos[i]-pos[j])·n_k <= max_off_k - min_off_k = 2*support
            # AND (pos[j]-pos[i])·n_k <= 2*support
            # i.e., |(pos[i]-pos[j])·n_k| <= 2*support(n_k)
            
            # Compute support for each normal
            # offsets·n_k for all 6 offsets
            # Due to symmetry of hexagon, support in direction n_k (which is an edge normal)
            # equals the distance from center to that edge = apothem = sqrt(3)/2
            # Wait, let me compute properly.
            
            dot_off = offsets @ normals.T  # (6, 6): offsets[i] · normals[j]
            min_off = dot_off.min(axis=0)  # (6,)
            max_off = dot_off.max(axis=0)  # (6,)
            width = max_off - min_off  # (6,)
            
            proj_diff = diff @ normals.T  # (6,): diff · n_k for each k
            
            # If for ANY axis, |proj_diff| > width, they are separated (no overlap)
            if np.any(np.abs(proj_diff) > width + tol):
                continue  # This pair doesn't overlap, check next pair
            
            # If we get here, no separating axis found -> overlap
            return False
    
    return True


def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    positions = np.array([[x, y] for x, y, a in inner])
    angles = np.array([a for x, y, a in inner])
    
    # Check no overlaps
    if not _check_no_overlap(positions):
        return False
    
    # Check all inside outer hexagon
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            a = np.radians(outer_angle)
            ca, sa = np.cos(a), np.sin(a)
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            s = outer_side
            if lx > s + tol or lx < -s - tol:
                return False
            if SQRT3 * lx + ly > SQRT3 * s + tol:
                return False
            if -SQRT3 * lx + ly > SQRT3 * s + tol:
                return False
            if -SQRT3 * lx - ly > SQRT3 * s + tol:
                return False
            if SQRT3 * lx - ly > SQRT3 * s + tol:
                return False
    return True


def _initial_packing_hex_grid():
    """
    Create initial hexagonal grid packing for N=11 unit hexagons (all angle=0).
    Hex grid with pointy-top hexagons (vertices at 0,60,...):
    Horizontal spacing: 2 (center to center for adjacent in same row)
    Vertical spacing: sqrt(3)
    Odd rows offset by 1 in x.
    
    For N=11, a good arrangement:
    Row 0 (y=0): 4 hexagons at x = -3, -1, 1, 3
    Row 1 (y=sqrt3): 4 hexagons at x = -2, 0, 2, 4  -> offset by 1
    Row 2 (y=2*sqrt3): 3 hexagons at x = -1, 1, 3
    
    Actually let's use a more symmetric arrangement.
    Center at origin.
    
    Better: use axial coordinates for hex grid.
    For pointy-top hexagons with side 1:
    Axial (q, r) -> pixel: x = sqrt(3)*(q + r/2), y = 1.5*r
    Wait, that's for flat-top. For pointy-top:
    x = 1.5 * q, y = sqrt(3) * (r + q/2)
    
    Let me just use explicit positions that are known to work.
    Unit hexagons with angle=0 (pointy-top, vertices at 0,60,120,180,240,300):
    The "radius" (center to vertex) is 1.
    Adjacent hexagons in a grid: center distance = 2.
    
    For a hex grid of pointy-top hexagons:
    - Same row: horizontal spacing = 2
    - Adjacent row: horizontal offset = 1, vertical spacing = sqrt(3)
    
    Let's place 11 hexagons in a roughly circular cluster:
    Center (0,0)
    6 neighbors at distance 2: (2,0), (1,sqrt3), (-1,sqrt3), (-2,0), (-1,-sqrt3), (1,-sqrt3)
    That's 7. Need 4 more.
    Next ring positions (distance from center ~ 2*sqrt(3) or so):
    (4,0), (3,sqrt3), (0,2*sqrt3), (-3,sqrt3), (-4,0), (-3,-sqrt3), (0,-2*sqrt3), (3,-sqrt3)
    Pick 4 that fit well: (4,0), (-4,0), (0,2*sqrt3), (0,-2*sqrt3)
    But (4,0) and (2,0) have distance 2, which is exactly the minimum -> they touch but don't overlap.
    Let's verify: (2,0) and (4,0) distance = 2. For two unit hexagons side by side horizontally,
    the rightmost vertex of left hex is at x=3, leftmost of right hex is at x=3. They touch.
    So they don't overlap (touching is OK for disjoint? The task says "disjoint" which typically means
    no interior overlap, touching is allowed).
    
    Actually, let me re-read: "Place disjoint regular hexagons" - disjoint means no overlap in interior.
    Touching boundaries should be fine.
    
    Let's use this 11-hexagon arrangement.
    """
    positions = [
        [0.0, 0.0],           # center
        [2.0, 0.0],           # right
        [1.0, SQRT3],         # upper-right
        [-1.0, SQRT3],        # upper-left
        [-2.0, 0.0],          # left
        [-1.0, -SQRT3],       # lower-left
        [1.0, -SQRT3],        # lower-right
        [4.0, 0.0],           # far right
        [-4.0, 0.0],          # far left
        [0.0, 2.0 * SQRT3],   # top
        [0.0, -2.0 * SQRT3],  # bottom
    ]
    return np.array(positions, dtype=np.float64)


def _local_search(pos, step=0.1, max_iter=500, tol=1e-9):
    """
    Local search: try small moves of each hexagon, accept if outer side decreases
    and no overlap is introduced.
    """
    n = len(pos)
    current_side = _min_side_fixed_orientation(pos)
    current_pos = pos.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        
        improved = False
        # Try moving each hexagon in 4 directions (and maybe 8)
        for i in range(n):
            for dx in [-step, step]:
                for dy in [-step, step]:
                    new_pos = current_pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Quick overlap check
                    if not _check_no_overlap(new_pos):
                        continue
                    
                    new_side = _min_side_fixed_orientation(new_pos)
                    if new_side < current_side - tol:
                        current_pos = new_pos
                        current_side = new_side
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try diagonal moves
            for i in range(n):
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pos = current_pos.copy()
                        new_pos[i, 0] += dx
                        new_pos[i, 1] += dy
                        if not _check_no_overlap(new_pos):
                            continue
                        new_side = _min_side_fixed_orientation(new_pos)
                        if new_side < current_side - tol:
                            current_pos = new_pos
                            current_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    return current_pos, current_side


def _random_packing(rng):
    """Generate a random valid-ish packing for restarts."""
    # Start from hex grid and add random perturbations
    base = _initial_packing_hex_grid()
    perturb = rng.normal(0, 0.3, size=base.shape)
    pos = base + perturb
    return pos


def construct():
    global START_TIME, DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Initial packing
    pos = _initial_packing_hex_grid()
    current_side = _min_side_fixed_orientation(pos)
    
    best_side = current_side
    best_pos = pos.copy()
    
    # Phase 1: Local search from initial packing
    pos, side = _local_search(pos, step=0.2, max_iter=300)
    if side < best_side:
        best_side = side
        best_pos = pos.copy()
    
    # Phase 2: Multiple random restarts with local search
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10.0:
            break
        
        # Generate random perturbation of the best packing
        scale = 0.5 + 0.3 * rng.random()
        perturbed = best_pos + rng.normal(0, scale, size=best_pos.shape)
        
        # Check if this random perturbation is valid (no overlaps)
        if not _check_no_overlap(perturbed):
            # Try smaller perturbation
            perturbed = best_pos + rng.normal(0, 0.1, size=best_pos.shape)
            if not _check_no_overlap(perturbed):
                continue
        
        # Local search from this perturbation
        pos_r, side_r = _local_search(perturbed, step=0.1, max_iter=200)
        
        if side_r < best_side:
            best_side = side_r
            best_pos = pos_r.copy()
    
    # Phase 3: Fine-tuning with very small steps
    if time.time() < DEADLINE - 5.0:
        pos, side = _local_search(best_pos, step=0.05, max_iter=200)
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
        
        pos, side = _local_search(best_pos, step=0.01, max_iter=200)
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
    
    # Phase 4: Even finer
    if time.time() < DEADLINE - 3.0:
        pos, side = _local_search(best_pos, step=0.005, max_iter=100)
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
    
    # Final verification
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), 0.0] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    if not _check_validity(inner_list, outer_center, best_side, outer_angle):
        # Fallback: use the initial packing
        pos_init = _initial_packing_hex_grid()
        side_init = _min_side_fixed_orientation(pos_init)
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), 0.0] for i in range(N)]
        best_side = side_init
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0


# EVOLVE-BLOCK-END
