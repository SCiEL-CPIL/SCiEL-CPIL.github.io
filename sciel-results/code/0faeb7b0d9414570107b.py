"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def _point_in_hex(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with side and rotation."""
    # Translate point to hexagon center
    dx = px - hx
    dy = py - hy
    # Rotate by -angle to align hexagon
    a = -math.radians(angle_deg)
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    # Now check if (rx, ry) is in unit hexagon scaled by side
    # Hexagon with vertices at angles 0,60,...,300 degrees
    # The 6 half-planes: for each edge, the normal points outward
    # Edge normals at angles 30, 90, 150, 210, 270, 330 degrees
    # Apothem = side * sqrt(3)/2
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        na = math.radians(30 + i * 60)
        nx = math.cos(na)
        ny = math.sin(na)
        if rx * nx + ry * ny > apothem:
            return False
    return True

def _hexes_overlap(v1, v2):
    """Check if two hexagons (as arrays of 6 vertices) overlap using SAT."""
    # Get all edge normals from both hexagons
    normals = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        # Project both hexagons onto this normal
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found, no overlap
    return True

def _all_inside(inner_verts, outer_verts):
    """Check if all vertices of each inner hex are inside the outer hexagon."""
    for iv in inner_verts:
        for pt in iv:
            if not _point_in_hex(pt[0], pt[1], outer_verts[:, 0].mean(), outer_verts[:, 1].mean(), 
                                  0, 0):  # This won't work directly
                pass
    # Better: check point in polygon for outer hex
    # Use the half-plane method for the outer hexagon
    # But we need the outer hex parameters, not just vertices
    return True  # Placeholder, will fix

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # State: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, ox, oy, oside, oangle]
    # 12*3 + 4 = 40 parameters
    
    # Initial placement: hexagonal grid
    # Use axial coordinates for a compact 12-hexagon arrangement
    # Rows: r=-1 (4 hex), r=0 (4 hex), r=1 (4 hex) - not quite hexagonal
    # Let me use a 3x4 grid with hexagonal offset
    hex_size = 1.0
    spacing_x = hex_size * math.sqrt(3)  # distance between centers horizontally
    spacing_y = hex_size * 1.5           # distance between centers vertically
    
    # 4 columns, 3 rows with hexagonal offset
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * spacing_x + (row % 2) * spacing_x / 2
            y = row * spacing_y
            positions.append((x, y, 0.0))
    
    # Center the arrangement
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [(p[0] - cx, p[1] - cy, p[2]) for p in positions]
    
    # Initial outer hexagon: large enough to contain all
    max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in positions) + 2.0
    outer_side = max_r * 2.0 / math.sqrt(3)  # apothem = side*sqrt(3)/2, so side = 2*apothem/sqrt(3)
    
    state = np.zeros(40)
    for i, (x, y, a) in enumerate(positions):
        state[i*3] = x
        state[i*3+1] = y
        state[i*3+2] = a
    state[36] = 0.0  # outer center x
    state[37] = 0.0  # outer center y
    state[38] = outer_side
    state[39] = 0.0  # outer angle
    
    def evaluate(state):
        """Return (valid, objective) where objective is outer_side if valid, else inf."""
        ox, oy = state[36], state[37]
        oside = state[38]
        oangle = state[39]
        
        if oside < 0.1:
            return False, float('inf')
        
        # Get outer hexagon vertices
        outer_verts = _hex_vertices(ox, oy, oside, oangle)
        
        # Get inner hexagon vertices
        inner_verts = []
        for i in range(N):
            x, y, a = state[i*3], state[i*3+1], state[i*3+2]
            v = _hex_vertices(x, y, 1.0, a)
            inner_verts.append(v)
        
        # Check containment: all inner vertices must be inside outer hexagon
        # Use half-plane test for outer hexagon
        # Outer hexagon face normals
        a_rad = math.radians(oangle)
        for i in range(6):
            na = a_rad + math.radians(30 + i * 60)
            nx = math.cos(na)
            ny = math.sin(na)
            apothem = oside * math.sqrt(3) / 2.0
            for iv in inner_verts:
                for pt in iv:
                    dx = pt[0] - ox
                    dy = pt[1] - oy
                    if dx * nx + dy * ny > apothem + 1e-9:
                        return False, float('inf')
        
        # Check non-overlap of inner hexagons
        for i in range(N):
            for j in range(i+1, N):
                if _hexes_overlap(inner_verts[i], inner_verts[j]):
                    return False, float('inf')
        
        return True, oside
    
    # Simulated Annealing
    best_state = state.copy()
    best_obj = float('inf')
    current_obj = float('inf')
    
    # Evaluate initial
    valid, obj = evaluate(state)
    if valid:
        best_state = state.copy()
        best_obj = obj
        current_obj = obj
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.001
    
    np.random.seed(42)
    
    step_count = 0
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Generate perturbation
        new_state = state.copy()
        
        # Perturb a random inner hexagon
        idx = np.random.randint(N)
        new_state[idx*3] += np.random.normal(0, 0.5)
        new_state[idx*3+1] += np.random.normal(0, 0.5)
        new_state[idx*3+2] += np.random.uniform(-30, 30)
        
        # Sometimes perturb outer hexagon
        if np.random.random() < 0.3:
            new_state[36] += np.random.normal(0, 0.3)
            new_state[37] += np.random.normal(0, 0.3)
            new_state[39] += np.random.uniform(-15, 15)
        
        valid, obj = evaluate(new_state)
        
        if valid:
            if obj < current_obj:
                state = new_state
                current_obj = obj
                if obj < best_obj:
                    best_state = state.copy()
                    best_obj = obj
            else:
                # Accept with probability exp((current_obj - obj)/T)
                if np.random.random() < math.exp((current_obj - obj) / T):
                    state = new_state
                    current_obj = obj
        else:
            # Reject invalid
            pass
        
        T *= cooling
        step_count += 1
        
        # If stuck, try a bigger jump
        if step_count % 500 == 0 and current_obj > best_obj * 1.1:
            # Reset to best with some noise
            state = best_state + np.random.normal(0, 0.2, 40)
            state[36] = best_state[36]
            state[37] = best_state[37]
            state[39] = best_state[39]
            valid, obj = evaluate(state)
            if valid:
                current_obj = obj
    
    # Final refinement: coordinate descent on best state
    if best_obj < float('inf'):
        for iteration in range(200):
            if time.time() > deadline:
                break
            improved = False
            for param_idx in range(40):
                old_val = best_state[param_idx]
                for delta in [0.01, -0.01, 0.001, -0.001]:
                    trial = best_state.copy()
                    trial[param_idx] = old_val + delta
                    valid, obj = evaluate(trial)
                    if valid and obj < best_obj:
                        best_state = trial
                        best_obj = obj
                        improved = True
                        break
            if not improved:
                break
    
    # Extract final answer
    inner = []
    for i in range(N):
        x = best_state[i*3]
        y = best_state[i*3+1]
        a = best_state[i*3+2]
        inner.append([x, y, a])
    
    outer_center = (best_state[36], best_state[37])
    outer_side = best_state[38]
    outer_angle = best_state[39]
    
    # Ensure the result is valid; if not, fall back to seed
    valid, obj = evaluate(best_state)
    if not valid:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
