import math
import random
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin for safety

    # Precompute normals for the outer hexagon (flat-top, vertex at (S,0))
    # Normals are perpendicular to edges.
    # Edges are between vertices.
    # Vertex 0: (S, 0)
    # Vertex 1: (S/2, S*sqrt(3)/2)
    # Edge 0-1 midpoint: (3S/4, S*sqrt(3)/4). Normal direction?
    # Vector (1, sqrt(3)). Unit normal (1/2, sqrt(3)/2). Angle 60 deg?
    # Let's check distance.
    # Apothem A = S * sqrt(3) / 2.
    # The normals to the sides of a regular hexagon (vertex at 0 deg) are at 30, 90, 150, 210, 270, 330 degrees.
    angles = [math.radians(a) for a in [30, 90, 150, 210, 270, 330]]
    normals = [(math.cos(a), math.sin(a)) for a in angles]
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Returns list of (x,y) for a regular hexagon of given side, center, and rotation."""
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        verts = []
        for i in range(6):
            theta = rad + math.radians(60 * i)
            # Vertices of unit hexagon (side 1) centered at 0 are at distance 1 from center?
            # No, for a regular hexagon, side length = radius (distance center to vertex).
            # So yes, radius = side length.
            vx = side * math.cos(theta)
            vy = side * math.sin(theta)
            # Rotate and translate
            rx = vx * cos_a - vy * sin_a + cx
            ry = vx * sin_a + vy * cos_a + cy
            verts.append((rx, ry))
        return verts

    def hexagon_overlap(v1, v2):
        """
        Check if two regular hexagons (defined by lists of 6 vertices) overlap.
        Uses Separating Axis Theorem (SAT).
        """
        # Axes to check: normals to all edges of both hexagons.
        # Since they are regular hexagons, we can just check the 6 axes from one and 6 from the other.
        # Actually, for convex polygons, we check axes perpendicular to edges.
        # For regular hexagons, the edge normals are the same set of directions (rotated).
        # We can generate axes from v1 and v2.
        
        def get_axes(verts):
            axes = []
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                # Edge vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normal
                nx = -dy
                ny = dx
                # Normalize
                norm = math.sqrt(nx*nx + ny*ny)
                if norm > 1e-9:
                    axes.append((nx/norm, ny/norm))
            return axes

        axes = get_axes(v1) + get_axes(v2)
        
        for nx, ny in axes:
            # Project v1
            min1 = min(p[0]*nx + p[1]*ny for p in v1)
            max1 = max(p[0]*nx + p[1]*ny for p in v1)
            # Project v2
            min2 = min(p[0]*nx + p[1]*ny for p in v2)
            max2 = max(p[0]*nx + p[1]*ny for p in v2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False, 0.0
            
            # Calculate overlap depth for penalty
            # Overlap is min(max1, max2) - max(min1, min2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return False, 0.0
            # We return True if overlapping.
            # To make penalty smooth, we might want to accumulate max overlap?
            # For SAT, if no axis separates, they overlap.
        
        return True, 1.0 # Return a simple boolean/penalty indicator

    def calculate_objective(states):
        """
        states: list of (x, y, angle_deg)
        Returns (S_required, penalty)
        """
        all_verts = []
        for i in range(N):
            x, y, ang = states[i]
            verts = get_hex_vertices(x, y, ang)
            all_verts.append(verts)
        
        # Calculate required S for outer hexagon
        # S = (2/sqrt(3)) * max( n_i . v ) for all vertices v
        max_proj = 0.0
        for verts in all_verts:
            for v in verts:
                for nx, ny in normals:
                    proj = v[0]*nx + v[1]*ny
                    if proj > max_proj:
                        max_proj = proj
        
        S_required = (2.0 / math.sqrt(3.0)) * max_proj
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                overlaps, _ = hexagon_overlap(all_verts[i], all_verts[j])
                if overlaps:
                    penalty += 10.0 # Large penalty to discourage overlaps
        
        return S_required, penalty

    # Initial state: Random positions in a small area, random angles
    # Start with a guess for S. The area of 12 unit hexagons is 12 * (3*sqrt(3)/2) ~ 31.
    # Area of outer hex side S is (3*sqrt(3)/2) * S^2.
    # S^2 >= 12 => S >= sqrt(12) ~ 3.46.
    # Realistically S will be around 4-5.
    
    best_states = []
    best_score = float('inf')
    best_S = float('inf')
    
    # Initialize
    current_states = []
    for _ in range(N):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        ang = random.uniform(0, 360)
        current_states.append([x, y, ang])
    
    curr_S, curr_pen = calculate_objective(current_states)
    curr_score = curr_S + curr_pen * 100.0 # Weight penalty heavily
    
    if curr_pen == 0:
        best_states = [list(s) for s in current_states]
        best_score = curr_score
        best_S = curr_S

    T = 0.5
    T_min = 0.001
    alpha = 0.999
    
    step = 0
    while time.time() < deadline and T > T_min:
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        
        # Random perturbation
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        d_ang = random.gauss(0, 10.0)
        
        new_states = [list(s) for s in current_states]
        new_states[idx][0] += dx
        new_states[idx][1] += dy
        new_states[idx][2] += d_ang
        
        new_S, new_pen = calculate_objective(new_states)
        new_score = new_S + new_pen * 100.0
        
        delta = new_score - curr_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_states = new_states
            curr_S = new_S
            curr_pen = new_pen
            curr_score = new_score
            
            if curr_pen == 0 and curr_S < best_S:
                best_S = curr_S
                best_score = curr_score
                best_states = [list(s) for s in current_states]
        
        T *= alpha
        step += 1
        if step % 1000 == 0:
            # Optional: Reheat or just continue
            pass

    # Final result
    # Return inner rows (x, y, angle_degrees)
    inner = [[s[0], s[1], s[2]] for s in best_states]
    center = (0.0, 0.0)
    side = best_S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
