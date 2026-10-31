import math
import random
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Hex lattice vectors for flat-top hexes
    # u = (1.5, sqrt(3)/2)
    # v = (0, sqrt(3))
    # w = (1.5, -sqrt(3)/2)
    
    # We will search in axial coordinates (q, r)
    # Cartesian: x = 1.5 * q, y = sqrt(3)/2 * (2*r + q) ? 
    # Let's use standard axial to cartesian for pointy-top?
    # Let's stick to the flat-top lattice defined by vectors u and v.
    # Point (i, j) -> i * u + j * v
    # u = (1.5, sqrt(3)/2)
    # v = (0, sqrt(3))
    
    def lattice_to_cartesian(i, j):
        x = 1.5 * i
        y = (math.sqrt(3)/2) * i + math.sqrt(3) * j
        return x, y

    def get_hex_vertices(cx, cy):
        # Flat-top hex vertices relative to center
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        verts = []
        for k in range(6):
            angle = math.pi / 3 * k
            x = cx + math.cos(angle)
            y = cy + math.sin(angle)
            verts.append((x, y))
        return verts

    def calc_bounding_side(positions):
        # positions: list of (x, y)
        # Calculate max of the 6 linear forms for all vertices
        max_vals = [0.0] * 6
        
        for (cx, cy) in positions:
            verts = get_hex_vertices(cx, cy)
            for (x, y) in verts:
                # 1. y
                max_vals[0] = max(max_vals[0], y)
                # 2. sqrt(3)x + y
                max_vals[1] = max(max_vals[1], math.sqrt(3)*x + y)
                # 3. -sqrt(3)x + y
                max_vals[2] = max(max_vals[2], -math.sqrt(3)*x + y)
                # 4. -y
                max_vals[3] = max(max_vals[3], -y)
                # 5. -sqrt(3)x - y
                max_vals[4] = max(max_vals[4], -math.sqrt(3)*x - y)
                # 6. sqrt(3)x - y
                max_vals[5] = max(max_vals[5], math.sqrt(3)*x - y)
        
        s1 = (2.0 / math.sqrt(3)) * max_vals[0]
        s2 = (1.0 / math.sqrt(3)) * max_vals[1]
        s3 = (1.0 / math.sqrt(3)) * max_vals[2]
        s4 = (2.0 / math.sqrt(3)) * max_vals[3]
        s5 = (1.0 / math.sqrt(3)) * max_vals[4]
        s6 = (1.0 / math.sqrt(3)) * max_vals[5]
        
        return max(s1, s2, s3, s4, s5, s6)

    # Initial configuration: 7 hex cluster + 4 random
    # Lattice points for 7 cluster:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) ?
    # Let's check neighbors in (i,j) lattice.
    # Neighbors of (0,0):
    # u: (1,0)
    # v: (0,1)
    # u-v: (1,-1)
    # -u: (-1,0)
    # -v: (0,-1)
    # v-u: (-1,1)
    
    initial_points = [(0,0), (1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)]
    
    # Add 4 more
    # Candidates: (2,0), (0,2), (2,-1), (-2,0), (0,-2), (-2,1), (1,1), (-1,-1)
    # Let's pick (2,0), (0,2), (2,-1), (-2,0)
    # Or just pick random empty spots near the cluster
    
    current_points = initial_points.copy()
    # Add 4 more
    candidates = [(2,0), (0,2), (2,-1), (-2,0), (0,-2), (-2,1), (1,1), (-1,-1), (2,1), (-2,-1)]
    random.shuffle(candidates)
    for p in candidates:
        if p not in current_points:
            current_points.append(p)
            if len(current_points) == N:
                break
    
    # Ensure we have N points
    if len(current_points) < N:
        # Fill with random
        while len(current_points) < N:
            i = random.randint(-5, 5)
            j = random.randint(-5, 5)
            if (i,j) not in current_points:
                current_points.append((i,j))

    best_side = calc_bounding_side([lattice_to_cartesian(i, j) for i, j in current_points])
    best_points = current_points.copy()
    
    # Local Search
    # Moves: pick a random point, move it to a random neighbor (or random point in range)
    # To ensure no overlap, we just check if the new point is in the list.
    # We don't explicitly check geometric overlap because lattice points are non-overlapping
    # for unit hexes on this lattice?
    # Distance between adjacent lattice points is sqrt(3).
    # Hex width is 2 (flat-top).
    # Wait, flat-top hex width is 2. Distance between centers is sqrt(3) ~ 1.732.
    # So they DO overlap!
    # My lattice definition was for TOUCHING hexes.
    # If I use this lattice, the hexes touch and do not overlap.
    # So any set of distinct lattice points is a valid non-overlapping configuration.
    
    temperature = 1.0
    cooling = 0.99
    
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Pick a random index
        idx = random.randint(0, N-1)
        old_i, old_j = current_points[idx]
        
        # Pick a new random point
        # Bias towards existing cluster
        if random.random() < 0.5:
            # Move to a neighbor of a random other point
            ref_i, ref_j = current_points[random.randint(0, N-1)]
            new_i = ref_i + random.randint(-2, 2)
            new_j = ref_j + random.randint(-2, 2)
        else:
            new_i = random.randint(-6, 6)
            new_j = random.randint(-6, 6)
            
        if (new_i, new_j) in current_points:
            continue
            
        # Swap
        current_points[idx] = (new_i, new_j)
        
        new_side = calc_bounding_side([lattice_to_cartesian(i, j) for i, j in current_points])
        
        if new_side < best_side:
            best_side = new_side
            best_points = current_points.copy()
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_side - best_side
            if delta > 0:
                prob = math.exp(-delta / temperature)
                if random.random() > prob:
                    # Reject
                    current_points[idx] = (old_i, old_j)
            
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    # Convert best_points to Cartesian
    inner = []
    for i, j in best_points:
        x, y = lattice_to_cartesian(i, j)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
