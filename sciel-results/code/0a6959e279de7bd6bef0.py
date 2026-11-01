import random
import numpy as np
import math
import time

N = 12

def hexagon_vertices(cx, cy, angle_deg, s=1.0):
    """Return 6 vertices of a regular hexagon of side s centered at (cx, cy) rotated by angle_deg."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def min_distance_polygons(verts1, verts2):
    """Calculate minimum distance between two convex polygons (hexagons).
    Returns 0 if they intersect or touch."""
    # For hexagons, we can check if any vertex of 1 is inside 2 or vice versa,
    # or if edges intersect.
    # Simpler: Use the Separating Axis Theorem (SAT) or just check all pairs of points?
    # Since they are convex, SAT is efficient.
    # Axes are normals to the edges of both polygons.
    
    def project(verts, axis):
        dots = [dot(v, axis) for v in verts]
        return min(dots), max(dots)
    
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]
        
    def subtract(a, b):
        return (a[0]-b[0], a[1]-b[1])
        
    def norm(a):
        return math.sqrt(a[0]**2 + a[1]**2)
        
    def normalize(a):
        n = norm(a)
        if n == 0: return (0,0)
        return (a[0]/n, a[1]/n)
        
    axes = []
    for i in range(6):
        # Edge normal for verts1
        p1 = verts1[i]
        p2 = verts1[(i+1)%6]
        edge = subtract(p2, p1)
        # Normal is perpendicular to edge
        normal = (-edge[1], edge[0])
        axes.append(normalize(normal))
        
    for i in range(6):
        p1 = verts2[i]
        p2 = verts2[(i+1)%6]
        edge = subtract(p2, p1)
        normal = (-edge[1], edge[0])
        axes.append(normalize(normal))
        
    min_dist = float('inf')
    for axis in axes:
        min1, max1 = project(verts1, axis)
        min2, max2 = project(verts2, axis)
        
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            return 0.0 # Overlap or touch
            
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def get_required_side(verts_list):
    """Calculate the minimal side length S of an outer hexagon (fixed orientation) 
    that contains all vertices in verts_list.
    Outer hexagon: centered at (0,0), vertices at angles 0, 60, ..., 300 degrees.
    Apothem A = S * sqrt(3) / 2.
    Condition: x*cos(a) + y*sin(a) <= A for a in [30, 90, 150, 210, 270, 330] degrees.
    """
    # Angles of the normals to the sides of the outer hexagon
    # If vertices are at 0, 60, ..., then sides are perpendicular to 30, 90, ...
    angles = [math.radians(30 + i*60) for i in range(6)]
    coss = [math.cos(a) for a in angles]
    sins = [math.sin(a) for a in angles]
    
    max_proj = 0.0
    for v in verts_list:
        x, y = v
        for i in range(6):
            proj = x * coss[i] + y * sins[i]
            if proj > max_proj:
                max_proj = proj
                
    # max_proj is the maximum distance from center to any side in the "support" sense.
    # We need max_proj <= A = S * sqrt(3) / 2
    # So S >= 2 * max_proj / sqrt(3)
    if max_proj <= 0:
        return 0.0
    return 2.0 * max_proj / math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_config = None
    
    # Initial configurations
    # 1. Hexagonal lattice
    # 2. Random cluster
    
    configs = []
    
    # Lattice
    # Center at 0,0.
    # 1, 6, 12? 
    # 1 center, 6 around, 5 more?
    # Let's just use a random grid or cluster.
    
    def random_config():
        # Place centers in a small disk
        centers = []
        angles = []
        for i in range(N):
            r = 1.0 * math.sqrt(i) # Rough spacing
            theta = i * 2.0 # Golden angle-ish?
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            centers.append([cx, cy])
            angles.append(0.0)
        return centers, angles
        
    # Generate a few initial states
    for _ in range(5):
        configs.append(random_config())
        
    # Also try a specific lattice
    # 1 center
    c0 = [0,0]
    a0 = [0]
    # 6 around
    for i in range(6):
        th = i * math.pi / 3
        c = [2.0 * math.cos(th), 2.0 * math.sin(th)] # Side 1, distance between centers 2?
        # Distance between centers of touching hexagons is 2*sqrt(3)/2? No.
        # Flat to flat distance is sqrt(3). Vertex to vertex is 2.
        # If aligned, distance 2.
        centers = [c0]
        angles = [a0]
        for i in range(6):
            th = i * math.pi / 3
            centers.append([2.0 * math.cos(th), 2.0 * math.sin(th)])
            angles.append(0.0)
        # 5 more
        for i in range(5):
            th = (i + 0.5) * math.pi / 3
            # Place further out
            centers.append([4.0 * math.cos(th), 4.0 * math.sin(th)])
            angles.append(0.0)
        configs.append((centers, angles))
        
    for init_idx, (centers, angles) in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Local search
        # We will use a simple iterative improvement.
        # Step size starts large, decays.
        
        current_centers = [c[:] for c in centers]
        current_angles = angles[:]
        
        # Calculate initial score
        all_verts = []
        for i in range(N):
            v = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
            all_verts.extend(v)
        S = get_required_side(all_verts)
        
        # Check overlaps
        def check_overlaps(centers, angles):
            for i in range(N):
                v1 = hexagon_vertices(centers[i][0], centers[i][1], angles[i])
                for j in range(i+1, N):
                    v2 = hexagon_vertices(centers[j][0], centers[j][1], angles[j])
                    d = min_distance_polygons(v1, v2)
                    if d < 1e-6:
                        return False
            return True
            
        if not check_overlaps(current_centers, current_angles):
            # If initial config overlaps, we need to resolve it first.
            # For simplicity, skip this config or do a repair step.
            # Let's do a repair step.
            pass # We'll handle overlaps in the loop by penalizing.
            
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Calculate current state
            all_verts = []
            for i in range(N):
                v = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                all_verts.extend(v)
            S = get_required_side(all_verts)
            
            # Check overlaps
            overlap_count = 0
            forces = [[0.0, 0.0] for _ in range(N)]
            
            for i in range(N):
                v1 = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                for j in range(i+1, N):
                    v2 = hexagon_vertices(current_centers[j][0], current_centers[j][1], current_angles[j])
                    d = min_distance_polygons(v1, v2)
                    if d < 1e-6:
                        overlap_count += 1
                        # Calculate push direction
                        # Vector from center i to center j
                        dx = current_centers[j][0] - current_centers[i][0]
                        dy = current_centers[j][1] - current_centers[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 1e-6:
                            nx = dx / dist
                            ny = dy / dist
                        else:
                            nx, ny = 1.0, 0.0
                        # Push apart
                        forces[i][0] -= nx
                        forces[i][1] -= ny
                        forces[j][0] += nx
                        forces[j][1] += ny
            
            # Attractive force to center to minimize S
            # We want to move points that are far from center towards center.
            # Approximate gradient of S: S is max of linear functions.
            # The active constraint is the vertex that determines S.
            # Let's just pull everything towards origin with a small force.
            pull_strength = 0.01
            for i in range(N):
                forces[i][0] -= pull_strength * current_centers[i][0]
                forces[i][1] -= pull_strength * current_centers[i][1]
                
            # Update positions
            step_size = 0.1 * (1.0 - step/500.0) + 0.01
            for i in range(N):
                current_centers[i][0] += forces[i][0] * step_size
                current_centers[i][1] += forces[i][1] * step_size
                # Random rotation perturbation
                if step % 10 == 0:
                    current_angles[i] += (random.random() - 0.5) * 0.1
                    
            # Evaluate
            if overlap_count == 0:
                all_verts = []
                for i in range(N):
                    v = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                    all_verts.extend(v)
                S_new = get_required_side(all_verts)
                if S_new < best_score:
                    best_score = S_new
                    best_config = ([c[:] for c in current_centers], current_angles[:])
                    
    if best_config is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles = best_config
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    # Calculate final S
    all_verts = []
    for i in range(N):
        v = hexagon_vertices(centers[i][0], centers[i][1], angles[i])
        all_verts.extend(v)
    S = get_required_side(all_verts)
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import math
    import time
    import random
    
    N = 12
    
    def hexagon_vertices(cx, cy, angle_deg, s=1.0):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + s * math.cos(theta)
            y = cy + s * math.sin(theta)
            verts.append((x, y))
        return verts

    def min_distance_polygons(verts1, verts2):
        def project(verts, axis):
            dots = [v[0]*axis[0] + v[1]*axis[1] for v in verts]
            return min(dots), max(dots)
            
        def subtract(a, b):
            return (a[0]-b[0], a[1]-b[1])
            
        def norm(a):
            return math.sqrt(a[0]**2 + a[1]**2)
            
        def normalize(a):
            n = norm(a)
            if n == 0: return (0,0)
            return (a[0]/n, a[1]/n)
            
        axes = []
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1)%6]
            edge = subtract(p2, p1)
            normal = (-edge[1], edge[0])
            axes.append(normalize(normal))
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i+1)%6]
            edge = subtract(p2, p1)
            normal = (-edge[1], edge[0])
            axes.append(normalize(normal))
            
        min_dist = float('inf')
        for axis in axes:
            min1, max1 = project(verts1, axis)
            min2, max2 = project(verts2, axis)
            if max1 < min2:
                dist = min2 - max1
            elif max2 < min1:
                dist = min1 - max2
            else:
                return 0.0
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def get_required_side(verts_list):
        angles = [math.radians(30 + i*60) for i in range(6)]
        coss = [math.cos(a) for a in angles]
        sins = [math.sin(a) for a in angles]
        max_proj = 0.0
        for v in verts_list:
            x, y = v
            for i in range(6):
                proj = x * coss[i] + y * sins[i]
                if proj > max_proj:
                    max_proj = proj
        if max_proj <= 0:
            return 0.0
        return 2.0 * max_proj / math.sqrt(3.0)

    def check_overlaps(centers, angles):
        for i in range(N):
            v1 = hexagon_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i+1, N):
                v2 = hexagon_vertices(centers[j][0], centers[j][1], angles[j])
                d = min_distance_polygons(v1, v2)
                if d < 1e-6:
                    return False
        return True

    start_time = time.time()
    deadline = start_time + 110.0
    best_score = float('inf')
    best_config = None
    
    # Generate initial configurations
    def random_config():
        centers = []
        angles = []
        # Use a hexagonal lattice as base, then perturb
        # 1 center
        centers.append([0,0])
        angles.append(0.0)
        # 6 around
        for i in range(6):
            th = i * math.pi / 3
            centers.append([2.0 * math.cos(th), 2.0 * math.sin(th)])
            angles.append(0.0)
        # 5 more
        for i in range(5):
            th = (i + 0.5) * math.pi / 3
            centers.append([4.0 * math.cos(th), 4.0 * math.sin(th)])
            angles.append(0.0)
            
        # Perturb
        for i in range(N):
            centers[i][0] += (random.random() - 0.5) * 0.5
            centers[i][1] += (random.random() - 0.5) * 0.5
            angles[i] = random.random() * 60.0
        return centers, angles

    configs = []
    for _ in range(10):
        configs.append(random_config())
        
    for init_idx, (centers, angles) in enumerate(configs):
        if time.time() > deadline:
            break
            
        current_centers = [c[:] for c in centers]
        current_angles = angles[:]
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Calculate forces
            forces = [[0.0, 0.0] for _ in range(N)]
            overlap_count = 0
            
            for i in range(N):
                v1 = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                for j in range(i+1, N):
                    v2 = hexagon_vertices(current_centers[j][0], current_centers[j][1], current_angles[j])
                    d = min_distance_polygons(v1, v2)
                    if d < 1e-6:
                        overlap_count += 1
                        dx = current_centers[j][0] - current_centers[i][0]
                        dy = current_centers[j][1] - current_centers[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 1e-6:
                            nx = dx / dist
                            ny = dy / dist
                        else:
                            nx, ny = 1.0, 0.0
                        forces[i][0] -= nx
                        forces[i][1] -= ny
                        forces[j][0] += nx
                        forces[j][1] += ny
            
            # Pull to center
            pull_strength = 0.05
            for i in range(N):
                forces[i][0] -= pull_strength * current_centers[i][0]
                forces[i][1] -= pull_strength * current_centers[i][1]
                
            # Update
            step_size = 0.05 * (1.0 - step/1000.0) + 0.01
            for i in range(N):
                current_centers[i][0] += forces[i][0] * step_size
                current_centers[i][1] += forces[i][1] * step_size
                if step % 5 == 0:
                    current_angles[i] += (random.random() - 0.5) * 0.2
                    
            # Check validity and score
            if overlap_count == 0:
                all_verts = []
                for i in range(N):
                    v = hexagon_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                    all_verts.extend(v)
                S = get_required_side(all_verts)
                if S < best_score:
                    best_score = S
                    best_config = ([c[:] for c in current_centers], current_angles[:])
                    
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles = best_config
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    all_verts = []
    for i in range(N):
        v = hexagon_vertices(centers[i][0], centers[i][1], angles[i])
        all_verts.extend(v)
    S = get_required_side(all_verts)
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
