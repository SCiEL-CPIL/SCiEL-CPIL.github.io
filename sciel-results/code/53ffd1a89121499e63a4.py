import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    N = 32
    TARGET = 76  # Harborth's optimum
    
    def dist2(p, q):
        """Squared distance between lattice points p=(a,b) and q=(c,d)."""
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist2(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def valid_placement(points):
        """Check all pairs have distance >= 1 (squared distance >= 1)."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < 1:
                    return False
        return True
    
    def generate_hexagon(n):
        """Generate a compact hexagonal configuration of n points."""
        # Start with center and expand rings
        points = [(0, 0)]
        if n == 1:
            return points
        
        # Ring sizes: 6, 12, 18, 24, ...
        # Ring k has 6k points
        ring = 1
        while len(points) < n:
            # Add points on ring 'ring'
            # The ring consists of 6 sides, each of length 'ring'
            # Side 1: (ring, 0) to (0, ring) - direction (-1, 1)
            # Actually, let me use the standard hexagonal coordinates
            
            # For ring k, the 6 corners are at:
            # (k, 0), (0, k), (-k, k), (-k, 0), (0, -k), (k, -k)
            # Wait, let me think about this more carefully.
            
            # In axial coordinates for triangular lattice:
            # The six neighbors of (0,0) are: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
            # So the ring at distance k has points where max(|a|,|b|,|a-b|) = k... 
            # Actually for triangular lattice, distance from origin is sqrt(a^2 + b^2 + ab)
            # Ring k: points where a^2 + b^2 + ab = k^2
            
            # Let me just generate all points in a bounding box and filter
            # by distance, then take the closest ones
            
            # Simpler approach: generate a hexagonal shape
            # For a hexagon with "radius" r, it has 1 + 3*r*(r+1) points
            # r=1: 7, r=2: 19, r=3: 37
            # We want 32, so r=2 gives 19, need 13 more from r=3 ring
            
            if len(points) >= n:
                break
                
            # Generate points on ring 'ring'
            ring_points = []
            # The ring at graph-distance 'ring' from center
            # For triangular lattice, these are points at L1-like distance
            # Let me enumerate all points with |a|<=ring+1, |b|<=ring+1, |a-b|<=ring+1
            # and check if they're exactly at the boundary
            
            for a in range(-ring, ring+1):
                for b in range(-ring, ring+1):
                    d2 = a*a + b*b + a*b
                    if d2 == ring*ring:
                        # Check it's not already in points
                        if (a, b) not in points:
                            ring_points.append((a, b))
            
            # Sort ring points by angle for compactness
            import math
            def angle(p):
                return math.atan2(p[1], p[0])
            ring_points.sort(key=angle)
            
            # Add up to needed
            needed = n - len(points)
            for p in ring_points[:needed]:
                points.append(p)
            # If we still need more, continue to next ring
            if len(points) < n:
                ring += 1
        
        return points[:n]
    
    def generate_compact(n):
        """Generate a compact configuration using a different approach."""
        # Build up layer by layer, always adding the point that maximizes
        # connections to existing points
        points = [(0, 0)]
        
        while len(points) < n:
            # Try all candidate positions in a bounding box
            best_score = -1
            best_pos = None
            max_coord = 10
            
            for a in range(-max_coord, max_coord+1):
                for b in range(-max_coord, max_coord+1):
                    pos = (a, b)
                    if pos in points:
                        continue
                    # Check minimum distance constraint
                    valid = True
                    for p in points:
                        if dist2(pos, p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Count connections
                    conn = 0
                    for p in points:
                        if dist2(pos, p) == 1:
                            conn += 1
                    if conn > best_score:
                        best_score = conn
                        best_pos = pos
            
            if best_pos is None:
                # Fallback: add a point far away
                best_pos = (len(points), 0)
                for p in points:
                    if dist2(best_pos, p) < 1:
                        best_pos = (len(points) + 1, 0)
                points.append(best_pos)
            else:
                points.append(best_pos)
        
        return points
    
    def local_search(points, max_time=None):
        """Local search: try swapping points or moving them to improve edge count."""
        if max_time is None:
            max_time = time.time() + 5.0
        
        points = list(points)
        current_score = count_edges(points)
        best_points = list(points)
        best_score = current_score
        
        n = len(points)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < max_time and no_improve < max_no_improve:
            improved = False
            no_improve += 1
            
            # Try moving each point to a nearby position
            for i in range(n):
                if time.time() > max_time:
                    break
                orig = points[i]
                # Try all positions in a small neighborhood
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        if da == 0 and db == 0:
                            continue
                        new_pos = (orig[0] + da, orig[1] + db)
                        
                        # Check bounds
                        if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                            continue
                        
                        # Check it's not already occupied
                        if new_pos in points:
                            continue
                        
                        # Check minimum distance to all other points
                        valid = True
                        for j in range(n):
                            if j == i:
                                continue
                            if dist2(new_pos, points[j]) < 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Calculate new score
                        # Remove connections from old position
                        old_conn = 0
                        for j in range(n):
                            if j != i:
                                if dist2(orig, points[j]) == 1:
                                    old_conn += 1
                        
                        # Add connections from new position
                        new_conn = 0
                        for j in range(n):
                            if j != i:
                                if dist2(new_pos, points[j]) == 1:
                                    new_conn += 1
                        
                        delta = new_conn - old_conn
                        if delta > 0:
                            new_score = current_score + delta
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p if j != i else new_pos for j, p in enumerate(points)]
                                points[i] = new_pos
                                current_score = new_score
                                improved = True
                                no_improve = 0
                                break
                        elif delta == 0:
                            # Accept with some probability (simulated annealing-like)
                            if random.random() < 0.1:
                                new_score = current_score
                                points[i] = new_pos
                                improved = True
                                no_improve = 0
                                break
                if improved and time.time() > max_time:
                    break
            
            if not improved:
                # Try random restarts within the neighborhood
                for _ in range(10):
                    if time.time() > max_time:
                        break
                    i = random.randint(0, n-1)
                    orig = points[i]
                    for _ in range(20):
                        da = random.randint(-5, 5)
                        db = random.randint(-5, 5)
                        if da == 0 and db == 0:
                            continue
                        new_pos = (orig[0] + da, orig[1] + db)
                        if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                            continue
                        if new_pos in points:
                            continue
                        valid = True
                        for j in range(n):
                            if j == i:
                                continue
                            if dist2(new_pos, points[j]) < 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        old_conn = 0
                        for j in range(n):
                            if j != i:
                                if dist2(orig, points[j]) == 1:
                                    old_conn += 1
                        new_conn = 0
                        for j in range(n):
                            if j != i:
                                if dist2(new_pos, points[j]) == 1:
                                    new_conn += 1
                        delta = new_conn - old_conn
                        if delta > 0:
                            new_score = current_score + delta
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p if j != i else new_pos for j, p in enumerate(points)]
                                points[i] = new_pos
                                current_score = new_score
                                no_improve = 0
                            break
        
        return best_points, best_score
    
    # Generate initial configurations
    best_points = None
    best_score = -1
    
    # Try multiple initializations
    init_configs = []
    
    # 1. Compact growth
    init_configs.append(generate_compact(N))
    
    # 2. Hexagon-like
    init_configs.append(generate_hexagon(N))
    
    # 3. Random compact
    for _ in range(3):
        random.seed(random.randint(0, 1000000))
        pts = []
        attempts = 0
        while len(pts) < N and attempts < 10000:
            a = random.randint(-8, 8)
            b = random.randint(-8, 8)
            pos = (a, b)
            if pos in pts:
                attempts += 1
                continue
            valid = True
            for p in pts:
                if dist2(pos, p) < 1:
                    valid = False
                    break
            if valid:
                pts.append(pos)
            attempts += 1
        if len(pts) == N:
            init_configs.append(pts)
    
    # Run local search on each
    for config in init_configs:
        if time.time() > deadline - 30:
            break
        result_points, result_score = local_search(config, max_time=min(time.time() + 20, deadline - 30))
        if result_score > best_score:
            best_score = result_score
            best_points = result_points
        
        # Early exit if we hit the optimum
        if best_score >= TARGET:
            break
    
    # Final polish: try to improve further
    if best_points is not None and best_score < TARGET and time.time() < deadline - 10:
        result_points, result_score = local_search(best_points, max_time=min(time.time() + 20, deadline - 10))
        if result_score > best_score:
            best_score = result_score
            best_points = result_points
    
    # Convert to list of lists
    if best_points is None:
        # Fallback
        return [[a, b] for a in range(16) for b in range(2)]
    
    return [[p[0], p[1]] for p in best_points]
