import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 20s safety margin before 160s limit
    
    N = 32
    MAX_COORD = 64
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def is_valid(points):
        """Check all pairs have distance >= 1 (i.e., dist_sq >= 1)."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True

    def hex_cluster(r):
        """Generate hex cluster of radius r centered at origin."""
        pts = set()
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if max(abs(a), abs(b), abs(a + b)) <= r:
                    pts.add((a, b))
        return list(pts)

    # Generate initial hex cluster of radius 3 (37 points)
    full_hex = hex_cluster(3)
    if len(full_hex) != 37:
        # Fallback if hex generation is flawed
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Greedy removal of 5 points with lowest degree in the current set
    current_points = full_hex[:]
    
    for _ in range(5):
        if len(current_points) <= N:
            break
            
        # Calculate degrees of all points in current_points
        degrees = {p: 0 for p in current_points}
        for i, p in enumerate(current_points):
            for j, q in enumerate(current_points):
                if i == j:
                    continue
                if dist_sq(p, q) == 1:
                    degrees[p] += 1
        
        # Find the point with minimum degree
        min_deg_point = min(current_points, key=lambda x: degrees[x])
        
        # Remove it
        current_points.remove(min_deg_point)
        
    # We should have 32 points now
    if len(current_points) != N:
        # If something went wrong, fallback to random removal
        random.seed(42)
        idxs = list(range(37))
        random.shuffle(idxs)
        remove_set = set(idxs[:5])
        current_points = [full_hex[i] for i in range(37) if i not in remove_set]
        
    # Verify validity
    if not is_valid(current_points):
        # Should not happen for distinct lattice points, but just in case
        return [[a, b] for a in range(16) for b in range(2)]
        
    # Count edges
    edges = count_edges(current_points)
    
    # If we didn't reach the theoretical maximum of 76, try local search to improve
    if edges < 76:
        points = current_points[:]
        best_points = points[:]
        best_edges = edges
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                orig = points[i]
                # Try moving point i to a nearby position
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0:
                            continue
                        cand = (orig[0] + da, orig[1] + db)
                        if abs(cand[0]) > MAX_COORD or abs(cand[1]) > MAX_COORD:
                            continue
                        
                        # Check if cand is already occupied
                        if cand in points:
                            continue
                            
                        # Check validity: new position must be >=1 from all other points
                        valid = True
                        for j in range(N):
                            if j == i:
                                continue
                            if dist_sq(cand, points[j]) < 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute new edge count
                        old_point = points[i]
                        points[i] = cand
                        new_edges = count_edges(points)
                        
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_points = points[:]
                            improved = True
                            if best_edges >= 76:
                                break
                        else:
                            points[i] = old_point
                    if best_edges >= 76:
                        break
                if best_edges >= 76:
                    break
            
        if best_edges > edges:
            current_points = best_points
            edges = best_edges
            
    return [[p[0], p[1]] for p in current_points]
# EVOLVE-BLOCK-END
