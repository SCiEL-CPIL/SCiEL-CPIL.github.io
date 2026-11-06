import time
import math

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of unit-distance edges.
    Uses a hexagonal spiral (BFS from origin) which is known to achieve Harborth's bound.
    For n=32, the optimum is 76 edges.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n_points = 32
    
    # Neighbor offsets for the triangular lattice in axial coordinates (a, b)
    offsets = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    
    # BFS to generate points in order of increasing hex distance from origin
    visited = set()
    visited.add((0, 0))
    points = [(0, 0)]
    queue = [(0, 0)]
    
    while len(points) < n_points:
        next_queue = []
        for a, b in queue:
            for da, db in offsets:
                na, nb = a + da, b + db
                if (na, nb) not in visited:
                    visited.add((na, nb))
                    points.append((na, nb))
                    next_queue.append((na, nb))
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
        queue = next_queue
        if time.time() > deadline:
            break
    
    # Verify we have exactly 32 distinct points
    if len(points) != n_points:
        points = [(a, b) for a in range(6) for b in range(6)][:n_points]
    
    # Count unit-distance edges
    def count_edges(pts):
        count = 0
        n = len(pts)
        for i in range(n):
            a1, b1 = pts[i]
            for j in range(i + 1, n):
                a2, b2 = pts[j]
                da = a1 - a2
                db = b1 - b2
                dist_sq = da * da + da * db + db * db
                if dist_sq == 1:
                    count += 1
        return count
    
    edges = count_edges(points)
    
    # Local search to improve: try replacing each point with a nearby unused lattice point
    # to increase edge count
    if edges < 76 and time.time() < deadline - 10:
        used = set(points)
        # Generate candidate replacement points near the cluster
        max_coord = max(max(abs(a), abs(b)) for a, b in points)
        candidates = []
        for a in range(-max_coord - 2, max_coord + 3):
            for b in range(-max_coord - 2, max_coord + 3):
                if (a, b) not in used and abs(a) <= 64 and abs(b) <= 64:
                    candidates.append((a, b))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n_points):
                if time.time() > deadline - 5:
                    break
                old_pt = points[i]
                for new_pt in candidates:
                    # Try replacing points[i] with new_pt
                    test_pts = points[:]
                    test_pts[i] = new_pt
                    new_edges = count_edges(test_pts)
                    if new_edges > edges:
                        points = test_pts
                        edges = new_edges
                        used.discard(old_pt)
                        used.add(new_pt)
                        candidates = [c for c in candidates if c != old_pt and c != new_pt]
                        # Remove new_pt from candidates, add old_pt
                        if new_pt in candidates:
                            candidates.remove(new_pt)
                        candidates.append(old_pt)
                        improved = True
                        break
    
    # Convert to list of lists as required
    result = [[a, b] for a, b in points]
    
    return result
