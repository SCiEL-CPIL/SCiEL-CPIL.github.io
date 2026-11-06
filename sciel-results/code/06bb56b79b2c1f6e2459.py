import time
import random

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _neighbors(p):
    a, b = p
    return [
        (a + 1, b), (a - 1, b),
        (a, b + 1), (a, b - 1),
        (a + 1, b - 1), (a - 1, b + 1)
    ]

def _edge_count(points):
    pset = set(points)
    cnt = 0
    for p in points:
        for n in _neighbors(p):
            if n in pset and p < n:
                cnt += 1
    return cnt

def _is_valid(points):
    if len(points) != 32:
        return False
    pset = set(points)
    if len(pset) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if _dist_sq(points[i], points[j]) < 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Initial heuristic: Compact cluster
    # Generate points in a hexagonal spiral until we have 32
    pts = [(0,0)]
    # Simple spiral generation
    # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # This is a bit complex to code correctly on the fly.
    # Let's use a grid fill.
    # Fill a 8x8 triangle?
    # Or just random restarts.
    
    best_score = -1
    best_points = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization in a box
        # Or better: Start with a known good shape and perturb
        # Let's start with a 16x2 strip (60 edges) and try to improve
        # Or start with a 6x6 square (36 pts, remove 4)
        
        # Heuristic: 6x6 square is 36 pts.
        # Remove 4 corner points?
        # 6x6 square: a in 0..5, b in 0..5.
        # 36 points.
        # Remove (0,0), (5,0), (0,5), (5,5).
        # 32 points left.
        pts = []
        for a in range(6):
            for b in range(6):
                if (a==0 and b==0) or (a==5 and b==0) or (a==0 and b==5) or (a==5 and b==5):
                    continue
                pts.append((a,b))
                
        if not _is_valid(pts):
            continue
            
        current_score = _edge_count(pts)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for i in range(32):
                if time.time() > deadline:
                    break
                p = pts[i]
                # Try all neighbors
                for n in _neighbors(p):
                    if abs(n[0]) > 64 or abs(n[1]) > 64:
                        continue
                    # Check if n is occupied
                    if n in pts and pts.index(n) != i:
                        continue
                    # Check distance to others
                    valid = True
                    for j in range(32):
                        if i == j:
                            continue
                        if _dist_sq(n, pts[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_pts = pts[:]
                    new_pts[i] = n
                    new_score = _edge_count(new_pts)
                    if new_score > current_score:
                        pts = new_pts
                        current_score = new_score
                        improved = True
                        break # Restart loop? Or continue?
                    elif new_score == current_score:
                        # Random move to escape local optima?
                        if random.random() < 0.1:
                            pts = new_pts
                            improved = True
                            break

        if current_score > best_score:
            best_score = current_score
            best_points = pts[:]
            
        if best_score == 76:
            break

    if best_points is None:
        # Fallback to strip
        best_points = [[a, b] for a in range(16) for b in range(2)]
        
    return [[p[0], p[1]] for p in best_points]
