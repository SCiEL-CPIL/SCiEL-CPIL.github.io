"""No-isosceles set: construct() -> list of (x, y) grid points in [0, N)^2 with no three points forming an isosceles triangle."""
import time
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill

    def dist2(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def is_valid_addition(pts, new_pt):
        """Check if adding new_pt to pts creates an isosceles triangle."""
        n = len(pts)
        # Distances from new_pt to each existing point
        dists_to_new = [dist2(pts[i], new_pt) for i in range(n)]
        
        # Check if any two distances from new_pt are equal (vertex at new_pt)
        seen_d = set()
        for d in dists_to_new:
            if d in seen_d:
                return False
            seen_d.add(d)
        
        # Check for isosceles with vertex at existing points
        for i in range(n):
            for j in range(i + 1, n):
                dij = dist2(pts[i], pts[j])
                din = dists_to_new[i]
                djn = dists_to_new[j]
                if dij == din or dij == djn or din == djn:
                    return False
        return True

    def is_valid_set(pts):
        """Check if a set of points has no isosceles triangle."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d1 = dist2(pts[i], pts[j])
                    d2 = dist2(pts[i], pts[k])
                    d3 = dist2(pts[j], pts[k])
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        return False
        return True

    def greedy_construct():
        """Greedy construction with randomized order."""
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        pts = []
        for pt in all_points:
            if is_valid_addition(pts, pt):
                pts.append(pt)
        return pts

    def local_search(pts):
        """Try to add points to improve the set."""
        pts = list(pts)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            all_points = [(x, y) for x in range(N) for y in range(N)]
            random.shuffle(all_points)
            for pt in all_points:
                if pt not in pts:
                    if is_valid_addition(pts, pt):
                        pts.append(pt)
                        improved = True
                        break
        return pts

    # Start with greedy construction
    best_pts = greedy_construct()
    best_size = len(best_pts)
    
    # Multiple restarts with local search
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        pts = greedy_construct()
        pts = local_search(pts)
        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = pts
    
    # Final local search on the best set
    best_pts = local_search(best_pts)
    
    return [(int(x), int(y)) for x, y in best_pts]

# EVOLVE-BLOCK-END
