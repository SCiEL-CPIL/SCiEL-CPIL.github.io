"""Combined program: two shifted parabolas with local search to minimize
largest 3-collinear-free subset for 20 points."""
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s limit

    def check_no_four_collinear(pts):
        """Check that no 4 points are collinear. O(n^4) but n=20 is fine."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        d1 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                        d2 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])
                        if d1 == 0 and d2 == 0:
                            return False
        return True

    def find_collinear_triples(pts):
        """Find all collinear triples. Returns list of (i, j, k) tuples."""
        n = len(pts)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                    if d == 0:
                        triples.append((i, j, k))
        return triples

    def largest_3_free_subset(pts):
        """Find the largest subset with no three collinear using branch and bound."""
        n = len(pts)
        triples = find_collinear_triples(pts)
        
        # For each triple, mark which points are involved
        # We want to find the maximum subset with no complete triple
        
        # Use a greedy + local search approach for speed, then verify
        # Actually, let's use a proper branch and bound for n=20
        
        # Precompute: for each point, which triples it's in
        point_triples = defaultdict(list)
        for idx, (i, j, k) in enumerate(triples):
            point_triples[i].append(idx)
            point_triples[j].append(idx)
            point_triples[k].append(idx)
        
        # Branch and bound
        best = [0]
        best_set = [set()]
        
        def branch(selected, remaining, current_size):
            if time.time() - start_time > DEADLINE:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                    best_set[0] = set(selected)
                return
            
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best[0]:
                return
            
            # Try to add points from remaining
            for idx, p in enumerate(remaining):
                new_remaining = remaining[idx+1:]
                # Check if adding p creates a collinear triple with selected
                valid = True
                for t in point_triples[p]:
                    i, j, k = triples[t]
                    if i in selected and j in selected:
                        valid = False
                        break
                    if i in selected and k in selected:
                        valid = False
                        break
                    if j in selected and k in selected:
                        valid = False
                        break
                
                if valid:
                    selected.add(p)
                    branch(selected, new_remaining, current_size + 1)
                    selected.discard(p)
                
                # Also try skipping p
                # (implicit in the loop continuing)
        
        # Start branch and bound
        all_points = list(range(n))
        branch(set(), all_points, 0)
        
        return best[0]

    def local_search(pts, max_iter=500):
        """Local search: try moving individual points to reduce the score."""
        n = len(pts)
        current_score = largest_3_free_subset(pts)
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            for i in range(n):
                # Try moving point i to nearby positions
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = pts[i][0] + dx
                        new_y = pts[i][1] + dy
                        if abs(new_x) > 100 or abs(new_y) > 100:
                            continue
                        
                        # Check distinctness
                        new_pts = [list(p) for p in pts]
                        new_pts[i] = [new_x, new_y]
                        if len(set(map(tuple, new_pts))) != n:
                            continue
                        
                        # Check no 4 collinear
                        if not check_no_four_collinear(new_pts):
                            continue
                        
                        # Compute new score (expensive, so limit this)
                        new_score = largest_3_free_subset(new_pts)
                        if new_score < current_score:
                            pts = new_pts
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return pts, current_score

    # Initial configuration: two shifted parabolas
    # Points: (i, i^2) and (i, i^2 + 1) for i in range(10)
    points = []
    for i in range(10):
        points.append([i, i * i])
        points.append([i, i * i + 1])
    
    # Verify initial configuration
    assert len(points) == 20
    assert len(set(map(tuple, points))) == 20
    for p in points:
        assert abs(p[0]) <= 100 and abs(p[1]) <= 100
    assert check_no_four_collinear(points), "Initial config has 4 collinear"
    
    # Compute initial score
    initial_score = largest_3_free_subset(points)
    
    # Try local search to improve
    if time.time() - start_time < DEADLINE:
        points, current_score = local_search(points, max_iter=200)
    
    # If local search didn't help or we're out of time, try a few alternative initial configs
    if time.time() - start_time < DEADLINE:
        # Try different offsets for the second parabola
        for offset in [2, 3, 5, 7]:
            if time.time() - start_time > DEADLINE:
                break
            alt_points = []
            for i in range(10):
                alt_points.append([i, i * i])
                alt_points.append([i, i * i + offset])
            
            if len(set(map(tuple, alt_points))) != 20:
                continue
            if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in alt_points):
                continue
            if not check_no_four_collinear(alt_points):
                continue
            
            alt_score = largest_3_free_subset(alt_points)
            if alt_score < current_score:
                points = alt_points
                current_score = alt_score
        
        # Try different x-ranges
        for x_start in range(-5, 6):
            if time.time() - start_time > DEADLINE:
                break
            for offset in [1, 2, 3]:
                if time.time() - start_time > DEADLINE:
                    break
                alt_points = []
                for i in range(10):
                    x = x_start + i
                    alt_points.append([x, x * x])
                    alt_points.append([x, x * x + offset])
                
                if len(set(map(tuple, alt_points))) != 20:
                    continue
                if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in alt_points):
                    continue
                if not check_no_four_collinear(alt_points):
                    continue
                
                alt_score = largest_3_free_subset(alt_points)
                if alt_score < current_score:
                    points = alt_points
                    current_score = alt_score
    
    # Final verification
    assert len(points) == 20
    assert len(set(map(tuple, points))) == 20
    for p in points:
        assert abs(p[0]) <= 100 and abs(p[1]) <= 100
    assert check_no_four_collinear(points), "Final config has 4 collinear"
    
    return points
