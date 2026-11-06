"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    random.seed(42)
    
    def collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])
    
    def no_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if collinear(points[i], points[j], points[k]) and collinear(points[i], points[j], points[l]):
                            return False
        return True
    
    def get_triples(points):
        """Return list of (i,j,k) where points i,j,k are collinear."""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i], points[j], points[k]):
                        triples.append((i,j,k))
        return triples
    
    def greedy_independent_set(points):
        """Greedy upper bound on max independent set size.
        Repeatedly pick a point, remove all triples containing it.
        The number of picks is an upper bound on the true max independent set."""
        n = len(points)
        triples = get_triples(points)
        # For each point, count how many triples it's in
        point_triple_count = [0]*n
        for (i,j,k) in triples:
            point_triple_count[i] += 1
            point_triple_count[j] += 1
            point_triple_count[k] += 1
        
        remaining = set(range(n))
        remaining_triples = set(triples)
        count = 0
        while remaining:
            # Pick the point in the most remaining triples (to remove most)
            best_point = None
            best_count = -1
            for p in remaining:
                cnt = sum(1 for t in remaining_triples if p in t)
                if cnt > best_count:
                    best_count = cnt
                    best_point = p
            if best_point is None:
                best_point = next(iter(remaining))
            count += 1
            remaining.discard(best_point)
            # Remove all triples containing best_point
            remaining_triples = set(t for t in remaining_triples if best_point not in t)
        return count
    
    def exact_max_independent_set(points):
        """Exact branch and bound for max independent set."""
        n = len(points)
        triples = get_triples(points)
        # Build adjacency: point i conflicts with j if they share a triple with some k
        # Actually, independent set means no three points are collinear
        # So we need the largest subset with no collinear triple
        
        # Use branch and bound
        best = [0]
        
        def bnb(remaining, current_size):
            if time.time() > deadline:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best[0]:
                return
            
            # Pick a point
            p = next(iter(remaining))
            remaining_list = list(remaining)
            
            # Option 1: include p
            # Find all points that form a triple with p
            conflict = set()
            for t in triples:
                if p in t:
                    conflict.update(t)
            conflict.discard(p)
            new_remaining = remaining - conflict
            bnb(new_remaining, current_size + 1)
            
            # Option 2: exclude p
            new_remaining2 = remaining - {p}
            bnb(new_remaining2, current_size)
        
        bnb(set(range(n)), 0)
        return best[0]
    
    # Greedy construction: build 20 points maximizing collinear triples
    # Start with a base set and greedily add points
    
    def initial_points():
        # Start with points on a few lines to create many triples
        # Use a grid-like structure but perturbed to avoid 4-collinear
        pts = []
        # Place points on several lines
        lines = [
            [(0,0), (1,0), (2,0), (3,0)],  # y=0
            [(0,1), (1,1), (2,1)],          # y=1
            [(0,2), (1,2), (2,2), (3,2)],   # y=2
            [(4,0), (4,1), (4,2)],          # x=4
            [(5,0), (5,1)],                 # x=5
            [(0,3), (1,3)],                 # y=3
            [(2,3), (3,3)],                 # y=3 continued
            [(4,3), (5,3)],                # y=3 continued
        ]
        for line in lines:
            for p in line:
                if p not in pts:
                    pts.append(p)
        # Need exactly 20
        while len(pts) < 20:
            # Add a random point
            p = (random.randint(-10, 10), random.randint(-10, 10))
            if p not in pts:
                pts.append(p)
        return pts[:20]
    
    def count_triples(points):
        return len(get_triples(points))
    
    # Start with a good initial configuration
    # Try several initial configurations and pick the best
    
    best_points = None
    best_score = 21  # higher is worse
    
    # Strategy 1: Greedy construction maximizing triples
    def greedy_construct():
        pts = []
        # Seed with a few points on a line
        base = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
        for p in base:
            if p not in pts:
                pts.append(p)
        
        # Greedily add points that create the most new triples
        candidates = set()
        for x in range(-10, 11):
            for y in range(-10, 11):
                candidates.add((x,y))
        candidates -= set(pts)
        
        while len(pts) < 20 and candidates:
            best_p = None
            best_gain = -1
            # Sample candidates for speed
            if len(candidates) > 200:
                sample = random.sample(list(candidates), 200)
            else:
                sample = list(candidates)
            
            for p in sample:
                test_pts = pts + [p]
                if not no_four_collinear(test_pts):
                    continue
                old_triples = count_triples(pts)
                new_triples = count_triples(test_pts)
                gain = new_triples - old_triples
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            
            if best_p is None:
                # Pick any valid point
                for p in sample:
                    test_pts = pts + [p]
                    if no_four_collinear(test_pts):
                        best_p = p
                        break
                if best_p is None:
                    break
            
            pts.append(best_p)
            candidates.discard(best_p)
        
        return pts
    
    # Try multiple random restarts
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        pts = greedy_construct()
        if len(pts) < 20:
            # Fill with random valid points
            for x in range(-100, 101):
                for y in range(-100, 101):
                    if len(pts) >= 20:
                        break
                    p = (x, y)
                    if p not in pts:
                        test_pts = pts + [p]
                        if no_four_collinear(test_pts):
                            pts.append(p)
                if len(pts) >= 20:
                    break
        
        if len(pts) == 20 and no_four_collinear(pts):
            # Evaluate with greedy bound (fast)
            greedy_score = greedy_independent_set(pts)
            if greedy_score < best_score:
                best_score = greedy_score
                best_points = pts[:]
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x*x] for x in range(-10, 10)]
    
    # Local search: try moving individual points
    current_points = best_points[:]
    current_score = greedy_independent_set(current_points)
    
    def try_local_search(points, max_time):
        pts = points[:]
        score = greedy_independent_set(pts)
        t0 = time.time()
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(20):
                if time.time() > max_time:
                    break
                old_p = pts[i]
                # Try moving point i to a random new location
                for _ in range(50):
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p == old_p or new_p in pts:
                        continue
                    test_pts = pts[:]
                    test_pts[i] = new_p
                    if not no_four_collinear(test_pts):
                        continue
                    new_score = greedy_independent_set(test_pts)
                    if new_score < score:
                        pts = test_pts
                        score = new_score
                        improved = True
                        break
        return pts, score
    
    current_points, current_score = try_local_search(current_points, deadline - 15)
    
    # Final check
    if len(current_points) == 20 and no_four_collinear(current_points):
        return [[p[0], p[1]] for p in current_points]
    else:
        return [[x, x*x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
