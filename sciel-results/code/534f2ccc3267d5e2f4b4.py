import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for 120s limit (actually 150s limit, but safe)

    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def calc_score(points):
        """
        Estimate the largest subset with no three collinear.
        This is equivalent to 20 - (min vertex cover size of the 3-uniform hypergraph of collinear triples).
        We approximate the min vertex cover size using a greedy algorithm:
        Repeatedly remove the vertex involved in the most hyperedges (triples).
        """
        n = len(points)
        
        # Build the hypergraph of collinear triples
        # A triple (i, j, k) is an edge if points i, j, k are collinear.
        # To do this efficiently, we can find all lines with >= 3 points.
        # For a line with k points, it contains C(k, 3) triples.
        
        # First, find all lines and the points on them
        line_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].append((i, j))
        
        # We need to identify sets of collinear points.
        # A simpler way for small n:
        # For each point i, and each other point j, the line is determined.
        # Let's just count triples directly but efficiently.
        
        # Actually, the greedy vertex cover on the hypergraph is what we want.
        # Let's build the hyperedges.
        hyperedges = []
        # To avoid O(n^4), we can use the line_points map.
        # But line_points currently has pairs. We need to group pairs by line.
        # The key is the line. The value is a list of pairs.
        # If a line has k points, the number of pairs is C(k, 2).
        # We can reconstruct the set of points on each line.
        
        # Let's just use a simpler heuristic for speed:
        # Count for each point how many other points are collinear with it.
        # This is O(n^3) which is 8000 ops, very fast.
        
        # Actually, the greedy vertex cover on the 3-uniform hypergraph is better.
        # Let's build the list of all collinear triples.
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                li = get_line(points[i], points[j])
                for k in range(j + 1, n):
                    if get_line(points[i], points[k]) == li:
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Greedy Vertex Cover Approximation
        # We want to remove the minimum number of vertices to cover all triples.
        # Greedy: pick the vertex in the most triples, remove it and all triples containing it.
        remaining_vertices = set(range(n))
        remaining_triples = [set(t) for t in triples]
        
        # Precompute incidence: vertex -> list of triple indices
        # Or just iterate.
        # Since n=20, and triples <= C(20,3)=1140, this is fast.
        
        removed = 0
        while remaining_triples:
            # Count occurrences of each vertex in remaining_triples
            counts = defaultdict(int)
            for t in remaining_triples:
                for v in t:
                    if v in remaining_vertices:
                        counts[v] += 1
            
            if not counts:
                break
                
            # Pick vertex with max count
            max_count = max(counts.values())
            # Pick one arbitrarily (or smallest index for determinism)
            candidates = [v for v, c in counts.items() if c == max_count]
            v_to_remove = min(candidates)
            
            remaining_vertices.remove(v_to_remove)
            removed += 1
            
            # Remove all triples containing v_to_remove
            new_remaining = []
            for t in remaining_triples:
                if v_to_remove not in t:
                    new_remaining.append(t)
            remaining_triples = new_remaining
            
        # The size of the largest subset with no 3 collinear is n - removed
        return n - removed

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Initial configuration: Random
    current_points = None
    for _ in range(1000):
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        if is_valid(pts):
            current_points = pts
            break
    
    if current_points is None:
        # Fallback: Parabola (no 3 collinear? No, any 3 points on a parabola are not collinear? 
        # Wait, a line intersects a parabola at at most 2 points. So no 3 collinear.
        # But we want to MINIMIZE the score. Score 20 is bad. We want lower score.
        # So we want MORE collinear triples.
        # Parabola has score 20. We need to find a config with lower score.
        current_points = [[x, x * x] for x in range(-10, 10)]
        
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        
        # Generate a candidate move
        # Strategy 1: Place on a line defined by two other points (to create a triple)
        # Strategy 2: Random move
        
        candidate = None
        
        # Try to create a triple
        # Pick two random other points i, j
        others = [k for k in range(20) if k != idx]
        random.shuffle(others)
        i, j = others[0], others[1]
        
        p1 = current_points[i]
        p2 = current_points[j]
        
        # Line through p1, p2
        # We want to place point idx on this line.
        # The line is A*x + B*y + C = 0.
        # We can parametrize the line or just try integer points on the line segment/ray.
        
        # Let's try to find an integer point on the line that is not p1, p2, and not any other point.
        # And check if it creates a 4-collinear violation.
        
        # To be efficient, let's just try a few points on the line.
        # The line passes through p1 and p2.
        # Vector v = p2 - p1.
        # Points: p1 + t * v for integer t.
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Try t in a small range
        found_candidate = False
        for t in range(-20, 21):
            if t == 0 or t == 1:
                continue
            nx = p1[0] + t * dx
            ny = p1[1] + t * dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                new_candidate = [nx, ny]
                if new_candidate not in current_points:
                    test_points = current_points[:]
                    test_points[idx] = new_candidate
                    if is_valid(test_points):
                        candidate = test_points
                        found_candidate = True
                        break
        
        if not found_candidate:
            # Try random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    candidate = test_points
        
        if candidate is None:
            T *= cooling_rate
            continue
            
        candidate_score = calc_score(candidate)
        
        # Acceptance criterion
        if candidate_score < current_score:
            current_points = candidate
            current_score = candidate_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Delta E = candidate_score - current_score (positive means worse)
            delta_e = candidate_score - current_score
            # Accept with probability exp(-delta_e / T)
            if random.random() < pow(2.718281828, -delta_e / T):
                current_points = candidate
                current_score = candidate_score
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor temperature

    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
