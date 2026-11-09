import random
import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        # Map line to list of points
        line_map = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # Line defined by dx, dy normalized
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Canonical form: dx > 0 or (dx == 0 and dy > 0)
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
                
                # Constant c: ax + by = c
                # Normalized a=dy, b=-dx
                a = dy
                b = -dx
                c = a * x1 + b * y1
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(i)
                
        for pts in line_map.values():
            if len(pts) >= 4:
                return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        line_map = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
                
                a = dy
                b = -dx
                c = a * x1 + b * y1
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(i)
                
        for pts in line_map.values():
            if len(pts) >= 3:
                for t in itertools.combinations(pts, 3):
                    triples.add(tuple(sorted(t)))
        return list(triples)

    def max_no_three_collinear(points, triples):
        n = len(points)
        # adjacency: for each point, which other points form a triple with it?
        # Actually, we need to check if a set S is independent in the hypergraph of triples.
        # A set S is independent if no triple is fully contained in S.
        
        # Branch and bound
        # Order points by degree (number of triples they are part of) to prune faster
        degrees = [0] * n
        for t in triples:
            for i in t:
                degrees[i] += 1
        
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        best_score = 0
        current_set = set()
        
        # Precompute for each point, the triples it is involved in
        point_triples = [[] for _ in range(n)]
        for t in triples:
            for i in t:
                point_triples[i].append(t)
                
        def is_valid_addition(i, current_set):
            # Check if adding i to current_set creates a triple
            # A triple is created if there exists (j, k) in current_set such that (i, j, k) is a triple
            for t in point_triples[i]:
                # t is (i, j, k) or some permutation
                # Check if the other two are in current_set
                count_in_set = 0
                for idx in t:
                    if idx == i:
                        continue
                    if idx in current_set:
                        count_in_set += 1
                if count_in_set == 2:
                    return False
            return True

        def backtrack(idx, current_size):
            nonlocal best_score
            if idx == n:
                if current_size > best_score:
                    best_score = current_size
                return
                
            # Pruning: even if we take all remaining, can we beat best_score?
            if current_size + (n - idx) <= best_score:
                return
                
            i = order[idx]
            
            # Option 1: Exclude i
            backtrack(idx + 1, current_size)
            
            # Option 2: Include i
            if is_valid_addition(i, current_set):
                current_set.add(i)
                backtrack(idx + 1, current_size + 1)
                current_set.remove(i)
                
        backtrack(0, 0)
        return best_score

    # Initial random valid configuration
    best_points = None
    best_score = 21
    
    # Generate initial points
    while True:
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        if not has_four_collinear(pts):
            best_points = pts
            break
            
    # Calculate initial score
    triples = get_collinear_triples(best_points)
    best_score = max_no_three_collinear(best_points, triples)
    
    # Local Search
    current_points = best_points
    current_triples = triples
    current_score = best_score
    
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try a few random new positions
        improved = False
        for _ in range(10):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Check if point already exists
            if (new_x, new_y) in [(p[0], p[1]) for p in current_points if current_points.index((p[0], p[1])) != idx]:
                continue
                
            new_points = current_points.copy()
            new_points[idx] = (new_x, new_y)
            
            if has_four_collinear(new_points):
                continue
                
            new_triples = get_collinear_triples(new_points)
            new_score = max_no_three_collinear(new_points, new_triples)
            
            if new_score < current_score:
                current_points = new_points
                current_triples = new_triples
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points
                improved = True
                break
            elif new_score == current_score:
                # Accept with some probability to explore
                if random.random() < 0.1:
                    current_points = new_points
                    current_triples = new_triples
                    break
                    
        if not improved:
            temperature *= cooling
            
    return [[p[0], p[1]] for p in best_points]
