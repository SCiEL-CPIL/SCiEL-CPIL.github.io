import time
import random
from itertools import combinations

def _collinear(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _count_collinear(points):
    """Counts total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j:
                    if _collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return False
    return True

def _precompute_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_indep_bnb(points, triples, time_limit=2.0):
    """
    Exact branch and bound for maximum independent set in the hypergraph
    where vertices are points and hyperedges are collinear triples.
    """
    start_time = time.time()
    n = len(points)
    best_size = 0
    
    # Precompute adjacency lists for triples
    # point_triples[i] = list of triple indices involving point i
    point_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        point_triples[i].append(idx)
        point_triples[j].append(idx)
        point_triples[k].append(idx)
        
    # Degree of each point (number of triples it participates in)
    degree = [len(pt) for pt in point_triples]
    
    # Order points by degree (high degree first) to prune earlier
    order = sorted(range(n), key=lambda x: -degree[x])
    
    # Map from ordered index to original index
    orig_idx = [0] * n
    for new_idx, old_idx in enumerate(order):
        orig_idx[new_idx] = old_idx
        
    # Rebuild point_triples in the new order? No, just use original indices in logic.
    # We iterate through 'order'.
    
    def bnb(idx, selected_mask, cur_size):
        nonlocal best_size
        if time.time() - start_time > time_limit:
            return
            
        # Upper bound: current size + remaining points
        if cur_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if cur_size > best_size:
                best_size = cur_size
            return
            
        p_orig = orig_idx[idx]
        
        # Option 1: Include p_orig
        new_mask = selected_mask | (1 << p_orig)
        valid = True
        for t_idx in point_triples[p_orig]:
            i, j, k = triples[t_idx]
            # Check if all three are in new_mask
            if (new_mask & (1 << i)) and (new_mask & (1 << j)) and (new_mask & (1 << k)):
                valid = False
                break
                
        if valid:
            bnb(idx + 1, new_mask, cur_size + 1)
            
        # Option 2: Exclude p_orig
        bnb(idx + 1, selected_mask, cur_size)

    bnb(0, 0, 0)
    return best_size

def _score(points, time_limit=2.0):
    triples = _precompute_triples(points)
    return _max_indep_bnb(points, triples, time_limit)

def _initial_config():
    """
    Generate points near a parabola y = x^2.
    Points on a parabola have the property that any line intersects it in at most 2 points.
    By perturbing them, we create collinear triples.
    We want to minimize the max independent set.
    """
    random.seed(42)
    # Choose x coordinates spread out to avoid immediate large collinear sets
    # but close enough to allow collinearities with perturbation.
    # Using a grid of x values.
    x_coords = []
    for i in range(20):
        # Spread x from -10 to 10
        x_coords.append(i - 10)
        
    points = []
    for x in x_coords:
        y = x * x
        # Add small random perturbation
        px = x + random.randint(-2, 2)
        py = y + random.randint(-2, 2)
        points.append([px, py])
        
    # Ensure distinct points
    seen = set()
    unique_points = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_points.append(p)
            
    # If we lost points due to collision, add more
    while len(unique_points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_points.append([x, y])
            
    # Ensure no 4 collinear
    if not _no_four_collinear(unique_points):
        # Fallback: simple grid with perturbation
        pts = []
        for i in range(5):
            for j in range(4):
                x = i * 20 + random.randint(-5, 5)
                y = j * 20 + random.randint(-5, 5)
                pts.append([x, y])
        if _no_four_collinear(pts):
            return pts
        # Last resort
        return [[i, i*i] for i in range(-10, 10)]
        
    return unique_points

def construct():
    deadline = time.time() + 110
    random.seed(123)
    
    points = _initial_config()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    # Local Search
    # Strategy: Pick a point involved in many triples, move it to a random location
    # that breaks those triples without creating too many new ones or 4-collinearities.
    
    while time.time() < deadline - 10:
        # Calculate current triples to guide moves
        triples = _precompute_triples(best_points)
        
        # Count how many triples each point is in
        point_triple_count = [0] * 20
        for i, j, k in triples:
            point_triple_count[i] += 1
            point_triple_count[j] += 1
            point_triple_count[k] += 1
            
        # Select a point with high triple count to move
        # If no triples, score is 20, which is likely not minimal, but we can't break what doesn't exist.
        # However, we want to *create* a structure that minimizes the max independent set.
        # Actually, the max independent set is the set of points with NO 3 collinear.
        # If we have many collinear triples, the independent set must exclude points from those triples.
        # Wait. The "score" is the largest subset with NO 3 collinear.
        # To MINIMIZE this score, we want to MAXIMIZE the number of collinear triples.
        # We want to force the "no 3 collinear" subset to be as small as possible.
        # This means we want to cover as many points as possible with collinear triples.
        
        # So, the objective is to maximize the "coverage" of collinear triples.
        # A point is "covered" if it is part of at least one collinear triple.
        # The max independent set is the set of points NOT in any collinear triple? No.
        # The max independent set is the largest set of points such that NO three are collinear.
        # This is equivalent to the maximum independent set in the hypergraph of collinear triples.
        # To minimize this, we want the hypergraph to be "dense" in a way that forces independence to be small.
        
        # Let's pick a random point to move, but prioritize those in many triples?
        # Or maybe just random?
        # Let's try moving a point that is in many triples to a place that creates NEW triples,
        # effectively "locking" it into the dependent set.
        
        idx = random.randint(0, 19)
        
        # Try a few random moves for this point
        for _ in range(5):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if [new_x, new_y] in best_points:
                continue
                
            candidate = [p[:] for p in best_points]
            candidate[idx] = [new_x, new_y]
            
            if not _no_four_collinear(candidate):
                continue
                
            new_score = _score(candidate, time_limit=1.5)
            if new_score < best_score:
                best_score = new_score
                best_points = candidate
                break # Move to next point
                
    return best_points
