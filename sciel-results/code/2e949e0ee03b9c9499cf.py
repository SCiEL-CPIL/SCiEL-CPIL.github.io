import time
import random
from math import gcd
from itertools import combinations

_DEADLINE = 115.0

def _elapsed():
    return time.time() - _start_time

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_triple_conflicts(points):
    """Build list of collinear triples. Each triple is a frozenset of 3 indices."""
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_points = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_points.append(k)
            m = len(line_points)
            if m >= 3:
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = frozenset([line_points[a], line_points[b], line_points[c]])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _greedy_score(points):
    """
    Fast greedy heuristic to estimate the largest subset with no three collinear.
    This is an upper bound on the true score (greedy might not find the max, but it's fast).
    Actually, to minimize the score, we want this value to be low.
    We use a greedy set cover approach: repeatedly pick the point that is in the most triples?
    No, we want the largest subset with NO triples.
    Greedy: Start with all points. While there is a collinear triple, remove one point from it.
    To minimize the final subset size, we should remove the point that breaks the most other triples?
    Or simply: repeatedly pick a random triple and remove a random point from it?
    A better greedy for *finding* a large 3-free set:
    Sort points by degree (number of collinear triples they participate in).
    Greedily add points to the independent set if they don't form a triple with existing points.
    This gives an estimate.
    """
    n = len(points)
    if n == 0: return 0
    
    # Build adjacency/conflict info: for each pair, does it form a triple with any other?
    # To make it fast, we precompute triples once.
    triples = _build_triple_conflicts(points)
    
    # Map point index to list of triples containing it
    point_to_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for p in t:
            point_to_triples[p].append(idx)
            
    # Greedy Independent Set:
    # Sort points by degree (descending) - points with more conflicts are "harder" to include?
    # Actually, standard greedy MIS: pick vertex with highest degree, add to independent set? No, that's for Max Clique or something.
    # For Max Independent Set: Pick vertex, add to set, remove it and its neighbors.
    # Here "neighbors" are not simple. If I add point i, I cannot add any pair (j,k) such that (i,j,k) is a triple.
    
    # Simpler heuristic:
    # Start with empty set.
    # Iterate points. If adding point p to current set S doesn't create a triple with any pair in S, add it.
    # To make it better, try multiple orderings or random restarts.
    
    best_size = 0
    # Try a few random orderings to get a better estimate
    for _ in range(10):
        order = list(range(n))
        random.shuffle(order)
        current_set = []
        # For fast check, maintain a set of points in current_set
        # When adding p, check if there are j,k in current_set such that (p,j,k) is a triple.
        # Precompute: for each p, list of pairs (j,k) that form a triple with p.
        # This is expensive to build every time.
        
        # Alternative: Just use the number of triples as a proxy?
        # Higher number of triples usually means lower max 3-free subset.
        # Let's just return the number of triples as a proxy for optimization, 
        # and rely on the fact that minimizing triples correlates with minimizing the score.
        # Actually, the task says "Minimize the score". 
        # If I minimize the number of collinear triples, do I minimize the max 3-free subset?
        # Not necessarily. But usually, more constraints (triples) make it harder to find a large 3-free set.
        
        # Let's stick to a very fast metric: Number of collinear triples.
        # Minimizing this metric is what the previous `_greedy_improve` was doing.
        # The previous code failed because it called `_score` (exact) too often.
        # If I use `len(triples)` as the score, I can run many more iterations.
        
        return len(triples)
        
    return len(triples)

def _score(points):
    """Compute the score: largest subset with no three collinear.
    Using a fast greedy approximation to allow more search steps.
    """
    # For the purpose of local search, minimizing the number of collinear triples
    # is a good proxy for minimizing the maximum 3-collinear-free subset size.
    # We will use this proxy for optimization.
    triples = _build_triple_conflicts(points)
    return len(triples)

def _two_parabola_config():
    """
    Generate a configuration using two shifted parabolas.
    y = x^2 and y = x^2 + shift
    This creates cross-layer collinearities that limit the max independent set.
    """
    points = []
    seen = set()
    
    # First parabola: y = x^2
    for x in range(-4, 5):  # 9 points
        y = x * x
        if (x, y) not in seen and -100 <= x <= 100 and -100 <= y <= 100:
            points.append([x, y])
            seen.add((x, y))
    
    # Second parabola: y = x^2 + shift
    # Try different shifts to avoid 4-collinear
    for shift in [1, 2, 3, 5, 7]:
        for x in range(-4, 5):  # 9 points
            y = x * x + shift
            if (x, y) not in seen and -100 <= x <= 100 and -100 <= y <= 100:
                if len(points) < 20:
                    points.append([x, y])
                    seen.add((x, y))
                else:
                    break
        if len(points) >= 20:
            break
    
    # If we don't have 20, add more
    if len(points) < 20:
        for x in range(-10, 11):
            for y in range(-10, 11):
                if len(points) >= 20:
                    break
                if (x, y) not in seen:
                    points.append([x, y])
                    seen.add((x, y))
            if len(points) >= 20:
                break
    
    return points[:20]

def _perturb_to_valid(points, max_iterations=1000):
    """Perturb points to ensure no four are collinear."""
    for _ in range(max_iterations):
        if _elapsed() > _DEADLINE - 15:
            break
        if _check_no_four_collinear(points):
            return points
        
        # Find a 4-collinear line and move one point
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                line_pts = []
                for k in range(20):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_pts.append(k)
                if len(line_pts) >= 4:
                    # Move a random point from this line
                    idx = random.choice(line_pts)
                    old_x, old_y = points[idx]
                    for _ in range(50):
                        new_x = old_x + random.randint(-5, 5)
                        new_y = old_y + random.randint(-5, 5)
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            if (new_x, new_y) not in [(p[0], p[1]) for k, p in enumerate(points) if k != idx]:
                                points[idx] = [new_x, new_y]
                                found = True
                                break
                    if found:
                        break
            if found:
                break
        if not found:
            break
    
    return points

def _greedy_improve(points, max_iters=5000):
    """
    Greedy improvement: iteratively move points to increase the number of collinear triples.
    More triples generally implies a smaller maximum 3-collinear-free subset.
    """
    current_triples = _build_triple_conflicts(points)
    current_count = len(current_triples)
    
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
        
        improved = False
        # Try moving each point to a nearby position
        for idx in range(20):
            old_x, old_y = points[idx]
            best_new = None
            best_new_triples = current_count
            
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if -100 > new_x or new_x > 100 or -100 > new_y or new_y > 100:
                        continue
                    
                    # Check duplicate
                    is_dup = False
                    for k in range(20):
                        if k == idx: continue
                        if points[k][0] == new_x and points[k][1] == new_y:
                            is_dup = True
                            break
                    if is_dup: continue
                    
                    new_points = [list(p) for p in points]
                    new_points[idx] = [new_x, new_y]
                    
                    if not _check_no_four_collinear(new_points):
                        continue
                    
                    new_triples = _build_triple_conflicts(new_points)
                    if len(new_triples) > best_new_triples:
                        best_new_triples = len(new_triples)
                        best_new = [new_x, new_y]
            
            if best_new is not None:
                points[idx] = best_new
                current_triples = _build_triple_conflicts(points)
                current_count = len(current_triples)
                improved = True
                break
        
        if not improved:
            # Try random moves
            idx = random.randint(0, 19)
            old_x, old_y = points[idx]
            for _ in range(20):
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                is_dup = False
                for k in range(20):
                    if k == idx: continue
                    if points[k][0] == new_x and points[k][1] == new_y:
                        is_dup = True
                        break
                if is_dup: continue
                
                new_points = [list(p) for p in points]
                new_points[idx] = [new_x, new_y]
                
                if not _check_no_four_collinear(new_points):
                    continue
                
                new_triples = _build_triple_conflicts(new_points)
                if len(new_triples) > current_count:
                    points = new_points
                    current_triples = new_triples
                    current_count = len(new_triples)
                    improved = True
                    break
    
    return points

_start_time = time.time()

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_score = float('inf')
    
    # Strategy: Generate multiple candidates and optimize them
    candidates = []
    
    # 1. Structured starts: Parabolas
    for shift1 in [1, 2, 3, 4, 5]:
        if _elapsed() > _DEADLINE - 20:
            break
        points = []
        seen = set()
        
        # First parabola: y = x^2
        for x in range(-4, 5):
            y = x * x
            if (x, y) not in seen:
                points.append([x, y])
                seen.add((x, y))
        
        # Second parabola: y = x^2 + shift1
        for x in range(-4, 5):
            y = x * x + shift1
            if (x, y) not in seen and len(points) < 20:
                points.append([x, y])
                seen.add((x, y))
        
        # Fill remaining if needed
        if len(points) < 20:
            for x in range(-10, 11):
                for y in range(-10, 11):
                    if len(points) >= 20:
                        break
                    if (x, y) not in seen:
                        points.append([x, y])
                        seen.add((x, y))
                if len(points) >= 20:
                    break
        
        points = points[:20]
        valid_points = _perturb_to_valid(points)
        if _check_no_four_collinear(valid_points):
            candidates.append(valid_points)
            
    # 2. Random starts
    for _ in range(5):
        if _elapsed() > _DEADLINE - 20:
            break
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        valid_points = _perturb_to_valid(points, max_iterations=500)
        if _check_no_four_collinear(valid_points):
            candidates.append(valid_points)
            
    if not candidates:
        # Fallback hardcoded
        best_points = [
            [0, 0], [1, 1], [2, 4], [3, 9], [-1, 1], [-2, 4], [-3, 9],
            [4, 16], [-4, 16], [1, 2], [2, 5], [3, 10], [-1, 2], [-2, 5],
            [-3, 10], [4, 17], [-4, 17], [0, 1], [5, 25], [-5, 25]
        ]
        if not _check_no_four_collinear(best_points):
            best_points = _perturb_to_valid(best_points)
        return best_points

    # Evaluate and improve candidates
    # We use the proxy score (number of triples) for optimization
    # Lower number of triples is NOT what we want. We want to MINIMIZE the max 3-free subset.
    # Wait, the previous logic in _greedy_improve was to INCREASE triples.
    # Let's re-verify.
    # Score = Largest subset with no 3 collinear.
    # If I have MORE collinear triples, it is HARDER to find a large subset with no 3 collinear.
    # So, increasing the number of triples should DECREASE the score.
    # Therefore, in _greedy_improve, we want to INCREASE the number of triples.
    # And in the main loop, we want to find the config with the LOWEST score.
    # Since score ~ 1/triples (roughly), we want the config with the HIGHEST number of triples.
    
    # Let's evaluate candidates by their triple count (higher is better for minimizing score)
    # But we must be careful: high triple count doesn't guarantee low score, but it's a good proxy.
    
    best_triple_count = -1
    best_candidate = None
    
    for config in candidates:
        if _elapsed() > _DEADLINE - 10:
            break
        triples = _build_triple_conflicts(config)
        count = len(triples)
        if count > best_triple_count:
            best_triple_count = count
            best_candidate = config
            
    if best_candidate:
        # Improve the best candidate
        improved = _greedy_improve(best_candidate, max_iters=5000)
        if _check_no_four_collinear(improved):
            best_points = improved
        else:
            best_points = best_candidate
    else:
        best_points = candidates[0]
        
    return best_points
