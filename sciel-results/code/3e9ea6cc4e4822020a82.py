"""Minimize largest 3-collinear-free subset for 20 points."""
import time
import random
import itertools
import sys

def _no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
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

def _collinear_triples(points):
    """Return list of collinear triples (as index tuples)."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            line_pts = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_pts.append(k)
            if len(line_pts) >= 3:
                # Add all triples from this line (but avoid duplicates)
                for a in range(len(line_pts)):
                    for b in range(a+1, len(line_pts)):
                        for c in range(b+1, len(line_pts)):
                            triples.append(tuple(sorted([line_pts[a], line_pts[b], line_pts[c]])))
    # Deduplicate
    return list(set(triples))

def _greedy_independent(points, triples):
    """Greedy upper bound on largest 3-collinear-free subset."""
    n = len(points)
    in_set = [True] * n
    # Greedily add points, removing those that would form a triple
    # Sort by degree (number of triples containing them) descending
    degree = [0] * n
    for t in triples:
        for idx in t:
            degree[idx] += 1
    order = sorted(range(n), key=lambda i: -degree[i])
    result = []
    for idx in order:
        if not in_set[idx]:
            continue
        # Check if adding idx creates a triple with two already-in-set points
        ok = True
        for t in triples:
            if idx in t:
                others = [x for x in t if x != idx]
                if all(in_set[o] and o in result for o in others):
                    ok = False
                    break
        if ok:
            result.append(idx)
    return len(result)

def _exact_max_independent(points, triples, deadline):
    """Branch and bound for exact largest 3-collinear-free subset."""
    n = len(points)
    triples_set = set(triples)
    
    # Build adjacency: for each point, which other points it's in a triple with
    # And for each pair, whether they're in a triple together
    pair_in_triple = {}
    for t in triples:
        a, b, c = t
        pair_in_triple[(a, b)] = True
        pair_in_triple[(b, a)] = True
        pair_in_triple[(a, c)] = True
        pair_in_triple[(c, a)] = True
        pair_in_triple[(b, c)] = True
        pair_in_triple[(c, b)] = True
    
    best = 0
    selected = []
    
    def branch(start_idx, current_size, current_set):
        nonlocal best
        if time.time() > deadline:
            return
        # Upper bound: current_size + remaining points
        remaining = n - start_idx
        if current_size + remaining <= best:
            return
        if start_idx == n:
            best = max(best, current_size)
            return
        
        # Option 1: skip point start_idx
        branch(start_idx + 1, current_size, current_set)
        
        # Option 2: include point start_idx if no conflict
        can_include = True
        for p in current_set:
            if pair_in_triple.get((start_idx, p), False):
                # Check if there's a third point in the triple already selected
                for t in triples:
                    if start_idx in t and p in t:
                        third = [x for x in t if x != start_idx and x != p][0]
                        if third in current_set:
                            can_include = False
                            break
                if not can_include:
                    break
        if can_include:
            branch(start_idx + 1, current_size + 1, current_set | {start_idx})
    
    branch(0, 0, frozenset())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    random.seed(42)
    
    best_points = None
    best_score = 21
    
    # Try several random initial configurations
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        # Generate 20 random distinct points
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 1000:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in points:
                points.append(p)
            attempts += 1
        
        if len(points) < 20:
            continue
        
        if not _no_four_collinear(points):
            continue
        
        # Compute triples and greedy score
        triples = _collinear_triples(points)
        greedy_score = _greedy_independent(points, triples)
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_points = list(points)
        
        # Local search: try swapping one point
        if time.time() > deadline - 30:
            break
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(20):
                if time.time() > deadline - 30:
                    break
                for _ in range(3):  # try a few random replacements
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = (new_x, new_y)
                    if new_p in points:
                        continue
                    new_points = list(points)
                    new_points[i] = new_p
                    if not _no_four_collinear(new_points):
                        continue
                    new_triples = _collinear_triples(new_points)
                    new_greedy = _greedy_independent(new_points, new_triples)
                    if new_greedy < greedy_score:
                        points = new_points
                        triples = new_triples
                        greedy_score = new_greedy
                        if greedy_score < best_score:
                            best_score = greedy_score
                            best_points = list(points)
                        improved = True
                        break
    
    # Final exact computation on best found
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    if time.time() < deadline - 5:
        triples = _collinear_triples(best_points)
        exact_score = _exact_max_independent(best_points, triples, deadline - 2)
        best_score = exact_score
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
