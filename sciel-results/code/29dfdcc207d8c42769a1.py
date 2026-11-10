"""Minimize largest 3-collinear-free subset for 20 points (no four collinear)."""
import time
import random
from math import gcd

# Deadline guard: kill after 135s to stay well under 150s evaluator limit
_DEADLINE = 135.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
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

def _build_triple_conflicts(points):
    """Build list of collinear triples as frozensets of 3 indices."""
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
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

def _max_independent_set_size(triples, n):
    """
    Exact branch and bound for maximum subset with no three collinear.
    Returns the size of the largest subset containing no complete triple.
    """
    index_to_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for p in t:
            index_to_triples[p].append(idx)
    
    triples_set = [set(t) for t in triples]
    
    best = [0]
    current = [False] * n
    
    degrees = [len(index_to_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    def backtrack(pos, count):
        if _elapsed() > _DEADLINE - 5:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        if count + (n - pos) <= best[0]:
            return
        
        idx = order[pos]
        
        can_include = True
        for t_idx in index_to_triples[idx]:
            t = triples_set[t_idx]
            others = [p for p in t if p != idx]
            if len(others) == 2:
                if current[others[0]] and current[others[1]]:
                    can_include = False
                    break
        
        if can_include:
            current[idx] = True
            backtrack(pos + 1, count + 1)
            current[idx] = False
        
        backtrack(pos + 1, count)
    
    backtrack(0, 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _build_triple_conflicts(points)
    return _max_independent_set_size(triples, len(points))

def _two_parabola_config(shift=1):
    """Generate 20 points from two shifted parabolas."""
    points = []
    x_range = list(range(-5, 5))
    
    for x in x_range:
        points.append([x, x * x])
    for x in x_range:
        points.append([x, x * x + shift])
    
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    if len(unique) != 20:
        return None
    
    for x, y in unique:
        if abs(x) > 100 or abs(y) > 100:
            return None
    
    return unique

def _perturb(points):
    """Perturb one or two points slightly."""
    new_points = [list(p) for p in points]
    num_perturb = random.choice([1, 2])
    for _ in range(num_perturb):
        idx = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]
    
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_score = 21
    
    # Start with two-parabola configurations
    for shift in [1, 2, 3, -1, -2, 5, 7, 4, 6, -3]:
        if _elapsed() > _DEADLINE - 10:
            break
        config = _two_parabola_config(shift)
        if config is None:
            continue
        if not _check_no_four_collinear(config):
            continue
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    # Local search from best configuration
    if best_points is not None:
        current = [list(p) for p in best_points]
        current_score = best_score
        stall_count = 0
        
        while _elapsed() < _DEADLINE - 15:
            perturbed = _perturb(current)
            if perturbed is None:
                stall_count += 1
                if stall_count > 50:
                    break
                continue
            
            if not _check_no_four_collinear(perturbed):
                stall_count += 1
                if stall_count > 50:
                    break
                continue
            
            s = _score(perturbed)
            if s < current_score:
                current = perturbed
                current_score = s
                best_points = [list(p) for p in current]
                best_score = s
                stall_count = 0
            else:
                stall_count += 1
    
    if best_points is None:
        # Fallback: simple valid configuration
        best_points = []
        for i in range(10):
            best_points.append([i, i * i])
            best_points.append([i, i * i + 1])
    
    return best_points
