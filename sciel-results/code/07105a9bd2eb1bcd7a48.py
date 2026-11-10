import time
import random
from math import gcd
from collections import defaultdict
import sys

_START_TIME = None
_DEADLINE = 135.0  # Leave 15s margin before 150s kill

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

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

def _get_triples(points):
    """Get all collinear triples as a list of (i, j, k) tuples."""
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
                            triple = (line_points[a], line_points[b], line_points[c])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _max_disjoint_triples(triples, n):
    """
    Find the maximum number of disjoint collinear triples using backtracking.
    triples: list of (i, j, k)
    n: number of points
    """
    if not triples:
        return 0
    
    # Sort triples by length (all 3) or some heuristic
    # We want to maximize the count of disjoint triples.
    best = [0]
    used = [False] * n
    
    # Order triples to improve pruning
    # Triples that share many points with others are more "constrained"
    # Or simply try to find a maximal matching greedily first to get a lower bound
    # For backtracking, we can just iterate.
    
    # To speed up, we can use a simple greedy matching to initialize best
    def greedy_matching():
        used_g = [False] * n
        count = 0
        remaining = list(range(len(triples)))
        random.shuffle(remaining)
        for t_idx in remaining:
            i, j, k = triples[t_idx]
            if not used_g[i] and not used_g[j] and not used_g[k]:
                used_g[i] = True
                used_g[j] = True
                used_g[k] = True
                count += 1
        return count
    
    best[0] = greedy_matching()
    
    # Backtracking search for better matching
    # This might be slow if there are many triples, but N=20 is small.
    # Number of triples is at most C(20,3) = 1140.
    # But realistically much fewer.
    
    def backtrack(idx, count):
        if _elapsed() > _DEADLINE - 5:
            return
        if count + (len(triples) - idx) <= best[0]:
            return
        if idx == len(triples):
            if count > best[0]:
                best[0] = count
            return
        
        # Option 1: Skip this triple
        backtrack(idx + 1, count)
        
        # Option 2: Take this triple if disjoint
        i, j, k = triples[idx]
        if not used[i] and not used[j] and not used[k]:
            used[i] = True
            used[j] = True
            used[k] = True
            backtrack(idx + 1, count + 1)
            used[i] = False
            used[j] = False
            used[k] = False
            
    # To make backtracking faster, we can sort triples
    # Triples that are "harder" to pack (share points with many others) first?
    # Or just random order with multiple restarts?
    # Given the small N, a single backtracking run with a good initial bound might be enough.
    # But to be safe, we can do a few random shuffles.
    
    # Actually, for N=20, the number of triples is small.
    # Let's just run the backtracking once.
    backtrack(0, 0)
    
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _get_triples(points)
    if not triples:
        return len(points)
    
    n = len(points)
    # Build conflict graph for B&B
    # conflict[i] = set of (j, k) such that {i, j, k} is a triple
    conflict = [set() for _ in range(n)]
    for (a, b, c) in triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))
    
    best = [0]
    selected = [False] * n
    
    # Order by degree
    degrees = [len(conflict[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    def backtrack(pos, count):
        if _elapsed() > _DEADLINE - 2:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
        
        if count + (n - pos) <= best[0]:
            return
        
        idx = order[pos]
        
        # Try including idx
        can_include = True
        for (j, k) in conflict[idx]:
            if selected[j] and selected[k]:
                can_include = False
                break
        
        if can_include:
            selected[idx] = True
            backtrack(pos + 1, count + 1)
            selected[idx] = False
        
        # Try excluding idx
        backtrack(pos + 1, count)
    
    backtrack(0, 0)
    return best[0]

def _local_search_for_triples(points, max_time=10.0):
    """
    Local search to maximize the number of disjoint collinear triples.
    """
    start_time = time.time()
    current_points = [list(p) for p in points]
    current_triples = _get_triples(current_points)
    current_matching = _max_disjoint_triples(current_triples, 20)
    best_points = [list(p) for p in current_points]
    best_matching = current_matching
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        if _elapsed() > _DEADLINE - 10:
            break
        if no_improve > max_no_improve:
            break
            
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        moved = False
        for _ in range(20):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = old_x + dx
            new_y = old_y + dy
            
            if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                continue
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(current_points) if i != idx]:
                continue
                
            new_points = [list(p) for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
                
            new_triples = _get_triples(new_points)
            new_matching = _max_disjoint_triples(new_triples, 20)
            
            if new_matching > current_matching:
                current_points = new_points
                current_triples = new_triples
                current_matching = new_matching
                if current_matching > best_matching:
                    best_matching = current_matching
                    best_points = [list(p) for p in current_points]
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
            else:
                no_improve += 1
        
        if not moved:
            no_improve += 1
            
    return best_points

def _random_valid_config():
    """Generate a random valid configuration."""
    while True:
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            key = (x, y)
            if key not in seen:
                seen.add(key)
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points

def _grid_config():
    """Generate a grid-based configuration with perturbations."""
    random.seed(123)
    candidates = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            candidates.append([x, y])
    random.shuffle(candidates)
    points = [list(p) for p in candidates[:20]]
    
    # Perturb to fix 4-collinear
    for _ in range(100):
        if _check_no_four_collinear(points):
            break
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
                    idx = random.choice(line_pts)
                    old_x, old_y = points[idx]
                    new_x = old_x + random.randint(-3, 3)
                    new_y = old_y + random.randint(-3, 3)
                    if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                        if (new_x, new_y) not in [(p[0], p[1]) for k, p in enumerate(points) if k != idx]:
                            points[idx] = [new_x, new_y]
                            found = True
                            break
            if found:
                break
    return points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    candidates = []
    
    # Grid config
    candidates.append(_grid_config())
    
    # Random configs
    for _ in range(5):
        if _elapsed() > _DEADLINE - 60:
            break
        candidates.append(_random_valid_config())
        
    # Improve each candidate
    for config in candidates:
        if _elapsed() > _DEADLINE - 20:
            break
        
        if not _check_no_four_collinear(config):
            continue
            
        improved = _local_search_for_triples(config, max_time=15.0)
        
        if _elapsed() > _DEADLINE - 10:
            break
            
        s = _score(improved)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in improved]
            
    # Final polish on the best
    if best_points and _elapsed() < _DEADLINE - 15:
        improved = _local_search_for_triples(best_points, max_time=15.0)
        if _elapsed() < _DEADLINE - 5:
            s = _score(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
                
    if best_points is None:
        best_points = _random_valid_config()
        
    return best_points
