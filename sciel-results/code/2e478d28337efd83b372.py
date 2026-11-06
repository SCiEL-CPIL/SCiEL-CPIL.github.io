import time
import random
import math
from collections import defaultdict

def _line_key(p1, p2):
    """Return canonical (a,b,c) for line through p1,p2, with a>=0, gcd(a,b,c)=1."""
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    # Normalize
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g != 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_lines(points):
    """Count lines with exactly 4 points, and check no 5 collinear."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] += 1
    # Each line with k points is counted C(k,2) times
    # We need to find actual point counts
    # Reconstruct: for each unique line key, count actual points on it
    exact4 = 0
    has5 = False
    for key, pair_count in line_counts.items():
        a, b, c = key
        # Count actual points on this line
        cnt = 0
        for p in points:
            x, y = p
            if a * x + b * y + c == 0:
                cnt += 1
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            has5 = True
    return exact4, has5

def _fitness(points):
    """Higher is better. Penalize 5-collinear heavily."""
    exact4, has5 = _count_lines(points)
    if has5:
        return -1000 + exact4
    return exact4

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _random_config():
    pts = set()
    while len(pts) < 16:
        pts.add(tuple(_random_point()))
    return [list(p) for p in pts]

def _grid_config():
    return [[x, y] for x in range(4) for y in range(4)]

def _mutate(points, rate=0.3):
    new_pts = [list(p) for p in points]
    for i in range(16):
        if random.random() < rate:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            new_pts[i][0] = max(-1000, min(1000, new_pts[i][0] + dx))
            new_pts[i][1] = max(-1000, min(1000, new_pts[i][1] + dy))
    # Ensure distinctness
    seen = set()
    unique = []
    for p in new_pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    while len(unique) < 16:
        new_p = _random_point()
        if tuple(new_p) not in seen:
            seen.add(tuple(new_p))
            unique.append(new_p)
    return unique[:16]

def _crossover(p1, p2):
    """Select random subset from p1, complement from p2."""
    indices = list(range(16))
    random.shuffle(indices)
    k = random.randint(8, 15)
    set1 = set(indices[:k])
    child = [None] * 16
    for i in range(16):
        if i in set1:
            child[i] = list(p1[i])
        else:
            child[i] = list(p2[i])
    # Ensure distinctness
    seen = set()
    unique = []
    for p in child:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    while len(unique) < 16:
        new_p = _random_point()
        if tuple(new_p) not in seen:
            seen.add(tuple(new_p))
            unique.append(new_p)
    return unique[:16]

def _local_search(points, max_iters=200):
    """Hill-climbing: try small perturbations to improve fitness."""
    best = [list(p) for p in points]
    best_fit = _fitness(best)
    for _ in range(max_iters):
        improved = False
        for i in range(16):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [list(p) for p in best]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                        continue
                    # Check distinctness
                    seen = set()
                    dup = False
                    for j, p in enumerate(new_pts):
                        t = tuple(p)
                        if t in seen:
                            dup = True
                            break
                        seen.add(t)
                    if dup:
                        continue
                    fit = _fitness(new_pts)
                    if fit > best_fit:
                        best = new_pts
                        best_fit = fit
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_fit

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin
    
    random.seed(42)
    
    # Initialize population
    pop = [_grid_config()]
    pop.append(_grid_config())
    # Add some random configs
    for _ in range(7):
        pop.append(_random_config())
    
    # Evaluate initial population
    scores = [_fitness(p) for p in pop]
    best_idx = max(range(len(pop)), key=lambda i: scores[i])
    best_config = [list(p) for p in pop[best_idx]]
    best_score = scores[best_idx]
    
    # Genetic algorithm
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: tournament
        def tournament():
            idxs = random.sample(range(len(pop)), min(3, len(pop)))
            return max(idxs, key=lambda i: scores[i])
        
        new_pop = [list(p) for p in pop]
        
        # Create offspring
        for _ in range(len(pop)):
            if time.time() > deadline:
                break
            i1 = tournament()
            i2 = tournament()
            if random.random() < 0.7:
                child = _crossover(pop[i1], pop[i2])
            else:
                child = _mutate(pop[i1])
            
            # Local search on child
            child, child_fit = _local_search(child, max_iters=50)
            
            new_pop.append(child)
            new_pop_scores = [_fitness(p) for p in new_pop]
            # Actually, let's just track separately
            new_pop[-1] = child
            # We'll re-evaluate at the end
        
        # Re-evaluate all
        all_configs = pop + new_pop[len(pop):]
        all_scores = [_fitness(p) for p in all_configs]
        
        # Keep best half
        n = len(all_configs)
        order = sorted(range(n), key=lambda i: all_scores[i], reverse=True)
        keep = n // 2
        pop = [all_configs[i] for i in order[:keep]]
        scores = [all_scores[i] for i in order[:keep]]
        
        # Track best
        best_i = max(range(len(pop)), key=lambda i: scores[i])
        if scores[best_i] > best_score:
            best_score = scores[best_i]
            best_config = [list(p) for p in pop[best_i]]
    
    # Final local search on best
    best_config, best_score = _local_search(best_config, max_iters=500)
    
    return best_config
