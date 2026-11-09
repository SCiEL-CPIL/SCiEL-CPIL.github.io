import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate a pool of candidate points.
    # Use a mix of grid points and random points for diversity
    random.seed(42)
    
    coords = list(range(-80, 81, 10))
    pool = []
    seen = set()
    for x in coords:
        for y in coords:
            if (x, y) not in seen:
                pool.append((x, y))
                seen.add((x, y))
    
    # Add random points to break symmetry
    for _ in range(100):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            pool.append((x, y))
            seen.add((x, y))
            
    points = pool
    n_pool = len(points)
    
    # 2. Precompute line incidences for the pool.
    line_map = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        a, b = dy, -dx
        c = a * x1 + b * y1
        return (a, b, c)

    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            
    for key in line_map:
        line_map[key] = list(set(line_map[key]))
        
    # Precompute point_to_lines
    point_to_lines = [[] for _ in range(n_pool)]
    for key, pts in line_map.items():
        for p in pts:
            point_to_lines[p].append(key)

    def calc_score_fast(subset):
        """
        Returns (triple_count, is_valid)
        Uses incremental counting via line incidence
        """
        line_counts = {}
        for idx in subset:
            for lkey in point_to_lines[idx]:
                line_counts[lkey] = line_counts.get(lkey, 0) + 1
        
        triples = 0
        for lkey, count in line_counts.items():
            if count >= 4:
                return -1, False
            if count == 3:
                triples += 1
        
        return triples, True

    def random_valid_subset():
        """Find a random valid subset of size 20"""
        for _ in range(500):
            subset = random.sample(range(n_pool), 20)
            _, valid = calc_score_fast(subset)
            if valid:
                return subset
        # Fallback: try greedy construction
        subset = []
        used_lines = set()
        candidates = list(range(n_pool))
        random.shuffle(candidates)
        for idx in candidates:
            if len(subset) == 20:
                break
            # Check if adding this point creates 4-collinear
            valid_add = True
            for lkey in point_to_lines[idx]:
                count = sum(1 for s in subset if lkey in point_to_lines[s])
                if count >= 3:
                    valid_add = False
                    break
            if valid_add:
                subset.append(idx)
        if len(subset) == 20:
            return subset
        return None

    best_subset = None
    best_triples = -1
    
    # Generate initial population with multiple restarts
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        s = random_valid_subset()
        if s is None:
            continue
        t, _ = calc_score_fast(s)
        if t > best_triples:
            best_triples = t
            best_subset = s[:]
            
    if best_subset is None:
        # Fallback: construct a valid set manually
        pts = []
        for i in range(5):
            for j in range(4):
                pts.append([i*20-40, j*20-30])
        return pts

    current_subset = best_subset[:]
    current_triples = best_triples
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.997
    no_improve_count = 0
    max_no_improve = 200
    
    outside = set(range(n_pool)) - set(current_subset)
    
    while time.time() < deadline - 3:
        if T < T_min:
            break
        
        improved = False
        
        # Try multiple swaps per iteration
        num_swaps = 100
        for _ in range(num_swaps):
            if time.time() > deadline - 3:
                break
            
            i = random.randint(0, 19)
            old_idx = current_subset[i]
            
            if not outside:
                break
            new_idx = random.choice(list(outside))
            
            new_subset = current_subset[:]
            new_subset[i] = new_idx
            
            new_triples, valid = calc_score_fast(new_subset)
            
            if valid:
                if new_triples > current_triples:
                    current_subset = new_subset
                    current_triples = new_triples
                    improved = True
                    outside.remove(new_idx)
                    outside.add(old_idx)
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best_subset = current_subset[:]
                        no_improve_count = 0
                else:
                    delta = new_triples - current_triples
                    if delta == 0:
                        # Accept equal moves with some probability to explore
                        if random.random() < 0.3:
                            current_subset = new_subset
                            outside.remove(new_idx)
                            outside.add(old_idx)
                    elif random.random() < pow(2.71828, delta / T):
                        current_subset = new_subset
                        current_triples = new_triples
                        outside.remove(new_idx)
                        outside.add(old_idx)
            else:
                pass
                
        T *= cooling
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve and time.time() < deadline - 25:
                # Random restart
                s = random_valid_subset()
                if s is not None:
                    t, _ = calc_score_fast(s)
                    if t > current_triples * 0.8:  # Accept if not too bad
                        current_subset = s
                        current_triples = t
                        outside = set(range(n_pool)) - set(current_subset)
                        no_improve_count = 0
                        T = max(T, 0.5)  # Reheat

    final_points = [points[i] for i in best_subset]
    return [[p[0], p[1]] for p in final_points]
