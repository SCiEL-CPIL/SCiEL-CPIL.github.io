import time
import random
import math

DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                    if slopes[slope] >= 3:
                        return True
                else:
                    slopes[slope] = 1
        return False
    
    def calc_score(points):
        """Exact B&B for largest subset with no 3 collinear, using bitmask."""
        n = len(points)
        
        # Precompute: for each vertex v, list of (u, w) pairs where v,u,w are collinear
        # and u < w in the processing order
        triple_count = [0] * n
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triple_count[i] += 1
                        triple_count[j] += 1
                        triple_count[k] += 1
                        triples.append((i, j, k))
        
        # Sort by triple count descending (hard vertices first)
        order = sorted(range(n), key=lambda v: -triple_count[v])
        pos = {v: i for i, v in enumerate(order)}
        
        # For each position, store forbidden pairs
        forbidden = [[] for _ in range(n)]
        for i, j, k in triples:
            pi, pj, pk = pos[i], pos[j], pos[k]
            # Ensure we store pairs with smaller index first for consistency
            forbidden[pi].append((pj, pk))
            forbidden[pj].append((pi, pk))
            forbidden[pk].append((pi, pj))
        
        best = [0]
        chosen = [False] * n
        
        def backtrack(pos_idx, current_size):
            if pos_idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - pos_idx) <= best[0]:
                return
            
            # Try including this vertex
            v = pos_idx
            valid = True
            for u, w in forbidden[v]:
                if chosen[u] and chosen[w]:
                    valid = False
                    break
            
            if valid:
                chosen[v] = True
                backtrack(pos_idx + 1, current_size + 1)
                chosen[v] = False
            
            # Try excluding
            backtrack(pos_idx + 1, current_size)
        
        backtrack(0, 0)
        return best[0]
    
    def generate_structured(seed_val):
        """Generate structured point configurations with many collinear triples."""
        random.seed(seed_val)
        patterns = []
        
        # Pattern 1: Grid-like with perturbations
        for offset in range(5):
            pts = []
            for i in range(4):
                for j in range(5):
                    x = -40 + i * 20 + random.randint(-2, 2)
                    y = -40 + j * 20 + random.randint(-2, 2)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 2: Circle with perturbations
        for angle_off in [0, 0.1, 0.2]:
            pts = []
            for i in range(20):
                angle = 2 * math.pi * i / 20 + angle_off
                r = 50 + 8 * math.sin(i * 1.5)
                x = int(round(r * math.cos(angle)))
                y = int(round(r * math.sin(angle)))
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 3: Clustered
        for cx, cy in [(0,0), (20,20), (-20,-20), (30,-30)]:
            pts = []
            for i in range(20):
                x = cx + random.randint(-25, 25)
                y = cy + random.randint(-25, 25)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        return patterns
    
    def count_triples(points):
        """Count collinear triples for use as surrogate."""
        count = 0
        for i in range(20):
            slopes = {}
            xi, yi = points[i]
            for j in range(20):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count
    
    best_points = None
    best_score = 21
    best_triples = 0
    
    # Try structured initializations
    structured = generate_structured(42)
    for pts in structured:
        if time.time() - start_time > DEADLINE:
            break
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    # Also try random initializations
    for trial in range(20):
        if time.time() - start_time > DEADLINE:
            break
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    if best_points is None:
        # Fallback: random valid config
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
                best_score = calc_score(pts)
                best_triples = count_triples(pts)
                break
    
    # Local search / simulated annealing to improve
    current_points = best_points[:]
    current_triples = best_triples
    current_score = best_score
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Try moving to a nearby point or random
        if random.random() < 0.7:
            # Local move
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            x = max(-100, min(100, old_p[0] + dx))
            y = max(-100, min(100, old_p[1] + dy))
        else:
            # Global move
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        
        new_p = (x, y)
        if new_p == old_p:
            continue
        if new_p in current_points:
            continue
        
        current_points[idx] = new_p
        
        if has_four_collinear(current_points):
            current_points[idx] = old_p
            continue
        
        new_triples = count_triples(current_points)
        
        # Accept if triples increased or with probability based on SA
        if new_triples > current_triples:
            current_triples = new_triples
            # Check exact score periodically
            if random.random() < 0.1 or new_triples > best_triples:
                exact_score = calc_score(current_points)
                if exact_score < best_score or (exact_score == best_score and new_triples > best_triples):
                    best_score = exact_score
                    best_triples = new_triples
                    best_points = current_points[:]
        elif random.random() < math.exp((new_triples - current_triples) / T):
            current_triples = new_triples
        else:
            current_points[idx] = old_p
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return [[p[0], p[1]] for p in best_points]
