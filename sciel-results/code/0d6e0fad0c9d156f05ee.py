import random
import time

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 0
            for k in range(n):
                if k != i and k != j and _are_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 2:
                        return True
    return False

def _count_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_no_three_collinear(points):
    n = len(points)
    # Precompute triple partners using bitmask
    triple_partners = [0] * n  # bitmask of pairs that conflict with point i
    # For each point i, store which pairs (a,b) cause a triple if i is added
    # Use a list of bitmasks: triple_conflict[i] = bitmask where bit j is set if 
    # adding i would create a triple with two already-selected points
    # Actually, let's use the pair-based approach from inspiration 2
    
    # pair_conflicts[i][j] = bitmask of k such that i,j,k are collinear
    pair_conflicts = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _are_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            pair_conflicts[i][j] = mask
            pair_conflicts[j][i] = mask
    
    # Sort by degree for better pruning
    degrees = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                degrees[i] += bin(pair_conflicts[i][j]).count('1')
    
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    order_inv = [0] * n
    for idx, i in enumerate(order):
        order_inv[i] = idx
    
    best = 0
    n_points = n
    
    def dfs(idx, current_size, included_mask):
        nonlocal best
        if current_size > best:
            best = current_size
        
        if idx == n_points:
            return
        
        # Upper bound
        if current_size + (n_points - idx) <= best:
            return
        
        i = order[idx]
        
        # Check if adding i is valid: no two already-included points j,k should have i in their conflict set
        valid = True
        for j in range(idx):
            if included_mask & (1 << order[j]):
                if included_mask & pair_conflicts[i][order[j]]:
                    valid = False
                    break
        
        if valid:
            dfs(idx + 1, current_size + 1, included_mask | (1 << i))
        
        dfs(idx + 1, current_size, included_mask)
    
    dfs(0, 0, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def generate_initial():
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 5000:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in points:
                points.append(p)
                if _has_four_collinear(points):
                    points.pop()
            attempts += 1
        if len(points) < 20:
            # Fallback: use a structured set
            points = []
            for x in range(-10, 10):
                points.append([x, x * x % 201 - 100])
            # Ensure distinctness and bounds
            seen = set()
            unique_points = []
            for p in points:
                if tuple(p) not in seen and -100 <= p[0] <= 100 and -100 <= p[1] <= 100:
                    seen.add(tuple(p))
                    unique_points.append(p)
            points = unique_points[:20]
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    points.append(p)
        return points
    
    def get_line_candidates(p1, p2, exclude_set):
        """Get all integer points on line through p1,p2 within bounds, excluding those in exclude_set"""
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0] * p1[1] - p1[0] * p2[1]
        candidates = []
        
        if A == 0 and B == 0:
            return candidates
        
        if A == 0:
            # Horizontal line: y = const
            if B != 0 and C % B == 0:
                y = -C // B
                if -100 <= y <= 100:
                    for x in range(-100, 101):
                        p = [x, y]
                        if p not in exclude_set:
                            candidates.append(p)
        elif B == 0:
            # Vertical line: x = const
            if A != 0 and C % A == 0:
                x = -C // A
                if -100 <= x <= 100:
                    for y in range(-100, 101):
                        p = [x, y]
                        if p not in exclude_set:
                            candidates.append(p)
        else:
            # General line: A*x + B*y + C = 0
            for x in range(-100, 101):
                num = -(A * x + C)
                if num % B == 0:
                    y = num // B
                    if -100 <= y <= 100:
                        p = [x, y]
                        if p not in exclude_set:
                            candidates.append(p)
        
        return candidates
    
    points = generate_initial()
    
    # Initial evaluation
    best_triples = _count_triples(points)
    best_score = _max_no_three_collinear(points)
    best_points = [p[:] for p in points]
    current_triples = best_triples
    
    # Simulated annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.5
    
    while time.time() < deadline:
        # Pick a point to move
        idx = random.randint(0, 19)
        old_p = points[idx]
        old_p_set = set(map(tuple, points))
        
        # Strategy: pick a random pair of other points and move to a point on their line
        # This encourages creating new collinear triples
        other_indices = [i for i in range(20) if i != idx]
        
        if random.random() < 0.7:
            # Line-based move: pick two other points, move to a point on their line
            i1, i2 = random.sample(other_indices, 2)
            p1, p2 = points[i1], points[i2]
            
            exclude_set = set(map(tuple, points))
            candidates = get_line_candidates(p1, p2, exclude_set)
            
            if not candidates:
                # Fallback to random
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = [x, y]
                if new_p in points or new_p == old_p:
                    continue
            else:
                new_p = random.choice(candidates)
                if new_p == old_p:
                    continue
        else:
            # Random move
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_p = [x, y]
            if new_p in points or new_p == old_p:
                continue
        
        points[idx] = new_p
        
        if _has_four_collinear(points):
            points[idx] = old_p
            continue
        
        new_triples = _count_triples(points)
        delta = new_triples - current_triples
        
        # Accept if better, or with probability exp(delta/T) for worse
        if delta >= 0 or random.random() < (2.71828 ** (delta / T)):
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                # Check exact score periodically
                if time.time() < deadline - 10:
                    current_score = _max_no_three_collinear(points)
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
        else:
            points[idx] = old_p
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final check
    if time.time() < deadline - 5:
        final_score = _max_no_three_collinear(points)
        if final_score < best_score:
            best_score = final_score
            best_points = [p[:] for p in points]
    
    return best_points
