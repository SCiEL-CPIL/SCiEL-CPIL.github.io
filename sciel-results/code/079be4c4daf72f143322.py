"""Minimize largest 3-collinear-free subset for 20 points via structured combinatorial search."""
import time
import random
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
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

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Uses bitmask-based conflict representation for speed.
    """
    n = len(points)
    
    # Precompute: for each pair (i,j), bitmask of points k collinear with i,j
    # conflict_mask[i] = bitmask of points that cannot coexist with i in a valid set
    # Actually, for 3-uniform: if we have set S, adding p is invalid if 
    # there exist a,b in S such that a,b,p collinear.
    # 
    # Better: precompute for each point p, and each other point q, the set of r
    # such that p,q,r collinear. Then when building set S, if we add p, we must
    # remove from candidates all r where (p, q, r) collinear for some q in S.
    
    # Precompute collinear triples as a list
    # For each point i, store a list of (j, k) pairs where i,j,k collinear
    # Actually, let's precompute: for each pair (i,j), the bitmask of k's collinear with them
    
    # collinear_mask[i][j] = bitmask of k such that i,j,k are collinear (k != i, k != j)
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask
    
    best = 0
    
    def backtrack(current_mask, candidates_mask, current_size):
        nonlocal best
        if current_size + bin(candidates_mask).count('1') <= best:
            return
        if candidates_mask == 0:
            if current_size > best:
                best = current_size
            return
        
        # Pick a candidate: the one with the highest index (or lowest, doesn't matter much)
        # Use lowest set bit for efficiency
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)
        
        # Branch 1: Include p_idx
        # Remove from candidates any k that forms a collinear triple with p_idx and any point in current_set
        # We need to know which points are in current_set. Track via current_mask.
        new_cand = rest_mask
        # For each q in current_set, remove collinear_mask[p_idx][q] from new_cand
        cm = current_mask
        while cm:
            q_bit = cm & -cm
            q_idx = q_bit.bit_length() - 1
            new_cand &= ~collinear_mask[p_idx][q_idx]
            cm ^= q_bit
        
        backtrack(current_mask | (1 << p_idx), new_cand, current_size + 1)
        
        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask, current_size)
    
    backtrack(0, (1 << n) - 1, 0)
    return best

def _generate_parabola_variant(seed):
    """Generate points near a parabola with perturbations."""
    random.seed(seed)
    pts = []
    for i in range(20):
        x = i - 10
        y = x * x // 4  # Parabolic shape
        # Add small perturbation
        px = x + random.randint(-2, 2)
        py = y + random.randint(-3, 3)
        px = max(-100, min(100, px))
        py = max(-100, min(100, py))
        pts.append([px, py])
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts.append(p)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def _generate_grid_based(seed):
    """Generate points on a structured grid with deliberate 3-collinear lines."""
    random.seed(seed)
    pts = []
    # Use a 5x4 grid pattern with some offsets
    base_positions = []
    for row in range(4):
        for col in range(5):
            x = col * 20 - 40
            y = row * 25 - 30
            base_positions.append([x, y])
    random.shuffle(base_positions)
    pts = base_positions[:20]
    # Add small random offsets
    for i in range(20):
        pts[i][0] += random.randint(-3, 3)
        pts[i][1] += random.randint(-3, 3)
    # Ensure within bounds
    for i in range(20):
        pts[i][0] = max(-100, min(100, pts[i][0]))
        pts[i][1] = max(-100, min(100, pts[i][1]))
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts.append(p)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def _generate_modular_construction(seed):
    """Generate points using modular arithmetic for structured collinear triples."""
    random.seed(seed)
    pts = []
    # Place points at (i, i^2 mod p) for prime p, mapped to range
    p = 19  # Prime
    for i in range(p):
        x = i - 9
        y = (i * i) % p - 9
        pts.append([x, y])
    # Add one more point
    pts.append([random.randint(-100, 100), random.randint(-100, 100)])
    # Ensure distinctness and bounds
    seen = set()
    unique_pts = []
    for p_pt in pts:
        pp = [max(-100, min(100, p_pt[0])), max(-100, min(100, p_pt[1]))]
        if tuple(pp) not in seen:
            seen.add(tuple(pp))
            unique_pts.append(pp)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def _generate_symmetric(seed):
    """Generate points with rotational symmetry."""
    random.seed(seed)
    pts = []
    # 4-fold symmetry: 5 points per quadrant
    for quadrant in range(4):
        for k in range(5):
            angle = (quadrant * 90 + k * 18 + random.randint(-5, 5)) * 3.14159 / 180
            r = 20 + k * 10 + random.randint(-5, 5)
            x = int(r * __import__('math').cos(angle))
            y = int(r * __import__('math').sin(angle))
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            pts.append([x, y])
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts.append(p)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def _crossover(parent1, parent2):
    """Crossover: take half from each parent."""
    n = 20
    pts = []
    for i in range(n):
        if random.random() < 0.5:
            pts.append(parent1[i][:])
        else:
            pts.append(parent2[i][:])
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts.append(p)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def _mutate(points, rate=0.3):
    """Mutate: move some points to random locations."""
    pts = [p[:] for p in points]
    for i in range(20):
        if random.random() < rate:
            pts[i][0] = random.randint(-100, 100)
            pts[i][1] = random.randint(-100, 100)
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts.append(p)
    while len(unique_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_pts.append([x, y])
    return unique_pts[:20]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate diverse initial populations from different construction families
    generators = [
        _generate_parabola_variant,
        _generate_grid_based,
        _generate_modular_construction,
        _generate_symmetric,
    ]
    
    population = []
    scores = []
    
    # Generate initial population from each generator with different seeds
    for gen in generators:
        for seed in range(5):
            if time.time() > deadline:
                break
            pts = gen(seed)
            if _check_no_four_collinear(pts):
                score = _max_independent_set(pts)
                population.append(pts)
                scores.append(score)
    
    # Also add some random configurations
    for _ in range(5):
        if time.time() > deadline:
            break
        pts = []
        for _ in range(20):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.append([x, y])
        # Ensure distinctness
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        unique_pts = unique_pts[:20]
        if _check_no_four_collinear(unique_pts):
            score = _max_independent_set(unique_pts)
            population.append(unique_pts)
            scores.append(score)
    
    if not population:
        # Fallback: just return a random valid set
        pts = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    pts.append([x, y])
                    break
        return pts
    
    # Find best initial
    best_idx = scores.index(min(scores))
    best_points = population[best_idx][:]
    best_score = scores[best_idx]
    
    # Genetic algorithm
    pop_size = min(len(population), 20)
    population = population[:pop_size]
    scores = scores[:pop_size]
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep top 2
        new_population = [population[0][:], population[1][:]] if len(population) >= 2 else [population[0][:]]
        new_scores = scores[:len(new_population)]
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Crossover
            p1_idx = random.randint(0, min(5, len(population)-1))
            p2_idx = random.randint(0, min(5, len(population)-1))
            child = _crossover(population[p1_idx], population[p2_idx])
            
            # Mutation
            child = _mutate(child, rate=0.4)
            
            # Check validity
            if _check_no_four_collinear(child):
                child_score = _max_independent_set(child)
                new_population.append(child)
                new_scores.append(child_score)
            
            if time.time() > deadline:
                break
        
        # If we ran out of time during generation, update best and return
        if new_scores:
            min_idx = new_scores.index(min(new_scores))
            if new_scores[min_idx] < best_score:
                best_score = new_scores[min_idx]
                best_points = new_population[min_idx][:]
        
        population = new_population
        scores = new_scores
        
        # If population is too small, restart with new randoms
        if len(population) < 3:
            for _ in range(5):
                if time.time() > deadline:
                    break
                pts = _generate_parabola_variant(random.randint(0, 1000))
                if _check_no_four_collinear(pts):
                    score = _max_independent_set(pts)
                    population.append(pts)
                    scores.append(score)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(f"Points: {pts}")
