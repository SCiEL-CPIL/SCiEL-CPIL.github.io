import random
import time
import math
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin for safety

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def normalize_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(abs(A), abs(B))
        if g != 0:
            g = gcd(g, abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def get_line_counts(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = normalize_line(points[i], points[j])
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        return line_counts

    def has_4_collinear(points):
        line_counts = get_line_counts(points)
        for pts_on_line in line_counts.values():
            if len(pts_on_line) >= 4:
                return True
        return False

    def count_triples(points):
        line_counts = get_line_counts(points)
        triples = 0
        for pts_on_line in line_counts.values():
            if len(pts_on_line) == 3:
                triples += 1
        return triples

    def get_triples_list(points):
        line_counts = get_line_counts(points)
        triples = []
        for pts_on_line in line_counts.values():
            if len(pts_on_line) == 3:
                pts = sorted(pts_on_line)
                triples.append((pts[0], pts[1], pts[2]))
        return triples

    def exact_independence_number(points):
        """
        Exact branch and bound for maximum subset with no 3 collinear.
        """
        n = len(points)
        triples = get_triples_list(points)
        
        # conflict[i]: set of frozensets {j, k} such that {i, j, k} is a triple
        conflict = [set() for _ in range(n)]
        for i, j, k in triples:
            conflict[i].add(frozenset((j, k)))
            conflict[j].add(frozenset((i, k)))
            conflict[k].add(frozenset((i, j)))
        
        # Order by degree (number of conflicts) descending to prune earlier
        degree = [len(conflict[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degree[i])
        
        # Remap conflicts to new order
        new_conflict = [set() for _ in range(n)]
        for new_idx, old_idx in enumerate(order):
            for fs in conflict[old_idx]:
                new_fs = frozenset(order.index(x) for x in fs)
                new_conflict[new_idx].add(new_fs)
        
        best = 0
        
        def bb(idx, current_mask, current_sum):
            nonlocal best
            if current_sum + (n - idx) <= best:
                return
            
            if idx == n:
                if current_sum > best:
                    best = current_sum
                return
            
            # Try including point idx
            valid = True
            for j in range(idx):
                if (current_mask & (1 << j)):
                    for k in range(j + 1, idx):
                        if (current_mask & (1 << k)):
                            if frozenset((j, k)) in new_conflict[idx]:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                bb(idx + 1, current_mask | (1 << idx), current_sum + 1)
            
            # Try excluding point idx
            bb(idx + 1, current_mask, current_sum)
        
        bb(0, 0, 0)
        return best

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_config():
        """Generate a random valid configuration (no 4 collinear)."""
        while True:
            points = [random_point() for _ in range(20)]
            if len(set(map(tuple, points))) < 20:
                continue
            if not has_4_collinear(points):
                return points

    def generate_structured_config():
        """
        Generate a structured configuration that encourages collinear triples.
        Strategy: place points on a few lines, but ensure no 4 are on the same line.
        """
        # Use a grid-like structure with some perturbation
        lines = []
        num_lines = random.randint(5, 8)
        for _ in range(num_lines):
            if random.random() < 0.3:
                y = random.randint(-90, 90)
                x1 = random.randint(-90, -60)
                x2 = random.randint(60, 90)
                line_points = [(x, y) for x in range(x1, x2, random.randint(15, 25))]
            elif random.random() < 0.6:
                x = random.randint(-90, 90)
                y1 = random.randint(-90, -60)
                y2 = random.randint(60, 90)
                line_points = [(x, y) for y in range(y1, y2, random.randint(15, 25))]
            else:
                m = random.choice([1, -1, 2, -2, 0.5, -0.5])
                b = random.randint(-50, 50)
                line_points = []
                for x in range(-90, 91, random.randint(15, 25)):
                    y = int(m * x + b)
                    if abs(y) <= 90:
                        line_points.append((x, y))
            lines.append(line_points)
        
        candidates = []
        for line in lines:
            candidates.extend(line)
        
        for _ in range(random.randint(5, 10)):
            candidates.append(tuple(random_point()))
        
        random.shuffle(candidates)
        points = []
        seen = set()
        for c in candidates:
            if c not in seen:
                points.append(list(c))
                seen.add(c)
            if len(points) == 20:
                break
        
        while len(points) < 20:
            p = random_point()
            if tuple(p) not in seen:
                points.append(p)
                seen.add(tuple(p))
        
        if has_4_collinear(points):
            for i in range(20):
                if not has_4_collinear(points):
                    break
                points[i] = random_point()
            if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
                return None
        return points

    best_points = None
    best_score = 21
    best_triple_count = 0

    def evaluate_and_update(points):
        nonlocal best_points, best_score, best_triple_count
        triples = count_triples(points)
        score = exact_independence_number(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            best_triple_count = triples
        elif score == best_score and triples > best_triple_count:
            best_triple_count = triples
        return score, triples

    # Phase 1: Generate diverse starting points
    initial_configs = []
    
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        pts = generate_valid_config()
        initial_configs.append(pts)
    
    for _ in range(15):
        if time.time() > deadline - 30:
            break
        pts = generate_structured_config()
        if pts is not None:
            initial_configs.append(pts)
    
    for pts in initial_configs:
        if time.time() > deadline - 20:
            break
        score, triples = evaluate_and_update(pts)
    
    if best_points is None:
        best_points = generate_valid_config()
        best_score, best_triple_count = evaluate_and_update(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    current_triples = best_triple_count

    # Phase 2: Simulated Annealing
    # We combine the structural insight from the inspiration (SA with grid/structured moves)
    # with the refinement machinery of the current program (exact B&B evaluation).
    
    # Use a larger grid for SA moves to allow more exploration while maintaining structure
    # The inspiration used a small grid (31x31), but we can use the full range or a subset.
    # Let's use a grid of -50 to 50 for SA moves to keep points somewhat close to center
    # but allow enough variety.
    SA_GRID_MIN = -50
    SA_GRID_MAX = 50
    
    def sa_move(points, i):
        """Perform a random move on point i, constrained to grid."""
        new_x = random.randint(SA_GRID_MIN, SA_GRID_MAX)
        new_y = random.randint(SA_GRID_MIN, SA_GRID_MAX)
        return [new_x, new_y]

    T = 10.0
    alpha = 0.95
    min_T = 0.01
    
    # SA loop
    while T > min_T and time.time() < deadline - 5:
        # Pick a random point to move
        i = random.randint(0, 19)
        orig_point = current_points[i]
        new_point = sa_move(current_points, i)
        
        if new_point == orig_point:
            continue
        
        current_points[i] = new_point
        
        # Check validity
        if len(set(map(tuple, current_points))) < 20 or has_4_collinear(current_points):
            current_points[i] = orig_point
            T *= alpha
            continue
        
        new_triples = count_triples(current_points)
        
        # Heuristic: use triples as a proxy for improvement
        # More triples -> likely lower independence number
        delta_triples = new_triples - current_triples
        
        if delta_triples > 0:
            # Likely improvement, verify with exact B&B
            new_score = exact_independence_number(current_points)
            if new_score < current_score:
                current_score = new_score
                current_triples = new_triples
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    best_triple_count = current_triples
            elif new_score == current_score:
                # Accept if same score (explore)
                current_triples = new_triples
            else:
                # Reject
                current_points[i] = orig_point
        elif delta_triples == 0:
            # Sideways move, accept with some probability
            if random.random() < 0.3:
                current_triples = new_triples
            else:
                current_points[i] = orig_point
        else:
            # Worse triples, accept with probability exp(-delta/T)
            # Note: delta_triples is negative, so -delta/T is positive
            prob = math.exp(-delta_triples / T)
            if random.random() < prob:
                # Verify with exact B&B to be safe
                new_score = exact_independence_number(current_points)
                if new_score <= current_score:
                    current_score = new_score
                    current_triples = new_triples
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        best_triple_count = current_triples
                else:
                    current_points[i] = orig_point
            else:
                current_points[i] = orig_point
        
        T *= alpha

    # Final local search refinement
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            for i in range(20):
                if time.time() > deadline - 5:
                    break
                
                orig_point = current_points[i]
                
                for _ in range(5):
                    new_point = random_point()
                    if new_point == orig_point:
                        continue
                    
                    current_points[i] = new_point
                    
                    if len(set(map(tuple, current_points))) < 20:
                        current_points[i] = orig_point
                        continue
                    
                    if has_4_collinear(current_points):
                        current_points[i] = orig_point
                        continue
                    
                    new_triples = count_triples(current_points)
                    
                    if new_triples > current_triples:
                        new_score = exact_independence_number(current_points)
                        if new_score < current_score:
                            current_score = new_score
                            current_triples = new_triples
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                                best_triple_count = current_triples
                            break
                        elif new_score == current_score:
                            current_triples = new_triples
                            improved = True
                            break
                        else:
                            current_points[i] = orig_point
                    elif new_triples == current_triples:
                        if random.random() < 0.1:
                            improved = True
                            break
                        else:
                            current_points[i] = orig_point
                    else:
                        current_points[i] = orig_point
                
                if improved:
                    break
        
        current_score = exact_independence_number(current_points)
        current_triples = count_triples(current_points)
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            best_triple_count = current_triples

    return best_points
