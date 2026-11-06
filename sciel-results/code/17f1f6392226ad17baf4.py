import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin before 150s limit
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def count_max_collinear(points):
        """Find the maximum number of collinear points in the set."""
        n = len(points)
        max_count = 2
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count > max_count:
                    max_count = count
        return max_count
    
    def has_4_collinear(points):
        """Check if any 4 points are collinear."""
        return count_max_collinear(points) >= 4
    
    def count_triples(points):
        """Count number of collinear triples."""
        n = len(points)
        triples = 0
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples += 1
        return triples
    
    def get_triples_list(points):
        """Get list of all collinear triples as (i, j, k)."""
        n = len(points)
        triples = []
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples.append((i, j, k))
        return triples
    
    def exact_independence_number(points):
        """
        Exact branch and bound for maximum subset with no 3 collinear.
        This is the score we want to minimize.
        """
        n = len(points)
        triples = get_triples_list(points)
        
        # pairs_with[i]: for each point i, list of (j, k) such that {i, j, k} is a triple
        pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_with[i].append((j, k))
            pairs_with[j].append((i, k))
            pairs_with[k].append((i, j))
        
        # Sort points by number of triples they participate in (descending)
        degree = [len(pairs_with[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degree[i])
        
        # Rebuild pairs_with in new ordering
        new_pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            ni = order.index(i)
            nj = order.index(j)
            nk = order.index(k)
            new_pairs_with[ni].append((nj, nk))
            new_pairs_with[nj].append((ni, nk))
            new_pairs_with[nk].append((ni, nj))
        
        best = 0
        
        def bb(idx, current_mask, current_sum):
            nonlocal best
            # Upper bound: current_sum + remaining points
            if current_sum + (n - idx) <= best:
                return
            
            if idx == n:
                if current_sum > best:
                    best = current_sum
                return
            
            # Try including point idx
            valid = True
            for j, k in new_pairs_with[idx]:
                if (current_mask & (1 << j)) and (current_mask & (1 << k)):
                    valid = False
                    break
            
            if valid:
                bb(idx + 1, current_mask | (1 << idx), current_sum + 1)
            
            # Try excluding point idx
            bb(idx + 1, current_mask, current_sum)
        
        bb(0, 0, 0)
        return best
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    def generate_parabola_config():
        """
        Generate points on a parabola y = x^2 + bx + c with perturbations.
        Points on a parabola have the property that no 3 are collinear,
        but we can create near-parabola structures with high triple density.
        """
        points = []
        seen = set()
        
        # Use a base parabola and add perturbations
        for x in range(-10, 10):
            base_y = x * x + random.randint(-5, 5)
            px = x + random.randint(-2, 2)
            py = base_y + random.randint(-3, 3)
            px = max(-100, min(100, px))
            py = max(-100, min(100, py))
            p = [px, py]
            if (p[0], p[1]) not in seen:
                points.append(p)
                seen.add((p[0], p[1]))
        
        # Fill if needed
        while len(points) < 20:
            p = random_point()
            if (p[0], p[1]) not in seen:
                points.append(p)
                seen.add((p[0], p[1]))
        
        points = points[:20]
        
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return None
        
        return points
    
    def generate_grid_config():
        """
        Generate a 4x5 grid with random offsets to break 4-collinearities.
        """
        points = []
        for x in range(5):
            for y in range(4):
                px = x * 12 + random.randint(-3, 3)
                py = y * 12 + random.randint(-3, 3)
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                points.append([px, py])
        
        # Remove duplicates and fill if necessary
        unique_points = []
        seen = set()
        for p in points:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_points.append(p)
        
        if len(unique_points) < 20:
            while len(unique_points) < 20:
                p = random_point()
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_points.append(p)
            points = unique_points
        else:
            points = unique_points[:20]
            
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return None
            
        return points
    
    def generate_line_cluster_config():
        """
        Generate a configuration with points clustered near a few lines.
        """
        points = []
        seen = set()
        
        num_lines = random.randint(5, 7)
        for _ in range(num_lines):
            if random.random() < 0.3:
                y = random.randint(-80, 80)
                x_base = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    x = x_base + i * random.randint(15, 25)
                    if -100 <= x <= 100:
                        p = [x, y]
                        if (p[0], p[1]) not in seen:
                            points.append(p)
                            seen.add((p[0], p[1]))
            elif random.random() < 0.6:
                x = random.randint(-80, 80)
                y_base = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    y = y_base + i * random.randint(15, 25)
                    if -100 <= y <= 100:
                        p = [x, y]
                        if (p[0], p[1]) not in seen:
                            points.append(p)
                            seen.add((p[0], p[1]))
            else:
                m = random.choice([1, -1, 2, -2])
                b = random.randint(-50, 50)
                x_base = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    x = x_base + i * random.randint(15, 25)
                    y = m * x + b
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        p = [x, y]
                        if (p[0], p[1]) not in seen:
                            points.append(p)
                            seen.add((p[0], p[1]))
        
        while len(points) < 20:
            p = random_point()
            if (p[0], p[1]) not in seen:
                points.append(p)
                seen.add((p[0], p[1]))
        
        points = points[:20]
        
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return None
            
        return points
    
    def generate_valid_config():
        """Generate a random valid configuration (no 4 collinear)."""
        while True:
            points = [random_point() for _ in range(20)]
            if len(set(map(tuple, points))) < 20:
                continue
            if not has_4_collinear(points):
                return points
    
    def evaluate_score(points):
        """Evaluate the score (independence number) of a configuration."""
        return exact_independence_number(points)
    
    def simulated_annealing(points, max_iterations=500):
        """
        Simulated annealing to minimize the independence number.
        Uses triple count as a proxy objective with occasional exact evaluation.
        """
        current_points = [p[:] for p in points]
        current_triples = count_triples(current_points)
        current_score = evaluate_score(current_points)
        
        best_points = [p[:] for p in current_points]
        best_score = current_score
        best_triples = current_triples
        
        T = 10.0
        cooling_rate = 0.995
        
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout - 10:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_point = current_points[idx][:]
            
            # Generate a new position: either small perturbation or random
            if random.random() < 0.7:
                # Small perturbation
                new_point = [
                    old_point[0] + random.randint(-5, 5),
                    old_point[1] + random.randint(-5, 5)
                ]
            else:
                # Random position
                new_point = random_point()
            
            # Clamp to range
            new_point[0] = max(-100, min(100, new_point[0]))
            new_point[1] = max(-100, min(100, new_point[1]))
            
            # Check duplicates
            if any((current_points[i][0] == new_point[0] and 
                   current_points[i][1] == new_point[1]) for i in range(20) if i != idx):
                continue
            
            current_points[idx] = new_point
            
            # Check 4-collinear constraint
            if has_4_collinear(current_points):
                current_points[idx] = old_point
                continue
            
            new_triples = count_triples(current_points)
            delta_triples = new_triples - current_triples
            
            # Accept with probability based on triple improvement
            if delta_triples > 0:
                # Accept if it improves triple count
                current_triples = new_triples
                current_score = evaluate_score(current_points)
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    best_triples = current_triples
            else:
                # Accept with probability exp(delta/T) where delta is negative
                if T > 0.01:
                    prob = min(1.0, max(0.0, 1.0 - abs(delta_triples) / (T * 10)))
                    if random.random() < prob:
                        current_triples = new_triples
                        # Don't update score every time for speed
                        if iteration % 50 == 0:
                            current_score = evaluate_score(current_points)
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                                best_triples = current_triples
                    else:
                        current_points[idx] = old_point
            
            T *= cooling_rate
        
        return best_points, best_score, best_triples
    
    def local_search(points, max_steps=100):
        """
        Local search: try moving each point to nearby positions to improve triple count.
        """
        current_points = [p[:] for p in points]
        current_triples = count_triples(current_points)
        current_score = evaluate_score(current_points)
        
        best_points = [p[:] for p in current_points]
        best_score = current_score
        best_triples = current_triples
        
        for step in range(max_steps):
            if time.time() - start_time > timeout - 10:
                break
            
            improved = False
            
            for i in range(20):
                if time.time() - start_time > timeout - 10:
                    break
                
                orig_point = current_points[i]
                
                # Try nearby positions
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_point = [orig_point[0] + dx, orig_point[1] + dy]
                        if not (-100 <= new_point[0] <= 100 and -100 <= new_point[1] <= 100):
                            continue
                        
                        # Check duplicates
                        if any((current_points[j][0] == new_point[0] and 
                               current_points[j][1] == new_point[1]) for j in range(20) if j != i):
                            continue
                        
                        current_points[i] = new_point
                        
                        if has_4_collinear(current_points):
                            current_points[i] = orig_point
                            continue
                        
                        new_triples = count_triples(current_points)
                        
                        if new_triples > current_triples:
                            current_triples = new_triples
                            current_score = evaluate_score(current_points)
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                                best_triples = current_triples
                            
                            improved = True
                            break
                        elif new_triples == current_triples:
                            # Accept neutral moves with some probability
                            if random.random() < 0.1:
                                improved = True
                                break
                            else:
                                current_points[i] = orig_point
                        else:
                            current_points[i] = orig_point
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_points, best_score, best_triples
    
    # Main algorithm
    best_points = None
    best_score = 21
    best_triples = 0
    
    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Grid configs
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        pts = generate_grid_config()
        if pts:
            initial_configs.append(pts)
    
    # Line cluster configs
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        pts = generate_line_cluster_config()
        if pts:
            initial_configs.append(pts)
    
    # Parabola configs
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        pts = generate_parabola_config()
        if pts:
            initial_configs.append(pts)
    
    # Random configs
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        pts = generate_valid_config()
        initial_configs.append(pts)
    
    # Evaluate initial configs
    for pts in initial_configs:
        if time.time() - start_time > timeout - 40:
            break
        score = evaluate_score(pts)
        triples = count_triples(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            best_triples = triples
    
    if best_points is None:
        best_points = generate_valid_config()
        best_score = evaluate_score(best_points)
        best_triples = count_triples(best_points)
    
    # Phase 2: Simulated annealing from best initial
    if time.time() - start_time < timeout - 30:
        sa_points, sa_score, sa_triples = simulated_annealing(best_points, max_iterations=300)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points
            best_triples = sa_triples
    
    # Phase 3: Local search refinement
    if time.time() - start_time < timeout - 10:
        ls_points, ls_score, ls_triples = local_search(best_points, max_steps=50)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
            best_triples = ls_triples
    
    # Phase 4: More simulated annealing from improved point
    if time.time() - start_time < timeout - 5:
        sa_points2, sa_score2, sa_triples2 = simulated_annealing(best_points, max_iterations=200)
        if sa_score2 < best_score:
            best_score = sa_score2
            best_points = sa_points2
            best_triples = sa_triples2
    
    # Final validation
    final_score = evaluate_score(best_points)
    if final_score < best_score:
        best_score = final_score
    
    return best_points
