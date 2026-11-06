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
        
        # Remap: new_index -> old_index
        remap = [0] * n
        for new_idx, old_idx in enumerate(order):
            remap[new_idx] = old_idx
        
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

    def generate_valid_config():
        """Generate a random valid configuration (no 4 collinear)."""
        while True:
            points = [random_point() for _ in range(20)]
            if len(set(map(tuple, points))) < 20:
                continue
            if not has_4_collinear(points):
                return points

    def generate_grid_config():
        """
        Generate a 4x5 grid with random offsets to break 4-collinearities.
        This creates high triple density (many near-collinear triples).
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
            
        # If invalid, fall back to random
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return generate_valid_config()
            
        return points

    def generate_line_cluster_config():
        """
        Generate a configuration with points clustered near a few lines.
        Each line has 3-4 points, creating many collinear triples.
        """
        points = []
        seen = set()
        
        # Create 5-7 lines with 3-4 points each
        num_lines = random.randint(5, 7)
        for _ in range(num_lines):
            # Random line
            if random.random() < 0.3:
                # Horizontal
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
                # Vertical
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
                # Diagonal
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
        
        # Fill with random points if needed
        while len(points) < 20:
            p = random_point()
            if (p[0], p[1]) not in seen:
                points.append(p)
                seen.add((p[0], p[1]))
        
        points = points[:20]
        
        # If invalid, fall back to random
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return generate_valid_config()
            
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
    
    # Grid configs (high triple density)
    for _ in range(10):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_grid_config()
        initial_configs.append(pts)
    
    # Line cluster configs
    for _ in range(10):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_line_cluster_config()
        initial_configs.append(pts)
    
    # Random configs
    for _ in range(10):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_valid_config()
        initial_configs.append(pts)
    
    # Evaluate all initial configs
    for pts in initial_configs:
        if time.time() - start_time > timeout - 20:
            break
        score, triples = evaluate_and_update(pts)
    
    if best_points is None:
        best_points = generate_valid_config()
        best_score, best_triple_count = evaluate_and_update(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    current_triples = best_triple_count
    
    # Phase 2: Local search with hill climbing
    for restart in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        # If not first restart, perturb
        if restart > 0:
            # Pick 1-3 random points and move them
            num_moves = random.randint(1, 3)
            for _ in range(num_moves):
                idx = random.randint(0, 19)
                current_points[idx] = random_point()
            
            # Validate
            attempts = 0
            while (has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20) and attempts < 50:
                idx = random.randint(0, 19)
                current_points[idx] = random_point()
                attempts += 1
            
            if has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                current_triples = best_triple_count
                continue
            
            current_score = exact_independence_number(current_points)
            current_triples = count_triples(current_points)
        
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            
            # Try moving each point
            for i in range(20):
                if time.time() - start_time > timeout - 5:
                    break
                
                orig_point = current_points[i]
                
                # Try several moves for this point
                for _ in range(8):
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
                                best_triple_count = new_triples
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
        
        # After local search, do exact evaluation of current point
        current_score = exact_independence_number(current_points)
        current_triples = count_triples(current_points)
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            best_triple_count = current_triples
        elif current_score == best_score and current_triples > best_triple_count:
            best_triple_count = current_triples
    
    # Final: return the best point found
    final_score = exact_independence_number(best_points)
    if final_score < best_score:
        best_score = final_score
    
    return best_points
