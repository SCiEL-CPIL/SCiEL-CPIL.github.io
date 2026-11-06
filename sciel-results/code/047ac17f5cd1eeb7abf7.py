import time
import random

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def max_no_3_collinear_exact(points):
        n = len(points)
        if n <= 2:
            return n

        triples = get_collinear_triples(points)
        if not triples:
            return n

        conflict_pairs = [[] for _ in range(n)]
        for u, v, w in triples:
            conflict_pairs[u].append((v, w))
            conflict_pairs[v].append((u, w))
            conflict_pairs[w].append((u, v))

        best_size = 0
        all_mask = (1 << n) - 1

        def solve(available_mask, current_size, current_mask):
            nonlocal best_size

            remaining = bin(available_mask).count('1')
            if current_size + remaining <= best_size:
                return

            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return

            v = (available_mask & -available_mask).bit_length() - 1
            v_bit = 1 << v

            # Exclude v
            solve(available_mask ^ v_bit, current_size, current_mask)

            # Include v
            valid = True
            for u, w in conflict_pairs[v]:
                if (current_mask & (1 << u)) and (current_mask & (1 << w)):
                    valid = False
                    break

            if valid:
                solve(available_mask ^ v_bit, current_size + 1, current_mask | v_bit)

        solve(all_mask, 0, 0)
        return best_size

    def score_config(points):
        return max_no_3_collinear_exact(points)

    # Initial configuration: 5x4 grid
    # Scale by 10 to allow perturbations
    base_points = []
    for i in range(5):
        for j in range(4):
            base_points.append([10 * i, 10 * j])
    
    # Random initial perturbations
    def random_perturbation(points, max_d=2):
        pts = [p[:] for p in points]
        for i in range(len(pts)):
            dx = random.randint(-max_d, max_d)
            dy = random.randint(-max_d, max_d)
            pts[i][0] += dx
            pts[i][1] += dy
        return pts

    best_score = 20
    best_points = base_points[:]

    # Local Search
    def local_search(points, max_time=5):
        current_points = [p[:] for p in points]
        current_score = score_config(current_points)
        start_ls = time.time()
        improved = True
        
        while improved and (time.time() - start_ls) < max_time and (time.time() - start_time) < timeout - 10:
            improved = False
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if (time.time() - start_time) > timeout - 10:
                    return current_points, current_score
                
                old_x, old_y = current_points[i]
                
                # Try small moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                            continue
                        
                        # Check collision
                        collision = False
                        for j, p in enumerate(current_points):
                            if j != i and p[0] == new_x and p[1] == new_y:
                                collision = True
                                break
                        if collision:
                            continue
                        
                        current_points[i] = [new_x, new_y]
                        
                        if has_4_collinear(current_points):
                            current_points[i] = [old_x, old_y]
                            continue
                        
                        new_score = score_config(current_points)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            current_points[i] = [old_x, old_y]
                            
                if improved:
                    break # Restart search after improvement
                    
        return current_points, current_score

    # Run multiple restarts
    for restart in range(50):
        if time.time() - start_time > timeout - 20:
            break
            
        random.seed(restart)
        pts = random_perturbation(base_points, max_d=3)
        
        if has_4_collinear(pts):
            # Try to fix 4-collinearities
            fixed = False
            for _ in range(100):
                if not has_4_collinear(pts):
                    fixed = True
                    break
                # Move a random point
                idx = random.randint(0, 19)
                old_x, old_y = pts[idx]
                for _ in range(10):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    nx, ny = old_x + dx, old_y + dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        # Check collision
                        coll = False
                        for j, p in enumerate(pts):
                            if j != idx and p[0] == nx and p[1] == ny:
                                coll = True
                                break
                        if not coll:
                            pts[idx] = [nx, ny]
                            if not has_4_collinear(pts):
                                fixed = True
                                break
            if not fixed:
                continue
                
        s = score_config(pts)
        if s < best_score:
            best_score = s
            best_points = pts[:]
            
        pts, new_s = local_search(pts, max_time=2)
        if new_s < best_score:
            best_score = new_s
            best_points = pts

    return best_points
