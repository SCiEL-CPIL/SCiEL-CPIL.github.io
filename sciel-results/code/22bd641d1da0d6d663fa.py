import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples

    def generate_grid_candidates():
        candidates = []
        # 4x5 grid
        for dx in [1, 2, 3]:
            for dy in [1, 2, 3]:
                pts = []
                for i in range(4):
                    for j in range(5):
                        x = i * dx
                        y = j * dy
                        # Center it
                        pts.append([x - 1.5*dx, y - 2.0*dy])
                # Scale to integers if needed, or just use integers
                # Let's stick to integers
                pts = [[i*dx, j*dy] for i in range(4) for j in range(5)]
                # Shift to center
                min_x = min(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                pts = [[p[0] - min_x + 1, p[1] - min_y + 1] for p in pts]
                candidates.append(pts)
        
        # 5x4 grid
        for dx in [1, 2, 3]:
            for dy in [1, 2, 3]:
                pts = [[i*dx, j*dy] for i in range(5) for j in range(4)]
                min_x = min(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                pts = [[p[0] - min_x + 1, p[1] - min_y + 1] for p in pts]
                candidates.append(pts)
                
        # Perturbed grids
        rng = random.Random(42)
        base = [[i, j] for i in range(4) for j in range(5)]
        for _ in range(10):
            pts = [p[:] for p in base]
            for _ in range(5):
                idx = rng.randint(0, 19)
                pts[idx][0] += rng.randint(-2, 2)
                pts[idx][1] += rng.randint(-2, 2)
            # Check bounds and uniqueness
            valid = True
            seen = set()
            for p in pts:
                if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                    valid = False
                    break
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            if valid:
                candidates.append(pts)
                
        return candidates

    def greedy_set_cover_construction(seed_offset=0):
        rng = random.Random(seed_offset)
        points = []
        used_positions = set()
        
        all_positions = []
        for x in range(-100, 101):
            for y in range(-100, 101):
                all_positions.append((x, y))
        
        for step in range(20):
            if time.time() > deadline - 2:
                break
            
            line_counts = {}
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    key = line_key(points[i], points[j])
                    line_counts[key] = line_counts.get(key, 0) + 1
            
            best_gain = -1
            best_pos = None
            best_candidates = []
            
            sample_size = min(len(all_positions), 5000)
            if step < 5:
                candidates = rng.sample(all_positions, sample_size)
            else:
                near_candidates = set()
                for p in points:
                    for dx in range(-15, 16):
                        for dy in range(-15, 16):
                            np = (p[0] + dx, p[1] + dy)
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                near_candidates.add(np)
                candidates = list(near_candidates)
                if len(candidates) < sample_size:
                    candidates += rng.sample(all_positions, min(sample_size - len(candidates), 2000))
                else:
                    candidates = rng.sample(candidates, sample_size)
            
            for pos in candidates:
                if pos in used_positions:
                    continue
                
                new_triples = 0
                creates_four = False
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        key = line_key(points[i], points[j])
                        dx = points[j][0] - points[i][0]
                        dy = points[j][1] - points[i][1]
                        if (pos[0] - points[i][0]) * dy == (pos[1] - points[i][1]) * dx:
                            current_count = line_counts.get(key, 2)
                            if current_count >= 3:
                                creates_four = True
                                break
                            else:
                                new_triples += 1
                    if creates_four:
                        break
                
                if creates_four:
                    continue
                
                if new_triples > best_gain:
                    best_gain = new_triples
                    best_candidates = [pos]
                elif new_triples == best_gain:
                    best_candidates.append(pos)
            
            if not best_candidates:
                pos = rng.choice(all_positions)
                while pos in used_positions:
                    pos = rng.choice(all_positions)
                points.append(list(pos))
                used_positions.add(pos)
                continue
            
            pos = rng.choice(best_candidates)
            points.append(list(pos))
            used_positions.add(pos)
        
        _, has_four = count_triples_and_check_four(points)
        if has_four:
            return None
        
        return points

    def directed_moves(points):
        n = len(points)
        moves = []
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                for t in range(1, 5):
                    for sign in [1, -1]:
                        new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
                        new_p = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
        
        for _ in range(15):
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p not in points:
                moves.append(new_p)
        
        return moves

    def local_search(points, time_limit, seed=0):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        rng = random.Random(seed)
        
        while time.time() < time_limit:
            moves = directed_moves(current_pts)
            for _ in range(20):
                moves.append([rng.randint(-100, 100), rng.randint(-100, 100)])
            
            sample_size = min(len(moves), 300)
            if len(moves) > sample_size:
                sampled_moves = rng.sample(moves, sample_size)
            else:
                sampled_moves = moves
            
            best_move_score = current_score
            best_move_idx = -1
            best_move_pos = None
            
            for new_p in sampled_moves:
                if new_p in current_pts:
                    continue
                for i in range(20):
                    if current_pts[i] == new_p:
                        continue
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    new_score = get_score(test_pts)
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move_idx = i
                        best_move_pos = new_p
            
            if best_move_idx != -1:
                current_pts[best_move_idx] = best_move_pos
                current_score = best_move_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
            else:
                if rng.random() < 0.05:
                    idx = rng.randint(0, 19)
                    new_p = [rng.randint(-100, 100), rng.randint(-100, 100)]
                    if new_p not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        new_score = get_score(test_pts)
                        if new_score > -1000000:
                            current_pts = test_pts
                            current_score = new_score
        
        return best_pts

    # Generate initial candidates from grids
    grid_candidates = generate_grid_candidates()
    best_overall = None
    best_overall_score = -1
    
    for pts in grid_candidates:
        score = get_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = pts
            
    # Also run greedy construction
    seed = 0
    while time.time() < deadline - 5:
        pts = greedy_set_cover_construction(seed)
        if pts is None:
            seed += 1
            continue
        
        score = get_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = pts
        
        if time.time() < deadline - 2:
            search_time = min(2.0, deadline - 2 - time.time())
            if search_time > 0.5:
                refined = local_search(pts, time.time() + search_time, seed=seed)
                refined_score = get_score(refined)
                if refined_score > best_overall_score:
                    best_overall_score = refined_score
                    best_overall = refined
        
        seed += 1
        
    if best_overall is not None and time.time() < deadline - 1:
        final_refined = local_search(best_overall, time.time() + min(2, deadline - 1 - time.time()), seed=seed+1)
        final_score = get_score(final_refined)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_refined
            
    return best_overall
