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
        """Compute a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        # Canonical direction: ensure dx > 0, or dx == 0 and dy > 0
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line equation: dy*x - dx*y = c
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        """Count collinear triples and check for 4-collinear in O(n^2)."""
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
                # C(count, 3) triples
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples
    
    def greedy_set_cover_construction(seed_offset=0):
        """
        Greedy set-cover: iteratively place points to maximize collinear triples.
        At each step, pick the candidate position that creates the most new triples
        without creating any 4-collinear.
        """
        rng = random.Random(seed_offset)
        points = []
        used_positions = set()
        
        # Precompute all valid positions
        all_positions = []
        for x in range(-100, 101):
            for y in range(-100, 101):
                all_positions.append((x, y))
        
        for step in range(20):
            if time.time() > deadline - 2:
                break
            
            # Track line counts as we add points
            line_counts = {}
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    key = line_key(points[i], points[j])
                    line_counts[key] = line_counts.get(key, 0) + 1
            
            # For each candidate position, compute the gain in triples
            best_gain = -1
            best_pos = None
            best_candidates = []
            
            # Sample a subset of positions for speed
            sample_size = min(len(all_positions), 5000)
            if step < 5:
                # Early steps: more thorough search
                candidates = rng.sample(all_positions, sample_size)
            else:
                # Later steps: focus on positions near existing points
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
                
                # Compute new triples created by adding this point
                new_triples = 0
                creates_four = False
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        key = line_key(points[i], points[j])
                        # Check if pos is on this line
                        dx = points[j][0] - points[i][0]
                        dy = points[j][1] - points[i][1]
                        if (pos[0] - points[i][0]) * dy == (pos[1] - points[i][1]) * dx:
                            current_count = line_counts.get(key, 2)
                            if current_count >= 3:
                                creates_four = True
                                break
                            else:
                                # This creates one new triple
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
                # Fallback: pick a random position
                pos = rng.choice(all_positions)
                while pos in used_positions:
                    pos = rng.choice(all_positions)
                points.append(list(pos))
                used_positions.add(pos)
                continue
            
            # Pick from best candidates (with some randomness)
            pos = rng.choice(best_candidates)
            points.append(list(pos))
            used_positions.add(pos)
        
        # Verify no 4-collinear
        _, has_four = count_triples_and_check_four(points)
        if has_four:
            return None
        
        return points
    
    def directed_moves(points):
        """Generate directed moves that try to create new collinear triples."""
        n = len(points)
        moves = []
        
        # For each pair of points, try to find a third point that would be collinear
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
                
                # Try extending the line in both directions
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
        
        # Also add some random moves for diversity
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
            # Generate candidate moves
            moves = directed_moves(current_pts)
            # Add some random moves for diversity
            for _ in range(20):
                moves.append([rng.randint(-100, 100), rng.randint(-100, 100)])
            
            # Sample a subset of moves to keep evaluation time low and allow more iterations
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
                # No improvement found in sample
                # Try a random jump to escape local optimum occasionally
                if rng.random() < 0.05:
                    idx = rng.randint(0, 19)
                    new_p = [rng.randint(-100, 100), rng.randint(-100, 100)]
                    if new_p not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        new_score = get_score(test_pts)
                        if new_score > -1000000: # Valid
                            current_pts = test_pts
                            current_score = new_score
        
        return best_pts
    
    # Run multiple restarts until time limit
    best_overall = None
    best_overall_score = -1
    
    t_end = deadline
    seed = 0
    
    while time.time() < t_end - 5:
        pts = greedy_set_cover_construction(seed)
        if pts is None:
            seed += 1
            continue
        
        score = get_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = pts
        
        # Refine with local search
        if time.time() < t_end - 2:
            search_time = min(2.0, t_end - 2 - time.time())
            if search_time > 0.5:
                refined = local_search(pts, time.time() + search_time, seed=seed)
                refined_score = get_score(refined)
                if refined_score > best_overall_score:
                    best_overall_score = refined_score
                    best_overall = refined
        
        seed += 1
    
    # If we still have time, do more local search from best
    if best_overall is not None and time.time() < t_end - 1:
        final_refined = local_search(best_overall, time.time() + min(2, t_end - 1 - time.time()), seed=seed+1)
        final_score = get_score(final_refined)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_refined
    
    return best_overall
