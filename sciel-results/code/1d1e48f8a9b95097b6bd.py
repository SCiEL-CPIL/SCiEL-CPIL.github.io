import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        
        # Normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        # Line equation: A*x + B*y = C
        # A = dy, B = -dx
        A = dy
        B = -dx
        C = A * x1 + B * y1
        
        # Normalize A, B, C
        # A and B are coprime.
        # Ensure unique sign
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def score(points):
        """
        Returns the number of lines with exactly 4 points.
        Returns -1 if any line has 5+ points.
        """
        pair_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    pair_counts[key] += 1
        
        score_val = 0
        for count in pair_counts.values():
            # Solve k(k-1)/2 = count
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) / 2
                if k == int(k):
                    k = int(k)
                    if k == 4:
                        score_val += 1
                    elif k > 4:
                        return -1 # Invalid: 5+ collinear
        return score_val

    def valid_points(points):
        # Check distinctness and bounds
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(tuple(p))
            if p[0] < -1000 or p[0] > 1000 or p[1] < -1000 or p[1] > 1000:
                return False
        return True

    # Initial configuration: 4x4 grid
    initial_grid = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_grid]
    best_score = score(best_points)
    
    # Simulated Annealing with Restarts
    num_rounds = 20
    time_per_round = (deadline - time.time()) / num_rounds
    
    for r in range(num_rounds):
        if time.time() > deadline - 5:
            break
            
        # Choose initial config for this round
        if r == 0:
            current = [p[:] for p in initial_grid]
        else:
            # Perturb best found so far
            current = [p[:] for p in best_points]
            # Randomly move 2-4 points significantly to escape local optimum
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
            
        if not valid_points(current):
            current = [p[:] for p in initial_grid]
            
        current_score = score(current)
        if current_score == -1:
            current_score = -100
            
        best_in_round = current_score
        best_in_round_pts = [p[:] for p in current]
        
        T = 5.0
        T_min = 0.1
        alpha = 0.9995
        
        # Run SA until time slice ends
        while time.time() < start_time + (r + 1) * time_per_round:
            # Propose move
            i = random.randint(0, 15)
            x, y = current[i]
            
            # Mix of small and large moves
            if random.random() < 0.1:
                # Large jump
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
            else:
                # Small move
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
                new_x = x + dx
                new_y = y + dy
            
            # Check bounds
            if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                continue
                
            new_points = [p[:] for p in current]
            new_points[i] = [new_x, new_y]
            
            # Check distinctness
            valid = True
            for j in range(16):
                if j == i: continue
                if new_points[j] == new_points[i]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            new_score = score(new_points)
            
            if new_score == -1:
                # Invalid, reject
                continue
            
            if new_score > current_score:
                current = new_points
                current_score = new_score
                if current_score > best_in_round:
                    best_in_round = current_score
                    best_in_round_pts = [p[:] for p in current]
            else:
                delta = new_score - current_score
                if delta == 0:
                    if random.random() < 0.5:
                        current = new_points
                        current_score = new_score
                else:
                    # delta < 0
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current = new_points
                        current_score = new_score
                        
            T *= alpha
            if T < T_min:
                T = T_min
                
        # End of round
        if best_in_round > best_score:
            best_score = best_in_round
            best_points = [p[:] for p in best_in_round_pts]
            
    # Final polish: simple hill climbing on best_points
    if time.time() < deadline - 2:
        current = [p[:] for p in best_points]
        current_score = score(current)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(16):
                if time.time() > deadline - 2:
                    break
                x, y = current[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if nx < -1000 or nx > 1000 or ny < -1000 or ny > 1000:
                            continue
                        new_pts = [p[:] for p in current]
                        new_pts[i] = [nx, ny]
                        if not valid_points(new_pts):
                            continue
                        ns = score(new_pts)
                        if ns == -1:
                            continue
                        if ns > current_score:
                            current = new_pts
                            current_score = ns
                            improved = True
        if current_score > best_score:
            best_score = current_score
            best_points = current

    return best_points
