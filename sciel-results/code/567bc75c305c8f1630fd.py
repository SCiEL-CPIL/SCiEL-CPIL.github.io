import time
import random
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def line_key(p1, p2):
        """Canonical key for the line through p1 and p2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        dx //= g
        dy //= g
        # Normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line: dy*x - dx*y = dy*x1 - dx*y1
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        """Map line_key -> set of point indices on that line."""
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        """Check if no 4 points are collinear."""
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def is_valid_after_move(points, idx):
        """Check validity assuming only point idx moved. Only check lines through idx."""
        n = len(points)
        # For each other point, check the line through idx and that point
        # We need to count how many points are on each line through idx
        # Group by line direction
        line_counts = {}
        for i in range(n):
            if i == idx:
                continue
            key = line_key(points[idx], points[i])
            if key not in line_counts:
                line_counts[key] = 1
            line_counts[key] += 1
        # Also need to check lines NOT through idx that might now have 4 points
        # But moving one point can only create new collinearities involving that point
        # or break existing ones. Lines not through idx are unchanged.
        # So we only need to check lines through idx.
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def get_collinear_triples(points):
        """Returns a list of tuples (i, j, k) representing collinear triples."""
        n = len(points)
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def heuristic_score(points):
        """
        Heuristic: Number of collinear triples.
        Higher is better for minimizing the max no-3-collinear subset.
        """
        return len(get_collinear_triples(points))

    def calculate_exact_score(points):
        """
        Calculate the size of the largest subset with no three collinear.
        Uses branch and bound on the 3-uniform hypergraph of collinear triples.
        """
        n = len(points)
        triples = get_collinear_triples(points)
        
        if not triples:
            return n

        conflict_mask = [0] * n
        for i, j, k in triples:
            conflict_mask[i] |= (1 << j) | (1 << k)
            conflict_mask[j] |= (1 << i) | (1 << k)
            conflict_mask[k] |= (1 << i) | (1 << j)

        best = [0]
        
        def greedy_upper_bound(mask):
            """Greedy coloring bound on the conflict graph restricted to mask."""
            # Greedy independent set: repeatedly pick vertex with max degree, remove it and its neighbors
            remaining = mask
            count = 0
            while remaining:
                # Find vertex with max conflict-degree within remaining
                best_v = -1
                best_deg = -1
                tmp = remaining
                while tmp:
                    bit = tmp & -tmp
                    v = bit.bit_length() - 1
                    deg = (conflict_mask[v] & remaining).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    tmp ^= bit
                if best_v < 0:
                    break
                count += 1
                # Remove best_v and all its conflicts
                remaining &= ~(1 << best_v)
                remaining &= ~conflict_mask[best_v]
            return count

        def backtrack(available_mask, current_size):
            # Pruning: even if we take all remaining, can we beat best?
            if current_size + available_mask.bit_count() <= best[0]:
                return
            
            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Tighter bound: greedy independent set size
            ub = greedy_upper_bound(available_mask)
            if current_size + ub <= best[0]:
                return
            
            # Pick vertex with max conflict-degree within available set
            best_v = -1
            best_deg = -1
            tmp = available_mask
            while tmp:
                bit = tmp & -tmp
                v = bit.bit_length() - 1
                deg = (conflict_mask[v] & available_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                tmp ^= bit
            
            v_bit = 1 << best_v
            
            # Option 1: Include v (explore first if it has high degree - likely to prune)
            new_available = available_mask ^ v_bit
            new_available &= ~conflict_mask[best_v]
            backtrack(new_available, current_size + 1)
            
            # Option 2: Exclude v
            backtrack(available_mask ^ v_bit, current_size)
            
        backtrack((1 << n) - 1, 0)
        return best[0]

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    # Initial Configuration
    best_config = None
    best_score = 20
    best_heuristic = -1
    
    # Try to find a valid starting point
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        pts = generate_random_points()
        if is_valid(pts):
            best_config = pts
            break
            
    if best_config is None:
        # Fallback: Parabola (Score 20)
        return [[x, x * x] for x in range(-10, 10)]

    # Local Search
    current_pts = [p[:] for p in best_config]
    current_heuristic = heuristic_score(current_pts)
    
    # Simulated Annealing / Random Walk
    temp = 1.0
    cooling = 0.995
    step_count = 0
    
    while time.time() < deadline - 5:
        step_count += 1
        # Pick a point to move
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]
        
        # New random position
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)
        
        if new_x == old_x and new_y == old_y:
            continue
            
        # Check distinctness
        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue
            
        # Apply move
        current_pts[idx] = [new_x, new_y]
        
        # Check validity (only lines through moved point)
        if not is_valid_after_move(current_pts, idx):
            current_pts[idx] = [old_x, old_y]
            continue
            
        # Calculate new heuristic
        new_heuristic = heuristic_score(current_pts)
        
        # Acceptance criteria
        if new_heuristic > current_heuristic:
            current_heuristic = new_heuristic
        else:
            delta = new_heuristic - current_heuristic
            if random.random() < pow(2.718, delta / temp):
                current_heuristic = new_heuristic
            else:
                current_pts[idx] = [old_x, old_y]
                continue
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
        # Check exact score when heuristic improves or periodically
        if current_heuristic > best_heuristic:
            best_heuristic = current_heuristic
            exact = calculate_exact_score(current_pts)
            if exact < best_score:
                best_score = exact
                best_config = [p[:] for p in current_pts]
        elif step_count % 500 == 0:
            exact = calculate_exact_score(current_pts)
            if exact < best_score:
                best_score = exact
                best_config = [p[:] for p in current_pts]
        
    # Final check on best_config
    if best_config:
        final_score = calculate_exact_score(best_config)
        if final_score < best_score:
            best_score = final_score
            
    return best_config

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
