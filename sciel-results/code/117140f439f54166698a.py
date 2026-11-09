import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        # Use a dictionary to count points on lines
        # Key: normalized line representation (A, B, C) for Ax + By + C = 0
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                # Line through p1, p2: (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = p2[0] * p1[1] - p1[0] * p2[1]
                
                # Normalize (A, B, C)
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
                if g == 0: g = 1
                A //= g
                B //= g
                C //= g
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
                
                # Early termination if any line has 4 points
                if len(line_counts[key]) >= 4:
                    return True
        
        # Final check for any line with >= 4
        for count in line_counts.values():
            if len(count) >= 4:
                return True
        return False

    def calculate_score(points):
        """
        Exact branch and bound for the largest subset with no 3 collinear points.
        """
        n = len(points)
        
        # Precompute collinearity conflicts
        # conflict[i][j] is a bitmask of k such that i, j, k are collinear
        conflict = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                conflict[i][j] = mask
                conflict[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        # Precompute degrees for heuristic ordering?
        # Let's use a simple recursive B&B with strong pruning.
        
        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Upper bound: current size + number of available points
            if current_size + bin(available).count('1') <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Heuristic: pick a point that is most "constrained" or just the first one.
            # To improve pruning, picking a point that conflicts with many others in 'available'
            # is often good. However, calculating dynamic degree is expensive.
            # Let's stick to the lowest index for determinism and speed, but ensure we explore both branches.
            
            # Optimization: If we pick i, we remove from 'available' all j that would form a triple with i and any k in selected_mask.
            # Actually, the standard approach for "no 3 collinear" (cap set in 2D integer lattice sense, but general position) is:
            # If we select i and j, we cannot select any k in conflict[i][j].
            # If we select i, we don't immediately restrict j unless there is a k in selected_mask such that i, j, k are collinear.
            # So, if we select i, for each j in available, if there exists k in selected_mask such that i, j, k are collinear, then j is forbidden.
            
            i_bit = available & -available
            i = i_bit.bit_length() - 1
            
            # Branch 1: Exclude i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Branch 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune new_available
            # A point j in new_available is forbidden if there exists k in selected_mask (which is now new_selected without i? No, selected_mask is the old one)
            # Wait, if we include i, the new selected set is selected_mask | {i}.
            # A point j is forbidden if it forms a collinear triple with i and some k in selected_mask.
            # OR if it forms a collinear triple with two points in selected_mask? No, that's already handled because we would have pruned it earlier when the second of those two was selected?
            # Let's trace:
            # If selected_mask has {a, b}, and we are considering adding c.
            # When we added b, we would have checked if b, a, c are collinear. If yes, c would have been removed from available.
            # So, when adding i, we only need to check conflicts between i and any j in available with respect to the *existing* selected_mask.
            # If i, j, k are collinear and k is in selected_mask, then j cannot be added.
            
            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= conflict[i][k]
                temp_sel ^= k_bit
            
            new_available &= ~forbidden
            
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    def generate_parabola_points():
        return [[x, x * x] for x in range(-10, 10)]

    best_points = None
    best_score = 21  # Max possible is 20

    # Strategy:
    # 1. Try to find a configuration with low score (many collinear triples) via random search + local search.
    # The score is the size of the largest "no 3 collinear" subset.
    # Minimizing this means maximizing the number of collinear triples such that they cover the points densely.
    
    # Initial random restarts
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        points = generate_random_points()
        if not has_four_collinear(points):
            score = calculate_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
    
    if best_points is None:
        best_points = generate_parabola_points()
        best_score = 20

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            old_point = best_points[i][:]
            
            # 1. Random moves
            for _ in range(10):
                new_point = [random.randint(-100, 100), random.randint(-100, 100)]
                if any(new_point == p for idx, p in enumerate(best_points) if idx != i):
                    continue
                
                test_points = best_points[:]
                test_points[i] = new_point
                
                if has_four_collinear(test_points):
                    continue
                    
                score = calculate_score(test_points)
                if score < best_score:
                    best_score = score
                    best_points = test_points
                    improved = True
                    break
                elif score == best_score and random.random() < 0.1:
                    best_points = test_points
                    improved = True
                    break
            
            # 2. Structured moves: place on a line defined by two other points
            other_indices = [idx for idx in range(20) if idx != i]
            random.shuffle(other_indices)
            for _ in range(10):
                if len(other_indices) < 2:
                    break
                j = other_indices[0]
                k = other_indices[1]
                
                p1 = best_points[j]
                p2 = best_points[k]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                found_move = False
                # Try small t values to stay within bounds and avoid 4-collinearity easily
                for t in range(-5, 6):
                    if t == 0 or t == 1:
                        continue
                        
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                        
                    new_point = [nx, ny]
                    
                    is_distinct = True
                    for idx_p, p in enumerate(best_points):
                        if idx_p != i and p == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    test_points = best_points[:]
                    test_points[i] = new_point
                    
                    if has_four_collinear(test_points):
                        continue
                        
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                    elif score == best_score and random.random() < 0.1:
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                        
                if found_move:
                    break

    return best_points
