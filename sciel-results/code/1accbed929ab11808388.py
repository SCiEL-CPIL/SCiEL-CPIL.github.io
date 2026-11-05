import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave margin for 120s limit

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1

        if A == 0 and B == 0:
            return None

        g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
        A //= g
        B //= g
        C //= g

        # Canonical sign
        if A < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0 and B < 0:
            B = -B
            C = -C

        return (A, B, C)

    def evaluate_full(points):
        """Full evaluation: return (num_4, num_5_plus, line_counts_dict)"""
        line_counts = {}
        for i in range(16):
            for j in range(i + 1, 16):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    line_counts[key] = line_counts.get(key, 0) + 1

        # Now count how many points are on each line
        num_4 = 0
        num_5_plus = 0
        for key in line_counts:
            A, B, C = key
            count = 0
            for p in points:
                if A * p[0] + B * p[1] == C:
                    count += 1
            if count == 4:
                num_4 += 1
            elif count >= 5:
                num_5_plus += 1

        return num_4, num_5_plus

    def incremental_eval(points, moved_idx, old_point, new_point):
        """
        Incremental evaluation after moving point moved_idx from old_point to new_point.
        Returns (num_4, num_5_plus) by recomputing only affected lines.
        """
        # Lines that were affected: all lines through old_point or new_point
        # We need to rebuild the line count for lines involving the moved point.

        # Collect all lines through old_point (with other 15 points)
        affected_keys = set()
        for j in range(16):
            if j == moved_idx:
                continue
            key_old = get_line_key(old_point, points[j])
            if key_old is not None:
                affected_keys.add(key_old)
            key_new = get_line_key(new_point, points[j])
            if key_new is not None:
                affected_keys.add(key_new)

        # For each affected line, count how many of the 16 points lie on it
        # using the current points (with moved_idx already at new_point)
        num_4 = 0
        num_5_plus = 0

        # First, count all non-affected lines
        # Actually, it's easier to just recount everything for affected lines
        # and assume non-affected lines haven't changed.

        # Let's do a full recount but only for affected lines
        # We need the total num_4 and num_5_plus

        # Strategy: compute num_4 and num_5_plus from scratch but only
        # re-evaluate affected lines. For non-affected lines, their counts
        # don't change. But we don't have the previous counts stored.

        # Simpler: just do full evaluation. With 16 points, it's fast enough.
        return evaluate_full(points)

    def is_valid(points):
        for p in points:
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return False
        if len(set(points)) != 16:
            return False
        return True

    # Initial configuration: 4x4 grid (affine plane of order 4)
    initial_points = [(i, j) for i in range(4) for j in range(4)]

    best_points = initial_points[:]
    best_score, best_inv = evaluate_full(best_points)

    # Simulated annealing
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999

    current_points = initial_points[:]
    curr_score, curr_inv = evaluate_full(current_points)

    if curr_inv == 0 and curr_score > best_score:
        best_score = curr_score
        best_points = current_points[:]

    T = T_start
    total_moves = 0

    while T > T_end:
        if time.time() - start_time > time_limit:
            break

        # Choose a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]

        # Generate a random move with larger radius
        radius = random.choice([5, 10, 20, 50, 100])
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        new_x = old_x + dx
        new_y = old_y + dy

        # Check bounds
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            T *= cooling_rate
            continue

        # Check distinctness
        new_point = (new_x, new_y)
        if new_point in current_points:
            T *= cooling_rate
            continue

        # Apply move
        current_points[idx] = new_point
        new_score, new_inv = evaluate_full(current_points)
        total_moves += 1

        # Compute energy difference
        # Lower is better: we want to minimize (inv, -score)
        # Energy = inv * 1000 - score (penalize invalidity heavily)
        old_energy = curr_inv * 1000 - curr_score
        new_energy = new_inv * 1000 - new_score
        delta_e = new_energy - old_energy

        accept = False
        if delta_e <= 0:
            accept = True
        else:
            # Accept with probability exp(-delta_e / T)
            prob = math.exp(-delta_e / T)
            if random.random() < prob:
                accept = True

        if accept:
            curr_score = new_score
            curr_inv = new_inv
            if curr_inv == 0 and curr_score > best_score:
                best_score = curr_score
                best_points = current_points[:]
        else:
            # Revert
            current_points[idx] = (old_x, old_y)

        T *= cooling_rate

    # Final: try some local search from best
    current_points = best_points[:]
    curr_score, curr_inv = evaluate_full(current_points)

    for step in range(50000):
        if time.time() - start_time > time_limit:
            break

        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]

        # Try small moves
        improved = False
        for dx in range(-20, 21):
            if dx == 0:
                continue
            for dy in range(-20, 21):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                new_point = (new_x, new_y)
                if new_point in current_points:
                    continue

                current_points[idx] = new_point
                new_score, new_inv = evaluate_full(current_points)

                if new_inv == 0 and new_score > curr_score:
                    curr_score = new_score
                    improved = True
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = current_points[:]
                    break
                else:
                    current_points[idx] = (old_x, old_y)

            if improved:
                break

    # Return best
    return [list(p) for p in best_points]
