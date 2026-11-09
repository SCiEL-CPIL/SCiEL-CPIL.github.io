import time
import numpy as np
from collections import Counter, defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def compute_areas_counter(pts):
        """Compute full area counter for all non-zero doubled areas."""
        n = len(pts)
        counter = Counter()
        for i in range(n):
            ax, ay = pts[i]
            for j in range(i + 1, n):
                bx, by = pts[j]
                dx1, dy1 = bx - ax, by - ay
                for k in range(j + 1, n):
                    cx, cy = pts[k]
                    dx2, dy2 = cx - ax, cy - ay
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counter[det] += 1
        return counter

    def max_counter(counter):
        if not counter:
            return 0
        return max(counter.values())

    def update_move(pts, idx, old_pt, new_pt, counter):
        """Incrementally update counter after moving point idx.
        Returns new counter and new max."""
        n = len(pts)
        c = Counter(counter)
        ox, oy = old_pt
        nx, ny = new_pt
        others = [i for i in range(n) if i != idx]
        # Remove old contributions
        for ii in range(len(others)):
            i = others[ii]
            ax, ay = pts[i]
            for jj in range(ii + 1, len(others)):
                j = others[jj]
                bx, by = pts[j]
                det = abs((ax - ox) * (by - oy) - (ay - oy) * (bx - ox))
                if det > 0:
                    c[det] -= 1
                    if c[det] <= 0:
                        del c[det]
        # Add new contributions
        for ii in range(len(others)):
            i = others[ii]
            ax, ay = pts[i]
            for jj in range(ii + 1, len(others)):
                j = others[jj]
                bx, by = pts[j]
                det = abs((ax - nx) * (by - ny) - (ay - ny) * (bx - nx))
                if det > 0:
                    c[det] += 1
        mx = max(c.values(), default=0)
        return c, mx

    def update_swap(pts, i, j, counter):
        """Incrementally update counter after swapping points i and j."""
        n = len(pts)
        c = Counter(counter)
        pi = pts[i]
        pj = pts[j]
        others = [k for k in range(n) if k != i and k != j]
        # Remove old
        for ii in range(len(others)):
            a = others[ii]
            ax, ay = pts[a]
            for jj in range(ii + 1, len(others)):
                b = others[jj]
                bx, by = pts[b]
                det_i = abs((ax - pi[0]) * (by - pi[1]) - (ay - pi[1]) * (bx - pi[0]))
                if det_i > 0:
                    c[det_i] -= 1
                    if c[det_i] <= 0:
                        del c[det_i]
                det_j = abs((ax - pj[0]) * (by - pj[1]) - (ay - pj[1]) * (bx - pj[0]))
                if det_j > 0:
                    c[det_j] -= 1
                    if c[det_j] <= 0:
                        del c[det_j]
        # Add new (i now at pj, j now at pi)
        for ii in range(len(others)):
            a = others[ii]
            ax, ay = pts[a]
            for jj in range(ii + 1, len(others)):
                b = others[jj]
                bx, by = pts[b]
                det_i_new = abs((ax - pj[0]) * (by - pj[1]) - (ay - pj[1]) * (bx - pj[0]))
                if det_i_new > 0:
                    c[det_i_new] += 1
                det_j_new = abs((ax - pi[0]) * (by - pi[1]) - (ay - pi[1]) * (bx - pi[0]))
                if det_j_new > 0:
                    c[det_j_new] += 1
        mx = max(c.values(), default=0)
        return c, mx

    # 1. Initialize with strong structured candidates
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]

    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))

    cand5 = [(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + \
            [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)]

    cand6 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)]

    cand7 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
            [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
            [(x, 4) for x in range(6)]

    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7]

    best_points = None
    best_score = 0
    best_counter = None

    for cand in candidates:
        counter = compute_areas_counter(cand)
        s = max_counter(counter)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_counter = counter

    # 2. Local search with incremental updates + simulated annealing
    current_points = list(best_points)
    current_counter = Counter(best_counter)
    current_score = best_score

    rng = random.Random(42)
    no_improve_count = 0
    step_size = 3
    temperature = 5.0
    cool_rate = 0.9995

    while elapsed() < DEADLINE:
        # Occasionally adjust temperature (simulated annealing)
        temperature = max(0.5, temperature * cool_rate)

        r = rng.random()
        if r < 0.7:
            # Single point move
            idx = rng.randint(0, 29)
            old_x, old_y = current_points[idx]

            if rng.random() < 0.02:
                new_x = rng.randint(-30, 30)
                new_y = rng.randint(-30, 30)
            elif rng.random() < 0.15:
                new_x = old_x + rng.randint(-6, 6)
                new_y = old_y + rng.randint(-6, 6)
            else:
                new_x = old_x + rng.randint(-step_size, step_size)
                new_y = old_y + rng.randint(-step_size, step_size)

            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue

            # Check distinctness
            is_valid = True
            for j in range(30):
                if j == idx and current_points[j] == new_pt:
                    continue
                if current_points[j] == new_pt:
                    is_valid = False
                    break
            if not is_valid:
                continue

            # Try the move
            current_points[idx] = new_pt
            new_counter, new_score = update_move(current_points, idx, (old_x, old_y), new_pt, current_counter)

            if new_score > current_score:
                current_counter = new_counter
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
                    best_counter = Counter(current_counter)
                    if step_size > 1 and rng.random() < 0.3:
                        step_size = max(1, step_size - 1)
            elif new_score == current_score and rng.random() < 0.1:
                # Accept equal moves occasionally for exploration
                current_counter = new_counter
                no_improve_count += 1
            else:
                # Accept worse moves per simulated annealing
                if rng.random() < np.exp(-(current_score - new_score) / temperature):
                    current_counter = new_counter
                    current_score = new_score
                    no_improve_count += 1
                else:
                    current_points[idx] = (old_x, old_y)
                    no_improve_count += 1
        else:
            # Two-point swap
            i = rng.randint(0, 29)
            j = rng.randint(0, 29)
            if i == j:
                continue
            pi = current_points[i]
            pj = current_points[j]
            if pi == pj:
                continue

            # Swap
            current_points[i], current_points[j] = pj, pi
            new_counter, new_score = update_swap(current_points, i, j, current_counter)

            if new_score > current_score:
                current_counter = new_counter
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
                    best_counter = Counter(current_counter)
            elif new_score == current_score and rng.random() < 0.1:
                current_counter = new_counter
                no_improve_count += 1
            else:
                if rng.random() < np.exp(-(current_score - new_score) / temperature):
                    current_counter = new_counter
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Undo swap
                    current_points[i], current_points[j] = pi, pj
                    no_improve_count += 1

        # If stuck for a long time, restart from best
        if no_improve_count > 1500:
            current_points = list(best_points)
            current_counter = Counter(best_counter)
            current_score = best_score
            no_improve_count = 0
            step_size = rng.randint(4, 10)
            temperature = max(temperature, 8.0)

        # Occasionally adjust step size
        if rng.random() < 0.003:
            step_size = rng.randint(1, 6)

    return best_points
