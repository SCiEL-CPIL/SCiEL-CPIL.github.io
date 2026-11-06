import time
import random
import sys
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (0, 0, 0)
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
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
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        line_map = build_line_map(points)
        triples = set()
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.add((pts[i], pts[j], pts[k]))
        return list(triples)

    def compute_exact_score(points):
        """
        Branch and bound to find the maximum size of a subset with no 3 collinear.
        This is the Maximum Independent Set (MIS) of the 3-uniform hypergraph of collinear triples.
        """
        triples = get_collinear_triples(points)
        if not triples:
            return N

        # conflict_mask[i] has bits set for all j, k such that (i, j, k) is a triple.
        # If we include i, we must exclude all j, k in conflict_mask[i].
        conflict_mask = [0] * N
        for i, j, k in triples:
            conflict_mask[i] |= (1 << j) | (1 << k)
            conflict_mask[j] |= (1 << i) | (1 << k)
            conflict_mask[k] |= (1 << i) | (1 << j)

        best = [0]
        
        # Pre-compute bit counts for speed if possible, but N=20 is small enough for bin().count
        # However, we can optimize the heuristic selection.

        def backtrack(available_mask, current_size):
            # Pruning: even if we take all remaining, can we beat best?
            if current_size + bin(available_mask).count('1') <= best[0]:
                return

            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return

            # Heuristic: Pick a vertex v from available_mask.
            # We want to pick v such that it removes many vertices from available_mask.
            # This is equivalent to picking the vertex with the highest degree in the induced subgraph.
            # For 3-uniform hypergraph, "degree" is number of edges containing v.
            # Removing v and its neighbors (conflicts) reduces the set.
            
            best_v = -1
            max_removed = -1
            temp_mask = available_mask
            
            # Iterate through vertices in available_mask
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                
                # If we include v, we remove v and all its conflicts that are in available_mask
                removed = (conflict_mask[v] & available_mask) | v_bit
                cnt = bin(removed).count('1')
                
                if cnt > max_removed:
                    max_removed = cnt
                    best_v = v
                    best_v_bit = v_bit
                temp_mask ^= v_bit

            if best_v == -1:
                return

            v_bit = best_v_bit

            # Option 1: Exclude v
            backtrack(available_mask ^ v_bit, current_size)

            # Option 2: Include v
            # Remove v and all its conflicts
            new_available = available_mask ^ v_bit
            new_available &= ~conflict_mask[best_v]

            backtrack(new_available, current_size + 1)

        backtrack((1 << N) - 1, 0)
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

    def generate_structured_points():
        """
        Generate points with structure to encourage collinearities.
        """
        pts = []
        method = random.randint(0, 2)
        if method == 0:
            # Points on 2 or 3 lines + noise
            num_lines = random.randint(2, 4)
            lines = []
            for _ in range(num_lines):
                # Random line y = mx + c
                # Ensure slope is rational and manageable
                m_num = random.randint(-5, 5)
                m_den = random.randint(1, 5)
                c = random.randint(-50, 50)
                lines.append((m_num, m_den, c))

            attempts = 0
            while len(pts) < N and attempts < 1000:
                attempts += 1
                if random.random() < 0.8:
                    m_num, m_den, c = random.choice(lines)
                    x = random.randint(COORD_MIN, COORD_MAX)
                    # y = (m_num/m_den)*x + c => m_den*y = m_num*x + m_den*c
                    # We need integer y.
                    rhs = m_num * x + m_den * c
                    if rhs % m_den == 0:
                        y = rhs // m_den
                        if -100 <= y <= 100:
                            p = [x, y]
                            if p not in pts:
                                pts.append(p)
                else:
                    x = random.randint(COORD_MIN, COORD_MAX)
                    y = random.randint(COORD_MIN, COORD_MAX)
                    p = [x, y]
                    if p not in pts:
                        pts.append(p)
        elif method == 1:
            # Grid-like with perturbations
            step = random.randint(8, 15)
            start_x = random.randint(-90, 90 - step * 4)
            start_y = random.randint(-90, 90 - step * 4)
            for i in range(5):
                for j in range(4):
                    x = start_x + i * step + random.randint(-2, 2)
                    y = start_y + j * step + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        p = [x, y]
                        if p not in pts:
                            pts.append(p)
            while len(pts) < N:
                x = random.randint(COORD_MIN, COORD_MAX)
                y = random.randint(COORD_MIN, COORD_MAX)
                p = [x, y]
                if p not in pts:
                    pts.append(p)
        else:
            return generate_random_points()

        return pts[:N]

    best_config = None
    best_score = N + 1
    best_triple_count = -1

    # Initial search for a valid configuration
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        if random.random() < 0.3:
            pts = generate_random_points()
        else:
            pts = generate_structured_points()
        if is_valid(pts):
            triples = get_collinear_triples(pts)
            if len(triples) > best_triple_count:
                best_triple_count = len(triples)
                best_config = pts
                # Calculate score only for the best candidate to save time initially
                # But we need a baseline score
                if best_score == N + 1 or len(triples) > best_triple_count:
                     # Heuristic: more triples usually means lower score
                     # Let's just pick the one with most triples as initial best
                     pass
            
            # Actually, let's just find any valid one and start optimizing
            if best_config is None:
                best_config = pts
                break

    if best_config is None:
        # Fallback: Parabola
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = [p[:] for p in best_config]
    current_triples = get_collinear_triples(current_pts)
    current_score = compute_exact_score(current_pts)
    best_score = current_score
    best_config = [p[:] for p in current_pts]
    best_triple_count = len(current_triples)

    # Local Search
    # Objective: Minimize Score. Surrogate: Maximize Triple Count.
    # We use a hybrid acceptance rule.
    
    temp = 10.0
    cooling = 0.99
    no_improve_steps = 0
    max_no_improve = 500

    while time.time() < deadline - 5:
        step_count = 1
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]

        # Generate a new point
        if random.random() < 0.5:
            # Random new point
            new_x = random.randint(COORD_MIN, COORD_MAX)
            new_y = random.randint(COORD_MIN, COORD_MAX)
        else:
            # Perturb existing point
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if new_x < COORD_MIN: new_x = COORD_MIN
            if new_x > COORD_MAX: new_x = COORD_MAX
            if new_y < COORD_MIN: new_y = COORD_MIN
            if new_y > COORD_MAX: new_y = COORD_MAX

        if new_x == old_x and new_y == old_y:
            continue

        new_pt = [new_x, new_y]
        
        # Check distinctness
        is_dup = False
        for i in range(N):
            if i != idx and current_pts[i] == new_pt:
                is_dup = True
                break
        if is_dup:
            continue

        current_pts[idx] = new_pt

        if not is_valid(current_pts):
            current_pts[idx] = [old_x, old_y]
            continue

        new_triples = get_collinear_triples(current_pts)
        new_triple_count = len(new_triples)
        
        # Surrogate improvement
        if new_triple_count > best_triple_count:
            # Strong signal of potential improvement
            new_score = compute_exact_score(current_pts)
            if new_score < best_score:
                best_score = new_score
                best_config = [p[:] for p in current_pts]
                best_triple_count = new_triple_count
                no_improve_steps = 0
                current_score = new_score
                temp = max(0.1, temp * cooling)
                continue
            else:
                # Triple count improved but score didn't (or got worse)
                # Accept with higher probability
                if random.random() < 0.8:
                    current_score = new_score
                    temp = max(0.1, temp * cooling)
                    continue
                else:
                    current_pts[idx] = [old_x, old_y]
                    continue
        
        # If triple count is not improving, we need to be careful
        # Calculate score only if we might accept
        # To save time, we can estimate score change or just calculate if temp is high
        
        # For efficiency, we only calculate exact score if:
        # 1. We have accepted many steps without checking score
        # 2. The triple count is close to best
        # 3. Randomly
        
        check_score = False
        if new_triple_count >= best_triple_count - 2:
            check_score = True
        elif random.random() < 0.1:
            check_score = True

        if check_score:
            new_score = compute_exact_score(current_pts)
            delta = new_score - current_score
            
            if new_score < best_score:
                best_score = new_score
                best_config = [p[:] for p in current_pts]
                best_triple_count = new_triple_count
                no_improve_steps = 0
                current_score = new_score
                temp = max(0.1, temp * cooling)
            else:
                if delta < 0:
                    current_score = new_score
                    no_improve_steps += 1
                else:
                    # Accept deterioration with probability exp(-delta/temp)
                    if random.random() < pow(2.718, -delta / temp):
                        current_score = new_score
                        no_improve_steps += 1
                    else:
                        current_pts[idx] = [old_x, old_y]
                        continue
                temp = max(0.1, temp * cooling)
        else:
            # Accept based on triple count heuristic
            if new_triple_count >= current_triple_count:
                current_triple_count = new_triple_count
                no_improve_steps += 1
                temp = max(0.1, temp * cooling)
            else:
                current_pts[idx] = [old_x, old_y]
                continue

        if no_improve_steps > max_no_improve:
            # Restart
            for _ in range(20):
                if time.time() > deadline - 10: break
                if random.random() < 0.5:
                    new_pts = generate_random_points()
                else:
                    new_pts = generate_structured_points()
                if is_valid(new_pts):
                    current_pts = [p[:] for p in new_pts]
                    current_triples = get_collinear_triples(current_pts)
                    current_triple_count = len(current_triples)
                    current_score = compute_exact_score(current_pts)
                    break
            no_improve_steps = 0
            temp = 5.0

    return best_config

if __name__ == "__main__":
    res = construct()
    print(res)
