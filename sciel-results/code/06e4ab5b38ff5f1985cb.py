import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    def count_edges(points):
        lookup = set((p[0], p[1]) for p in points)
        count = 0
        neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a + da, b + db) in lookup:
                    count += 1
        return count // 2

    # Generate hex spiral: sort by (hex_dist, angle), take first 32
    R = 6
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])

    candidates_sorted = sorted(candidates, key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates_sorted[:32]]

    # Verify: should achieve 76 edges (Harborth optimum for n=32)
    edge_count = count_edges(result)
    valid = all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in result)
    distinct = len(set((p[0], p[1]) for p in result)) == 32

    if valid and distinct and edge_count == 76:
        return result

    # Fallback: if spiral doesn't achieve 76 (shouldn't happen), try a quick local search
    if edge_count < 76:
        import random
        current_points = [p[:] for p in result]
        current_score = edge_count
        neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

        for _ in range(50000):
            if time.time() > deadline - 5.0:
                break
            if current_score >= 76:
                break

            idx = random.randint(0, 31)
            a, b = current_points[idx]
            da = random.randint(-3, 3)
            db = random.randint(-3, 3)
            new_a = a + da
            new_b = b + db

            if abs(new_a) > 64 or abs(new_b) > 64:
                continue
            if any(p[0] == new_a and p[1] == new_b for j, p in enumerate(current_points) if j != idx):
                continue

            current_points[idx] = [new_a, new_b]
            new_score = count_edges(current_points)

            if new_score > current_score:
                current_score = new_score
            else:
                current_points[idx] = [a, b]

        return current_points

    return result
