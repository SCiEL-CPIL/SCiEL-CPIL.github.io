import math

def construct():
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a + b)) // 2

    def angle(p):
        a, b = p
        x = a + b * 0.5
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # Generate hexagonal spiral points
    candidates = []
    R = 4
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])

    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates[:32]]

    # Verify and return
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def count_edges(points):
        lookup = set((p[0], p[1]) for p in points)
        count = 0
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a + da, b + db) in lookup:
                    count += 1
        return count // 2

    def verify(points):
        if len(points) != 32:
            return False
        seen = set()
        for p in points:
            a, b = p
            if abs(a) > 64 or abs(b) > 64:
                return False
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True

    if not verify(result):
        # Fallback: simpler grid-based construction
        result = []
        for a in range(-5, 6):
            for b in range(-5, 6):
                if len(result) < 32:
                    result.append([a, b])

    return result
