<<<<<<< SEARCH
def _build_triples(points):
    n = len(points)
    triples = set()
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            for a, b in combinations(k, 2):
                triples.add(frozenset((i, a, b)))
    return [tuple(t) for t in triples]

def _min_hitting_set(triples_list, n, deadline_start):
=======
def _build_triples(points):
    n = len(points)
    triples = set()
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            for a, b in combinations(k, 2):
                triples.add(frozenset((i, a, b)))
    return [tuple(t) for t in triples]

def _count_triples(points):
    return len(_build_triples(points))

def _min_hitting_set(triples_list, n, deadline_start):
>>>>>>> REPLACE

