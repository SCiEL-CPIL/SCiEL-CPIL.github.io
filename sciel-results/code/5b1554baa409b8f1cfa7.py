    def _count_triples(points):
        """Count the number of distinct collinear triples in the set of points."""
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                # Normalize direction to avoid duplicates if we were using lines, 
                # but here we just check k > j
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.add((i, j, k))
        return len(triples)
    