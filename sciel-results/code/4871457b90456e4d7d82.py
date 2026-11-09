def score(points):
    P = np.array(points, dtype=np.int64)
    n = len(P)
    # Compute all pairwise differences? No, just triples.
    # Vectorized triple loop is tricky but N=30 is small.
    # Simple loop is fast enough.
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Determinant
                dx1 = P[j,0] - P[i,0]
                dy1 = P[j,1] - P[i,1]
                dx2 = P[k,0] - P[i,0]
                dy2 = P[k,1] - P[i,1]
                det = dx1*dy2 - dy1*dx2
                if det != 0:
                    areas.append(abs(det))
    if not areas: return 0
    # Count frequencies
    # Using collections.Counter or np.unique
    # np.unique is fast for small arrays
    vals, counts = np.unique(areas, return_counts=True)
    return max(counts)
