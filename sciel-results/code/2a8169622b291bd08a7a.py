    centers = []
    centers.append((0.0, 0.0))
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if q == 0 and r == 0: continue
            x = math.sqrt(3) * (q + r / 2.0)
            y = math.sqrt(3) * (math.sqrt(3) / 2.0) * r
            dist_sq = x**2 + y**2
            candidates.append((dist_sq, q, r, x, y))
    candidates.sort()
    for _, _, _, x, y in candidates:
        centers.append((x, y))
        if len(centers) == N:
            break
