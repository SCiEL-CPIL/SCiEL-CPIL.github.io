def _is_valid(pos, r, eps=1e-9):
    n = len(pos)
    for i in range(n):
        xi, yi = pos[i]
        ri = r[i]
        # Boundary check
        if xi < -eps or yi < -eps or xi + ri > 1.0 + eps or yi + ri > 1.0 + eps:
            return False
        # Pairwise disjoint check
        for j in range(i + 1, n):
            xj, yj = pos[j]
            rj = r[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            min_dist = ri + rj - eps
            if dist_sq < min_dist * min_dist:
                return False
    return True
