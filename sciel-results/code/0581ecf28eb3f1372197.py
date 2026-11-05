    def _has_four_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        for i in range(n):
            for j in range(i+1, n):
                cnt = 0
                pi, pj = pts[i], pts[j]
                for k in range(n):
                    if k == i or k == j: continue
                    if _collinear3(pi, pj, pts[k]):
                        cnt += 1
                        if cnt >= 2:
                            return True
        return False
    