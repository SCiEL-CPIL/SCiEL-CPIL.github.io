def _try_extend(current, deadline):
    n = len(current)
    p0 = current[0]
    x0, y0 = p0
    
    max_k = 400 
    
    t0 = time.perf_counter()
    
    max_d_existing = 0
    for p in current[1:]:
        dx = p[0] - x0
        dy = p[1] - y0
        d2 = dx*dx + dy*dy
        if d2 > max_d_existing:
            max_d_existing = d2
    max_r_existing = isqrt(max_d_existing)
    
    max_dist_bound = max_k + max_r_existing + 10
    max_sq_needed = max_dist_bound * max_dist_bound
    
    squares = set()
    sqrt_map = {}
    limit = isqrt(max_sq_needed)
    for i in range(limit + 1):
        sq = i * i
        squares.add(sq)
        sqrt_map[sq] = i
        
    for k in range(1, max_k + 1):
        if time.perf_counter() > deadline:
            break
        k_sq = k * k
        for u in range(-k, k + 1):
            rem = k_sq - u * u
            if rem < 0: continue
            if rem not in sqrt_map:
                continue
            v = sqrt_map[rem]
            
            cands = []
            cands.append((x0 + u, y0 + v))
            if v != 0:
                cands.append((x0 + u, y0 - v))
            
            for (x, y) in cands:
                if (x, y) == p0:
                    continue
                if (x, y) in current:
                    continue
                
                ok = True
                for i in range(1, n):
                    dx = x - current[i][0]
                    dy = y - current[i][1]
                    d2 = dx * dx + dy * dy
                    if d2 not in squares:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                bad = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if _collinear3(current[a], current[b], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                
                for a, b, c in combinations(range(n), 3):
                    if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    continue
                
                return current + [(x, y)]
                    
    return current
