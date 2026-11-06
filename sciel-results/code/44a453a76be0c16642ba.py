def _sa_run(k, budget, seed=0):
    rnd = random.Random(seed)
    c = np.array([rnd.randrange(k) for _ in range(N)], dtype=np.int8)
    
    ap_bad = np.zeros(_NUM_APS, dtype=np.int8)
    total_cost = 0
    for idx, ap in enumerate(_APS):
        d = _count_distinct(c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]])
        if d < 3:
            ap_bad[idx] = 1
            total_cost += 1
            
    best_c = c.copy()
    best_cost = total_cost
    initial_cost = total_cost
    
    t0 = time.time()
    T = 5.0
    cool = 0.9999
    exp = math.exp
    
    while time.time() - t0 < budget:
        p = rnd.randrange(N)
        newc = rnd.randrange(k)
        if newc == c[p]:
            continue
            
        oldc = c[p]
        c[p] = newc
        
        delta = 0
        for idx in _pos_aps[p]:
            ap = _APS[idx]
            old_bad = ap_bad[idx]
            d = _count_distinct(c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]])
            new_bad = 1 if d < 3 else 0
            if new_bad != old_bad:
                delta += new_bad - old_bad
                ap_bad[idx] = new_bad
                
        new_cost = total_cost + delta
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_c = c.copy()
            if best_cost == 0:
                return best_c
        
        if new_cost <= total_cost or rnd.random() < exp(-delta / T):
            total_cost = new_cost
        else:
            c[p] = oldc
            for idx in _pos_aps[p]:
                ap = _APS[idx]
                d = _count_distinct(c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]])
                ap_bad[idx] = 1 if d < 3 else 0
                
        T *= cool
        
        # Early abort check
        if time.time() - t0 > 1.0 and best_cost > initial_cost * 0.9:
            # No significant improvement in 1s
            # Maybe this k is impossible or very hard
            # But we shouldn't abort too early for small k
            pass
            
    return best_c
