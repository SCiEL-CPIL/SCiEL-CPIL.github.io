<<<<<<< SEARCH
def _greedy_construct(a, b, n):
    """Greedy construction: assign facilities one by one to the best location."""
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Order facilities by some heuristic (e.g., row sum of a)
    order = np.argsort(-a.sum(axis=1), kind='stable')
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used[loc]:
                continue
            # Cost contribution of placing fac at loc, considering already placed facilities
            # sum over already placed k: a[fac,k]*b[loc,p[k]] + a[k,fac]*b[p[k],loc] + a[fac,fac]*b[loc,loc]
            placed = np.where(~used)[0]
            if len(placed) == 0:
                cost = a[fac, fac] * b[loc, loc]
            else:
                # Only consider already placed facilities
                idx = np.where(~used & (np.arange(n) != loc))[0]
                # Actually, we need to check which are already assigned
                assigned_locs = p[~used]  # this doesn't work cleanly
                # Let's just compute incrementally
                cost = a[fac, fac] * b[loc, loc]
                for k in range(n):
                    if not used[k]:
                        continue
                    cost += a[fac, k] * b[loc, p[k]] + a[k, fac] * b[p[k], loc]
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used[best_loc] = True
    
    return p.tolist()
=======
def _greedy_construct(a, b, n):
    """Greedy construction: assign facilities one by one to the best location."""
    p = np.empty(n, dtype=np.int64)
    p.fill(-1)
    used = np.zeros(n, dtype=bool)
    
    order = np.argsort(-a.sum(axis=1), kind='stable')
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used[loc]:
                continue
            cost = a[fac, fac] * b[loc, loc]
            for k in range(n):
                if not used[k]:
                    continue
                cost += a[fac, k] * b[loc, p[k]] + a[k, fac] * b[p[k], loc]
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used[best_loc] = True
    
    return p.tolist()
>>>>>>> REPLACE

