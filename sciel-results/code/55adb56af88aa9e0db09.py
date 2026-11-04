import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 100  # leave buffer for overhead
    
    # Generate support-size-2 candidates (baseline: 220)
    candidates2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates2.append(v)
    candidates2 = np.array(candidates2, dtype=np.int64)
    baseline = len(candidates2)  # 220
    
    # Generate support-size-3 candidates
    candidates3 = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates3.append(v)
    candidates3 = np.array(candidates3, dtype=np.int64)
    n3 = len(candidates3)  # 1320
    
    # Build conflict graph for support-3
    # All have norm^2=3, so conflict if dot product > 1
    dots3 = candidates3 @ candidates3.T
    conflict3 = (dots3 > 1)
    np.fill_diagonal(conflict3, False)
    
    # Convert to adjacency list for non-conflicts (complement graph edges)
    # For maximum independent set in conflict3, it's maximum clique in complement
    adj3 = [set() for _ in range(n3)]
    for i in range(n3):
        for j in range(i+1, n3):
            if not conflict3[i, j]:
                adj3[i].add(j)
                adj3[j].add(i)
    
    # Branch and bound for maximum clique in complement graph
    best_size = 0
    best_clique = []
    
    # Use recursive search with coloring bound
    # To keep it simple and within time, use a heuristic-guided search
    
    # Order vertices by degree
    degrees = [len(adj3[i]) for i in range(n3)]
    order = sorted(range(n3), key=lambda x: -degrees[x])
    
    # Remap
    remap = {old: new for new, old in enumerate(order)}
    n_remapped = n3
    adj_remapped = [set() for _ in range(n_remapped)]
    for i in range(n3):
        for j in adj3[i]:
            adj_remapped[remap[i]].add(remap[j])
    
    # Maximum clique with branch and bound
    # Simple recursive with pruning
    def max_clique_search(R, P, depth):
        nonlocal best_size, best_clique
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Upper bound: len(R) + len(P)
        if len(R) + len(P) <= best_size:
            return
        
        # Pick a vertex from P
        # Use heuristic: pick vertex with max degree in P
        v = None
        max_deg = -1
        for u in P:
            deg = len([x for x in P if x in adj_remapped[u]])
            if deg > max_deg:
                max_deg = deg
                v = u
            if time.time() - start_time > time_limit:
                return
        
        if v is None:
            return
        
        # Branch 1: include v
        R.append(v)
        P_new = P.intersection(adj_remapped[v])
        max_clique_search(R, P_new, depth+1)
        R.pop()
        
        # Branch 2: exclude v
        P.discard(v)
        max_clique_search(R, P, depth+1)
        P.add(v)
    
    # This will be too slow for 1320 nodes. Let's use a greedy + local search instead.
    
    # Greedy independent set on conflict3
    used = [False] * n3
    independent_set = []
    remaining = list(range(n3))
    
    # Sort by degree in conflict graph (ascending for independent set)
    conflict_degrees = [conflict3[i].sum() for i in range(n3)]
    remaining.sort(key=lambda x: conflict_degrees[x])
    
    for v in remaining:
        if used[v]:
            continue
        # Check if v conflicts with any already selected
        can_add = True
        for u in independent_set:
            if conflict3[v, u]:
                can_add = False
                break
        if can_add:
            independent_set.append(v)
            used[v] = True
            # Mark conflicts
            for u in range(n3):
                if conflict3[v, u]:
                    used[u] = True
    
    greedy_size = len(independent_set)
    
    # Try local search: swap to improve
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        current_set = set(independent_set)
        for v in current_set:
            # Try removing v and adding a better one
            for u in range(n3):
                if u in current_set:
                    continue
                # Check if u can replace v
                # u must not conflict with any in current_set - {v}
                ok = True
                for w in current_set:
                    if w != v and conflict3[u, w]:
                        ok = False
                        break
                if ok:
                    # This swap keeps size same, try to find improvement
                    pass
        
        # Try adding any non-conflicting vertex
        for u in range(n3):
            if u in current_set:
                continue
            ok = True
            for w in current_set:
                if conflict3[u, w]:
                    ok = False
                    break
            if ok:
                independent_set.append(u)
                current_set.add(u)
                improved = True
    
    final_size = len(independent_set)
    
    if final_size > baseline:
        result = candidates3[independent_set]
        return result
    else:
        return candidates2
