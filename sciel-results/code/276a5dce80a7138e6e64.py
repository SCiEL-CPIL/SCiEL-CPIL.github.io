import time
import math
from random import Random


def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def build_neigh(n, edges):
        m = len(edges)
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)
        return neigh

    def dsatur_color(m, neigh, deg, rng=None, rand_order=False):
        color = [-1] * m
        sat = [set() for _ in range(m)]
        remaining = list(range(m))
        for _ in range(m):
            best = -1
            best_key = None
            for v in remaining:
                if rand_order and rng is not None:
                    key = (len(sat[v]), deg[v], rng.random())
                else:
                    key = (len(sat[v]), deg[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            c = 0
            while c in sat[best]:
                c += 1
            color[best] = c
            remaining.remove(best)
            for u in neigh[best]:
                if color[u] == -1:
                    sat[u].add(c)
        return color, (max(color) + 1 if m else 0)

    def greedy_clique(m, neigh):
        neighset = [set(s) for s in neigh]
        cand = list(range(m))
        clique = 0
        while cand:
            best = cand[0]
            best_d = -1
            for x in cand:
                s = neighset[x]
                dd = sum(1 for y in cand if y in s)
                if dd > best_d:
                    best_d = dd
                    best = x
            clique += 1
            keep = neighset[best]
            cand = [x for x in cand if x != best and x in keep]
        return clique

    def recompute_conflict(color, neigh, m):
        conflict = [0] * m
        for v in range(m):
            cv = color[v]
            s = 0
            for u in neigh[v]:
                if color[u] == cv:
                    s += 1
            conflict[v] = s
        return conflict, sum(conflict) // 2

    def local_search(m, neigh, color_init, K, rng, time_limit):
        if K <= 0:
            return color_init, False
        color = [c if c < K else rng.randrange(K) for c in color_init]
        conflict, total = recompute_conflict(color, neigh, m)
        if total == 0:
            return color, True
        best_color = color[:]
        best_total = total
        no_improve = 0
        max_no_improve = 6000
        it = 0
        while time.time() < time_limit:
            if total == 0:
                return color, True
            # 2-opt move with probability 0.3, else 1-opt
            if total > 0 and rng.random() < 0.3:
                cand = [x for x in range(m) if conflict[x] > 0]
                if len(cand) >= 2:
                    v = cand[rng.randrange(len(cand))]
                    u = -1
                    for w in neigh[v]:
                        if color[w] != color[v]:
                            u = w
                            break
                    if u == -1:
                        u = cand[rng.randrange(len(cand))]
                    cv, cu = color[v], color[u]
                    if cv != cu:
                        delta = 0
                        for x in neigh[v]:
                            if color[x] == cv:
                                delta -= 1
                            elif color[x] == cu:
                                delta += 1
                        for x in neigh[u]:
                            if color[x] == cv:
                                delta += 1
                            elif color[x] == cu:
                                delta -= 1
                        if delta < 0 or (delta == 0 and rng.random() < 0.3):
                            for x in neigh[v]:
                                if color[x] == cv:
                                    conflict[x] -= 1
                                    conflict[v] -= 1
                                elif color[x] == cu:
                                    conflict[x] += 1
                                    conflict[v] += 1
                            for x in neigh[u]:
                                if color[x] == cv:
                                    conflict[x] += 1
                                    conflict[u] += 1
                                elif color[x] == cu:
                                    conflict[x] -= 1
                                    conflict[u] -= 1
                            color[v] = cu
                            color[u] = cv
                            total += delta
                            if total < best_total:
                                best_total = total
                                best_color = color[:]
                                no_improve = 0
                            else:
                                no_improve += 1
                            it += 1
                            if no_improve > max_no_improve:
                                color = best_color[:]
                                conflict, total = recompute_conflict(color, neigh, m)
                                no_improve = 0
                            continue
            # 1-opt move
            if total > 0:
                if rng.random() < 0.7:
                    v = -1
                    bv = -1
                    for x in range(m):
                        if conflict[x] > bv:
                            bv = conflict[x]
                            v = x
                else:
                    cand = [x for x in range(m) if conflict[x] > 0]
                    v = cand[rng.randrange(len(cand))]
            else:
                v = rng.randrange(m)
            old = color[v]
            best_c = old
            best_delta = 0
            for c in range(K):
                if c == old:
                    continue
                cnt_c = 0
                cnt_old = 0
                for u in neigh[v]:
                    if color[u] == c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                delta = cnt_c - cnt_old
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta
                    best_c = c
            if rng.random() < 0.1:
                best_c = rng.randrange(K)
                cnt_c = 0
                cnt_old = 0
                for u in neigh[v]:
                    if color[u] == best_c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                best_delta = cnt_c - cnt_old
            if best_c == old:
                continue
            for u in neigh[v]:
                if color[u] == old:
                    conflict[u] -= 1
                    conflict[v] -= 1
                if color[u] == best_c:
                    conflict[u] += 1
                    conflict[v] += 1
            color[v] = best_c
            total += best_delta
            if total < best_total:
                best_total = total
                best_color = color[:]
                no_improve = 0
            else:
                no_improve += 1
            it += 1
            if no_improve > max_no_improve:
                color = best_color[:]
                conflict, total = recompute_conflict(color, neigh, m)
                no_improve = 0
        return best_color, best_total == 0

    def mis_color_class(m, neigh, remaining_set, rng, time_limit):
        """
        Find a large independent set in the subgraph induced by remaining_set
        using a greedy + local search approach.
        Returns a list of vertices in the independent set.
        """
        if not remaining_set:
            return []
        
        # Greedy construction: repeatedly pick vertex with max degree in remaining
        # and add it, then remove it and its neighbors
        # Actually, for independent set we want to pick vertices that are NOT adjacent.
        # Greedy: pick vertex with minimum degree in the induced subgraph? 
        # Or use a heuristic: pick a random vertex, add to set, remove neighbors.
        
        # Better heuristic: 
        # 1. Order vertices by some criterion (e.g., degree in remaining)
        # 2. Greedily add vertex if no conflict with current set
        
        # Let's use a randomized greedy approach
        # Shuffle remaining, then greedily add if no neighbor in current set
        
        verts = list(remaining_set)
        rng.shuffle(verts)
        
        # Sort by degree in the induced subgraph (ascending) - lower degree first
        deg_in_rem = {}
        for v in verts:
            deg_in_rem[v] = sum(1 for u in neigh[v] if u in remaining_set)
        
        verts.sort(key=lambda x: deg_in_rem[x])
        
        current_set = []
        current_set_set = set()
        
        for v in verts:
            # Check if v has any neighbor in current_set
            conflict = False
            for u in neigh[v]:
                if u in current_set_set:
                    conflict = True
                    break
            if not conflict:
                current_set.append(v)
                current_set_set.add(v)
        
        # Local search: try to improve by swapping vertices
        # This is a simple 1-opt: remove a vertex, try to add a different one
        # Or: try to add a vertex not in set, remove a conflicting one
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add each vertex not in current_set
            for v in verts:
                if v in current_set_set:
                    continue
                # Find neighbors of v in current_set
                conflicts = [u for u in neigh[v] if u in current_set_set]
                if not conflicts:
                    current_set.append(v)
                    current_set_set.add(v)
                    improved = True
                elif len(conflicts) == 1:
                    # Swap: remove conflict, add v if v has fewer conflicts with rest
                    c = conflicts[0]
                    # Check if v has other conflicts (it shouldn't since we only found 1)
                    # Remove c, add v
                    current_set.remove(c)
                    current_set_set.remove(c)
                    current_set.append(v)
                    current_set_set.add(v)
                    improved = True
                    break
        
        return current_set

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        neigh = build_neigh(n, edges)
        deg = [len(neigh[i]) for i in range(m)]
        rng = Random(seed * 7919 + 42)

        # Best valid initial colouring: try several DSATUR variants, keep fewest colours
        best_color, best_K = dsatur_color(m, neigh, deg)
        for _ in range(8):
            col, K = dsatur_color(m, neigh, deg, rng=rng, rand_order=True)
            if K < best_K:
                best_color, best_K = col, K

        LB = greedy_clique(m, neigh)
        LB = max(LB, 1)

        # Use MIS-based coloring to try to reduce K further
        # Start from best_K and try to reduce by finding large independent sets
        
        current_color = best_color
        current_K = best_K
        
        # Try to reduce K by merging color classes using MIS
        # For each target_K from current_K-1 down to LB:
        #   Try to find a valid coloring with target_K colors
        
        for target_K in range(current_K - 1, LB - 1, -1):
            if time.time() > time_limit - 0.5:
                break
            
            # Try multiple restarts
            found = False
            for restart in range(5):
                if time.time() > time_limit - 0.5:
                    break
                
                # Initialize: take current_color and reassign colors >= target_K
                init = [c if c < target_K else rng.randrange(target_K) for c in current_color]
                
                # Run local search to fix conflicts
                res, ok = local_search(m, neigh, init, target_K, rng,
                                       min(time.time() + 1.5, time_limit - 0.5))
                if ok:
                    current_color = res
                    current_K = target_K
                    found = True
                    break
            
            if not found:
                # Try MIS-based approach: 
                # 1. Start with all vertices uncolored
                # 2. Repeatedly find a large independent set, assign it a color
                # 3. Continue until all vertices colored
                # 4. If we use <= target_K colors, we're done
                
                if time.time() < time_limit - 0.5:
                    color = [-1] * m
                    remaining = set(range(m))
                    color_id = 0
                    
                    while remaining and color_id < target_K:
                        # Find a large independent set in remaining
                        ind_set = mis_color_class(m, neigh, remaining, rng,
                                                 min(time.time() + 0.3, time_limit - 0.5))
                        if not ind_set:
                            # Fallback: pick a single vertex
                            v = next(iter(remaining))
                            ind_set = [v]
                        
                        for v in ind_set:
                            color[v] = color_id
                            remaining.remove(v)
                        
                        color_id += 1
                    
                    if not remaining:
                        # Valid coloring with color_id colors
                        if color_id <= target_K:
                            # Compress colors
                            current_color = compress(color)
                            current_K = len(set(current_color))
                            found = True
                        else:
                            # More colors than target_K, but still valid
                            # Use it if better than current
                            if color_id < current_K:
                                current_color = compress(color)
                                current_K = color_id
                                found = True
            
            if not found:
                break
        
        return compress(current_color)

    def fallback():
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            neigh = build_neigh(n, edges)
            deg = [len(neigh[i]) for i in range(m)]
            col, _ = dsatur_color(m, neigh, deg)
            out.append(col)
        return out

    try:
        start = time.time()
        master_deadline = start + 115.0
        weights = [d * d for (_, d, _) in specs]
        total_w = sum(weights)
        out = []
        cursor = start
        for idx, (n, d, seed) in enumerate(specs):
            frac = weights[idx] / total_w
            budget = (master_deadline - cursor) * frac
            budget = max(budget, 3.0)
            inst_deadline = min(cursor + budget, master_deadline)
            out.append(solve_instance(n, d, seed, inst_deadline))
            cursor = max(cursor, time.time())
        return out
    except Exception:
        return fallback()
