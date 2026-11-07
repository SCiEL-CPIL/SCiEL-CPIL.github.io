import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 85.0          # safety margin under the 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)         # 91
    all_edges_set = set(all_edges)

    # ------------------------------------------------------------------
    # Fast incremental scorer: keeps the multiset (counts) of edge sums
    # and edge products so a single edge swap updates in O(1).
    # ------------------------------------------------------------------
    def make_scoring(values):
        sv = values[:]
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        def add(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 0:
                if which == 's':
                    distinct_sum += 1
                else:
                    distinct_prod += 1
            cnt[x] = c + 1

        def rem(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 1:
                del cnt[x]
                if which == 's':
                    distinct_sum -= 1
                else:
                    distinct_prod -= 1
            else:
                cnt[x] = c - 1

        def swap(out, inn):
            u, v = out
            s0 = sv[u] + sv[v]
            p0 = sv[u] * sv[v]
            rem(s0, cnt_sum, 's')
            rem(p0, cnt_prod, 'p')
            u, v = inn
            s1 = sv[u] + sv[v]
            p1 = sv[u] * sv[v]
            add(s1, cnt_sum, 's')
            add(p1, cnt_prod, 'p')

        def add_edge(e):
            u, v = e
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def raw():
            return max(distinct_sum, distinct_prod)

        def init_from(edges):
            for e in edges:
                add_edge(e)

        return init_from, swap, add_edge, raw, (cnt_sum, cnt_prod)

    def full_score(values, edge_list):
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(values[u] + values[v])
            p.add(values[u] * values[v])
        return max(len(s), len(p))

    # ------------------------------------------------------------------
    # Candidate value pools: small / dense / composite-rich so that both
    # sums and products collide heavily among the 35 chosen edges.
    # ------------------------------------------------------------------
    def value_pools():
        pools = []
        # dense small
        for hi in (18, 22, 28, 36, 48, 64, 90, 140, 220):
            pools.append(list(range(1, hi + 1)))
        # composites / multiples (product collisions)
        base_sets = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 27, 30,
             32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 60, 63, 64, 70, 72, 80, 81,
             84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144,
             147, 150, 154, 160, 162, 168, 175, 180, 189, 196, 200],
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 25, 27, 32, 36, 40, 45,
             49, 54, 64, 72, 80, 81, 90, 98, 100, 108, 121, 128, 144, 147, 162, 169,
             196, 200, 225, 242, 243, 256, 288, 289, 324, 338, 343, 361, 392, 400,
             441, 484, 512, 576, 625, 676],
            # powers-of-2 rich (multiplicative structure)
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72,
             81, 96, 108, 128, 144, 162, 192, 216, 256, 288, 324, 384, 432, 486,
             512, 576, 648, 768, 864, 972, 1024, 1152, 1296, 1536, 1728, 2048,
             2304, 2592, 3072, 4096],
        ]
        pools.extend(base_sets)
        return pools

    def random_values(pool):
        if len(pool) >= n:
            return random.sample(pool, n)
        vals = []
        seen = set()
        while len(vals) < n:
            v = random.randint(1, 10000)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    # ------------------------------------------------------------------
    # Greedy edge selection for fixed values (build high-collision set).
    # ------------------------------------------------------------------
    def greedy_edges(values):
        init_from, swap, add_edge, raw, _ = make_scoring(values)
        chosen = []
        chosen_set = set()
        # order edges by small sum (tends to reuse small sums/products)
        order = sorted(all_edges, key=lambda e: (values[e[0]] + values[e[1]],
                                                 values[e[0]] * values[e[1]]))
        for e in order:
            if len(chosen) == m:
                break
            add_edge(e)
            chosen.append(e)
            chosen_set.add(e)
        return chosen, chosen_set

    # ------------------------------------------------------------------
    # Edge-only local search (incremental).
    # ------------------------------------------------------------------
    def polish_edges(values, chosen, chosen_set, steps, deadline):
        init_from, swap, add_edge, raw, _ = make_scoring(values)
        init_from(chosen)
        cur = raw()
        best = cur
        for _ in range(steps):
            if time.time() > deadline:
                break
            i = random.randrange(m)
            e_out = chosen[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            swap(e_out, e_in)
            r = raw()
            if r <= cur:
                cur = r
                chosen[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                if r < best:
                    best = r
            else:
                # revert
                swap(e_in, e_out)
        return chosen, chosen_set, best

    # ------------------------------------------------------------------
    # New: Generate regular graph edges
    # ------------------------------------------------------------------
    def generate_regular_edges(degree):
        # 14 vertices. 3-regular has 42/2 = 21 edges. 4-regular has 56/2 = 28 edges.
        # We need 35 edges. 35 is not a regular degree count for n=14 (35*2=70, 70/14=5).
        # So 5-regular is exactly 35 edges!
        if degree == 5:
            # Construct a 5-regular graph on 14 vertices
            # Simple approach: random permutation and connect i to i+1, i+2, i+3, i+4, i+5 mod 14?
            # This creates a circulant graph.
            edges = []
            for i in range(n):
                for j in range(1, degree // 2 + 1):
                    u = i
                    v = (i + j) % n
                    if u > v: u, v = v, u
                    if (u, v) not in edges:
                        edges.append((u, v))
            # Ensure we have exactly m edges
            if len(edges) < m:
                # Fill with random edges
                remaining = [e for e in all_edges if e not in edges]
                random.shuffle(remaining)
                edges.extend(remaining[:m - len(edges)])
            elif len(edges) > m:
                edges = edges[:m]
            return [list(e) for e in edges]
        else:
            return random.sample(all_edges, m)

    # ------------------------------------------------------------------
    # New: Collision-guided value move
    # ------------------------------------------------------------------
    def get_guided_value_moves(vals, edges, cnt_sum, cnt_prod, max_candidates=10):
        # For each vertex, find values that create collisions
        # We'll pick a random vertex and try to find a value for it
        idx = random.randrange(n)
        old_val = vals[idx]
        
        # Get neighbors
        neighbors = set()
        for u, v in edges:
            if u == idx:
                neighbors.add(v)
            elif v == idx:
                neighbors.add(u)
        
        # If no neighbors, random move
        if not neighbors:
            return idx, random.randint(1, 50)
        
        # Existing sums and products involving idx
        current_sums = set()
        current_prods = set()
        for nb in neighbors:
            current_sums.add(vals[idx] + vals[nb])
            current_prods.add(vals[idx] * vals[nb])
            
        # Other sums and products in the graph (excluding edges involving idx)
        other_sums = set(cnt_sum.keys())
        other_prods = set(cnt_prod.keys())
        
        # Remove current sums/prods from "other" sets to find collisions with non-incident edges?
        # Actually, we want to collide with ANY existing sum/prod in the graph.
        # If we change vals[idx], the sums/prods for edges (idx, nb) change.
        # We want new (vals[idx] + vals[nb]) to be in cnt_sum (current counts)
        # or new (vals[idx] * vals[nb]) to be in cnt_prod.
        
        # Note: cnt_sum/cnt_prod currently include the edges involving idx.
        # When we propose a new value, we should check against the set of sums/prods 
        # that would remain if we removed the old edges involving idx?
        # Approximation: Check against current cnt_sum/cnt_prod. 
        # If a new sum matches an old sum of another edge, it's a collision.
        
        candidates = set()
        
        # Try values in a small range around old_val and random small values
        search_range = range(1, 60)
        
        for nv in search_range:
            if nv == old_val:
                continue
            # Check if nv is already used by another vertex
            # (We will check this in the main loop, but filtering here saves time)
            if nv in vals:
                continue
                
            # Check collisions for sums
            sum_collisions = 0
            prod_collisions = 0
            for nb in neighbors:
                ns = nv + vals[nb]
                np_ = nv * vals[nb]
                # If this sum exists in the graph (count > 0), it's a potential collision
                # We subtract 1 from count if it was the old sum? 
                # Simplified: just check existence in the global sets.
                if ns in cnt_sum:
                    sum_collisions += 1
                if np_ in cnt_prod:
                    prod_collisions += 1
            
            # Score this candidate
            score = sum_collisions + prod_collisions
            if score > 0:
                candidates.add((score, nv))
                
        if not candidates:
            # Fallback to random
            return idx, random.randint(1, 50)
            
        # Pick the best candidate (highest collision count)
        # Randomize among top candidates to avoid getting stuck
        candidates = sorted(candidates, reverse=True)
        top_k = candidates[:min(max_candidates, len(candidates))]
        _, nv = random.choice(top_k)
        return idx, nv

    # ------------------------------------------------------------------
    # Joint value+edge local search with Simulated Annealing
    # ------------------------------------------------------------------
    def joint_search(vals0, edges0, steps, deadline, use_annealing=True):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, (cnt_sum, cnt_prod) = make_scoring(vals)
        init_from(edges)
        
        temp = 10.0 # Initial temperature
        cooling = 0.995
        
        for i in range(steps):
            if time.time() > deadline:
                break
            
            if use_annealing:
                temp *= cooling
            
            if random.random() < 0.5:
                # Edge move
                e_out_idx = random.randrange(m)
                e_out = edges[e_out_idx]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                swap(e_out, e_in)
                r = raw()
                delta = r - cur
                if delta < 0 or (use_annealing and random.random() < math.exp(-delta / max(1e-6, temp))):
                    cur = r
                    edges[e_out_idx] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    swap(e_in, e_out)
            else:
                # Value move
                # 30% guided, 70% random
                if random.random() < 0.3:
                    idx, nv = get_guided_value_moves(vals, edges, cnt_sum, cnt_prod)
                else:
                    idx = random.randrange(n)
                    if random.random() < 0.6:
                        nv = random.randint(1, 40)
                    else:
                        nv = random.randint(1, 120)
                
                old = vals[idx]
                if nv == old or nv in vals:
                    continue
                    
                vals[idx] = nv
                r = full_score(vals, edges)
                delta = r - cur
                if delta < 0 or (use_annealing and random.random() < math.exp(-delta / max(1e-6, temp))):
                    cur = r
                    # Rebuild incremental state
                    init_from, swap, add_edge, raw, (cnt_sum, cnt_prod) = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    vals[idx] = old
                    
        return best, best_vals, best_edges

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 1: Structured value pools -> greedy edges -> polish edges.
    pools = value_pools()
    random.shuffle(pools)
    for pool in pools:
        if time.time() > deadline - 10.0:
            break
        for _ in range(2):
            if time.time() > deadline - 9.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            # polish edges for this fixed value set
            steps = 6000
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               steps, deadline - 8.0)
            consider(psc, vals, polished)

    # Phase 2: Joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))
    
    # Add regular graph starts
    for deg in [5]:
        reg_edges = generate_regular_edges(deg)
        for _ in range(3):
            vals = random_values(pools[random.randrange(len(pools))])
            starts.append((vals, reg_edges))
            
    for _ in range(4):
        vals = random_values(pools[random.randrange(len(pools))])
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    remaining = deadline - 10.0 - time.time()
    per = max(4000, int(remaining / max(1, len(starts)) / 0.0002))
    
    for i, (vals0, edges0) in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        # Use annealing for all, but maybe stronger for later ones?
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 9.0, use_annealing=True)
        consider(sc, vals, edges)

    # Phase 3: Final edge polish for best values.
    if time.time() < deadline - 4.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           40000, deadline - 3.0)
        consider(psc, best_vals, polished)

    # Phase 4: Final value polish for best edges.
    if time.time() < deadline - 1.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        while it < 20000 and time.time() < deadline - 1.0:
            it += 1
            # Use guided moves here too
            if random.random() < 0.5:
                idx, nv = get_guided_value_moves(vals, edges, {}, {}) # Note: cnts are stale, but heuristic is okay
            else:
                idx = random.randrange(n)
                nv = random.randint(1, 80) if random.random() < 0.6 else random.randint(1, 160)
            
            old = vals[idx]
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
            else:
                vals[idx] = old

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)

    # ------------------------------------------------------------------
    # Strict type conversion (Python int / list only).
    # ------------------------------------------------------------------
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    # defensive: ensure exactly 35 unique simple edges, 14 distinct ints
    seen = set()
    clean = []
    for e in final_edges:
        a, b = e
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean.append([a, b])
    # if somehow short, pad with missing edges (should not happen)
    if len(clean) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean.append([e[0], e[1]])
                if len(clean) == m:
                    break
    final_edges = clean[:m]

    return {'values': final_values, 'edges': final_edges}
