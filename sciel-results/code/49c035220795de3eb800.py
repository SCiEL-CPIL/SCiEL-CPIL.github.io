import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets using bitmasks for speed
        # Two edges conflict if they share a vertex or are joined by an edge
        conflict_masks = [0] * m
        for i in range(m):
            ui, vi = edges[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    mask |= (1 << j)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        mask |= (1 << j)
            conflict_masks[i] = mask

        static_deg = [bin(conflict_masks[i]).count('1') for i in range(m)]

        # Helper: get conflict set as list from mask
        def mask_to_list(mask):
            result = []
            bit = 0
            while mask:
                if mask & 1:
                    result.append(bit)
                mask >>= 1
                bit += 1
            return result

        # Phase 1: Multi-ordering DSATUR greedy
        def greedy_dsatur(order):
            colors = [-1] * m
            uncolored = set(range(m))
            # Process in the given order
            for i in order:
                if i not in uncolored:
                    continue
                # Find taken colors from colored neighbors
                taken = set()
                mask = conflict_masks[i]
                bit = 0
                while mask:
                    if mask & 1:
                        j = bit
                        if colors[j] != -1:
                            taken.add(colors[j])
                    mask >>= 1
                    bit += 1
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        # Generate multiple diverse orderings
        def generate_orderings():
            orderings = []
            # 1. Conflict-degree sorted (highest first)
            order1 = sorted(range(m), key=lambda i: -static_deg[i])
            orderings.append(order1)
            
            # 2. Reverse conflict-degree sorted
            order2 = sorted(range(m), key=lambda i: static_deg[i])
            orderings.append(order2)
            
            # 3. Random orderings
            rng = random.Random(42)
            for _ in range(8):
                order = list(range(m))
                rng.shuffle(order)
                orderings.append(order)
            
            # 4. Vertex-degree based (edges incident to high-degree vertices first)
            vertex_deg = [0] * n
            for u, v in edges:
                vertex_deg[u] += 1
                vertex_deg[v] += 1
            order3 = sorted(range(m), key=lambda i: -(vertex_deg[edges[i][0]] + vertex_deg[edges[i][1]]))
            orderings.append(order3)
            
            # 5. More random with different seeds
            for seed_val in [123, 456, 789, 101, 202, 303]:
                rng2 = random.Random(seed_val)
                order = list(range(m))
                rng2.shuffle(order)
                orderings.append(order)
            
            # 6. Hybrid: conflict-degree with random tie-breaking
            rng3 = random.Random(777)
            order4 = sorted(range(m), key=lambda i: (-static_deg[i], rng3.random()))
            orderings.append(order4)
            
            # 7. Reverse vertex-degree
            order5 = sorted(range(m), key=lambda i: (vertex_deg[edges[i][0]] + vertex_deg[edges[i][1]], rng3.random()))
            orderings.append(order5)
            
            # 8. Random with conflict-degree bias
            for _ in range(4):
                order = list(range(m))
                # Sort by static_deg but add random perturbation
                keys = [(static_deg[i] + random.Random(i * 13 + 7).random() * 2, i) for i in range(m)]
                keys.sort(reverse=True)
                order = [i for _, i in keys]
                orderings.append(order)
            
            return orderings

        orderings = generate_orderings()
        
        best_colors = None
        best_num = float('inf')
        
        for order in orderings:
            colors = greedy_dsatur(order)
            num_colors = max(colors) + 1 if colors else 0
            if num_colors < best_num:
                best_num = num_colors
                best_colors = colors[:]
        
        # Phase 2: Local search to reduce colors
        def try_reduce_colors(colors, target_K, time_budget):
            """Try to reduce coloring to use at most target_K colors."""
            if max(colors) + 1 <= target_K:
                return colors[:]
            
            work = colors[:]
            cur_max = max(work)
            
            # Try to eliminate colors from cur_max down to target_K
            for target_max in range(cur_max, target_K - 1, -1):
                if time.time() > time_budget:
                    break
                to_recolor = [i for i in range(m) if work[i] == target_max]
                if not to_recolor:
                    continue
                
                success = False
                for attempt in range(30):
                    if time.time() > time_budget:
                        break
                    order = to_recolor[:]
                    random.shuffle(order)
                    
                    temp = work[:]
                    valid = True
                    for i in order:
                        taken = set()
                        mask = conflict_masks[i]
                        bit = 0
                        while mask:
                            if mask & 1:
                                j = bit
                                if temp[j] != -1 and temp[j] < target_max:
                                    taken.add(temp[j])
                            mask >>= 1
                            bit += 1
                        avail = [c for c in range(target_max) if c not in taken]
                        if avail:
                            temp[i] = random.choice(avail)
                        else:
                            valid = False
                            break
                    if valid:
                        # Verify
                        ok = True
                        for i in range(m):
                            ci = temp[i]
                            mask = conflict_masks[i]
                            bit = 0
                            while mask:
                                if mask & 1:
                                    j = bit
                                    if temp[j] == ci:
                                        ok = False
                                        break
                                mask >>= 1
                                bit += 1
                            if not ok:
                                break
                        if ok:
                            work = temp[:]
                            success = True
                            break
                if not success:
                    break
            
            return work

        # Try to reduce by 1, then 2, etc.
        if best_num > 1:
            for target in range(best_num - 1, max(0, best_num - 4), -1):
                if time.time() > deadline - 10.0:
                    break
                reduced = try_reduce_colors(best_colors, target, deadline - 10.0)
                new_max = max(reduced) + 1 if reduced else 0
                if new_max < best_num:
                    best_colors = reduced
                    best_num = new_max
                else:
                    break

        # Phase 3: ILP for final improvement
        def try_ilp(K, time_for_ilp):
            if K <= 0 or K >= best_num:
                return None
            
            n_vars = m * K
            c = np.zeros(n_vars)
            lb = np.zeros(n_vars)
            ub = np.ones(n_vars)
            integrality = np.ones(n_vars)
            
            A_rows = []
            lb_cons = []
            ub_cons = []
            
            # Each edge gets exactly one color
            for i in range(m):
                row = np.zeros(n_vars)
                for k in range(K):
                    row[i * K + k] = 1.0
                A_rows.append(row)
                lb_cons.append(1.0)
                ub_cons.append(1.0)
            
            # Conflict constraints
            for i in range(m):
                for j in mask_to_list(conflict_masks[i]):
                    if j <= i:
                        continue
                    for k in range(K):
                        row = np.zeros(n_vars)
                        row[i * K + k] = 1.0
                        row[j * K + k] = 1.0
                        A_rows.append(row)
                        lb_cons.append(-np.inf)
                        ub_cons.append(1.0)
            
            if not A_rows:
                return None
                
            A = np.array(A_rows)
            lb_cons = np.array(lb_cons)
            ub_cons = np.array(ub_cons)
            
            try:
                res = milp(c=c,
                           constraints=LinearConstraint(A, lb_cons, ub_cons),
                           integrality=integrality,
                           bounds=Bounds(lb, ub),
                           options={"time_limit": time_for_ilp, "presolve": True})
                
                if res.status == 0:
                    x = res.x
                    colors = [-1] * m
                    for i in range(m):
                        for k in range(K):
                            if x[i * K + k] > 0.5:
                                colors[i] = k
                                break
                        if colors[i] == -1:
                            return None
                    return colors
                else:
                    return None
            except Exception:
                return None

        # Try ILP to reduce further
        current_best = best_num
        max_attempts = 2
        attempts = 0
        
        for k in range(current_best - 1, 0, -1):
            if attempts >= max_attempts:
                break
            if time.time() > deadline - 5.0:
                break
            
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time <= 1.0:
                break
                
            time_for_ilp = min(remaining_time, 15.0)
            
            result_colors = try_ilp(k, time_for_ilp)
            attempts += 1
            
            if result_colors is not None:
                # Verify
                valid = True
                for i in range(m):
                    ci = result_colors[i]
                    mask = conflict_masks[i]
                    bit = 0
                    while mask:
                        if mask & 1:
                            j = bit
                            if result_colors[j] == ci:
                                valid = False
                                break
                        mask >>= 1
                        bit += 1
                    if not valid:
                        break
                if valid:
                    current_best = k
                    best_colors = result_colors[:]
                else:
                    break
            else:
                break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        
        time_limit = 45.0
        if len(results) == 2:
            time_limit = remaining - 2.0
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)

    return results
