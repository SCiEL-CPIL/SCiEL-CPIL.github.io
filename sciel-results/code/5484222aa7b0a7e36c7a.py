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

        # Precompute conflict sets: two edges conflict if they share a vertex
        # or are joined by an edge (distance 2 in line graph sense)
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        static_deg = [len(conflict_sets[i]) for i in range(m)]

        # Phase 1: DSATUR greedy initialization
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            while uncolored:
                best_i, best_score = None, (-1, -1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg, static_deg[i])
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        best_colors = greedy_dsatur()
        best_num = max(best_colors) + 1 if best_colors else 0

        # Phase 2: Local search via swaps to reduce colors
        # Try to reduce from best_num-1 down by finding valid recolorings
        def try_reduce_to(K, colors, time_budget):
            """Try to recolor to use at most K colors via local search."""
            if K >= max(colors) + 1:
                return colors[:]
            
            # Work on a copy
            work = colors[:]
            # First, ensure all colors are < K by remapping if needed
            # If current max >= K, we need to find a valid assignment with K colors
            # This is harder; let's use a different strategy:
            # Start from scratch with K-color constraint using backtracking with pruning
            
            # Actually, let's do a simpler local search:
            # Randomly try to move edges to lower colors if possible
            # Then try to eliminate one color class entirely
            
            # Better approach: use simulated annealing / hill climbing
            # Objective: minimize max color used, subject to validity
            
            # Let's do: repeatedly try to reduce max color by 1
            # by finding a valid recoloring of the highest color class into lower colors
            
            current = work[:]
            cur_max = max(current) if current else -1
            
            # Try to eliminate colors from cur_max down to K
            for target_max in range(cur_max, K - 1, -1):
                if time.time() > time_budget:
                    break
                # Try to recolor all edges with color target_max into colors < target_max
                # This is a subproblem: can we color the edges of color target_max
                # using only colors 0..target_max-1, given the current coloring of others?
                
                # Collect edges with color target_max
                to_recolor = [i for i in range(m) if current[i] == target_max]
                if not to_recolor:
                    continue
                
                # For each edge in to_recolor, available colors are those < target_max
                # that don't conflict with any already-colored neighbor
                # We'll do a backtracking search with randomization
                
                success = False
                for attempt in range(50):  # Limited attempts
                    if time.time() > time_budget:
                        break
                    # Random order
                    order = to_recolor[:]
                    random.shuffle(order)
                    
                    temp = current[:]
                    valid = True
                    for i in order:
                        taken = set()
                        for j in conflict_sets[i]:
                            if temp[j] != -1 and temp[j] < target_max:
                                taken.add(temp[j])
                        # Try to find a color < target_max not in taken
                        avail = [c for c in range(target_max) if c not in taken]
                        if avail:
                            # Pick randomly from available for diversity
                            temp[i] = random.choice(avail)
                        else:
                            valid = False
                            break
                    if valid:
                        # Verify full validity
                        ok = True
                        for i in range(m):
                            ci = temp[i]
                            for j in conflict_sets[i]:
                                if temp[j] == ci:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            current = temp[:]
                            success = True
                            break
                if not success:
                    break  # Can't eliminate this color, stop reducing
            
            return current

        # Apply local search to reduce colors
        if best_num > 1:
            # Try to reduce by 1, then by 2, etc.
            for target in range(best_num - 1, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                reduced = try_reduce_to(target, best_colors, deadline - 5.0)
                if reduced is not None:
                    new_max = max(reduced) + 1 if reduced else 0
                    if new_max < best_num:
                        best_colors = reduced
                        best_num = new_max
                    else:
                        break

        # Phase 3: ILP for final improvement
        # Only try ILP if we have time and if local search didn't reach the conjecture bound
        conjecture_bound = int(5 * d * d / 4) + 1
        if best_num > conjecture_bound and time.time() < deadline - 10.0:
            # Try ILP to see if we can do better
            def try_ilp(K, time_budget):
                if K <= 0:
                    return None
                
                n_vars = m * K
                c = np.zeros(n_vars)
                lb = np.zeros(n_vars)
                ub = np.ones(n_vars)
                integrality = np.ones(n_vars)
                
                A_rows = []
                lb_cons = []
                ub_cons = []
                
                # Constraint 1: Each edge gets exactly one color
                for i in range(m):
                    row = np.zeros(n_vars)
                    for k in range(K):
                        row[i * K + k] = 1.0
                    A_rows.append(row)
                    lb_cons.append(1.0)
                    ub_cons.append(1.0)
                
                # Constraint 2: Conflicts
                for i in range(m):
                    for j in conflict_sets[i]:
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
                               options={"time_limit": time_budget, "presolve": True})
                    
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
                        # Verify
                        valid = True
                        for i in range(m):
                            ci = colors[i]
                            for j in conflict_sets[i]:
                                if colors[j] == ci:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            return colors
                        else:
                            return None
                    else:
                        return None
                except Exception:
                    return None

            # Try K = best_num - 1, then best_num - 2, etc.
            for K in range(best_num - 1, max(1, conjecture_bound - 1), -1):
                if time.time() > deadline - 5.0:
                    break
                remaining = deadline - 5.0 - time.time()
                if remaining < 2.0:
                    break
                time_for_ilp = min(remaining, 25.0)
                result = try_ilp(K, time_for_ilp)
                if result is not None:
                    new_max = max(result) + 1 if result else 0
                    if new_max < best_num:
                        best_colors = result
                        best_num = new_max
                else:
                    # If ILP failed, we don't know if K is possible or not
                    # Stop trying lower Ks
                    break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
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
        
        # Allocate time: give more to larger instances
        # n=40: ~30s, n=60: ~35s, n=80: ~40s
        if n == 40:
            time_limit = min(30.0, remaining - 2.0)
        elif n == 60:
            time_limit = min(35.0, remaining - 2.0)
        else:
            time_limit = min(40.0, remaining - 2.0)
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)

    return results
