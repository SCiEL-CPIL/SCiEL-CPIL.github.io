import time
import math
import numpy as np
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

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph on edges (two edges conflict if they share a vertex
        # or are joined by an edge in the original graph)
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share a vertex?
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    # Joined by an edge?
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        # Precompute conflict adjacency as sets for fast lookup
        neigh_set = [set(x) for x in neigh]
        deg = np.array([len(x) for x in neigh], dtype=np.int32)
        max_deg = int(deg.max()) if m > 0 else 0

        # DSATUR initial coloring
        def dsatur():
            color = [-1] * m
            color_sets = [set() for _ in range(m)]
            uncolored = set(range(m))
            while uncolored:
                best = max(uncolored, key=lambda v: (len(color_sets[v]), deg[v]))
                forbidden = set()
                for u in neigh[best]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[best] = c
                for u in neigh[best]:
                    if color[u] != -1:
                        color_sets[u].add(c)
                uncolored.discard(best)
            return color

        color0 = dsatur()
        K0 = max(color0) + 1

        # Count conflicts in a coloring
        def count_conflicts(color):
            total = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and color[i] == color[j]:
                        total += 1
            return total

        # Local search: try to fix a coloring with K colors
        # Use systematic vertex reassignment with conflict minimization
        def local_search_fix(color, K, rng, time_limit):
            """Try to fix coloring to be valid with K colors."""
            color = color[:]
            conflicts = count_conflicts(color)
            if conflicts == 0:
                return color

            best_color = color[:]
            best_conflicts = conflicts

            # Build list of vertices involved in conflicts
            conflict_verts = set()
            for i in range(m):
                for j in neigh[i]:
                    if color[i] == color[j]:
                        conflict_verts.add(i)
                        conflict_verts.add(j)

            max_iters = 50000
            no_improve = 0
            threshold = 5000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color

                if no_improve > threshold:
                    # Restart with perturbation
                    for i in conflict_verts:
                        if rng.random() < 0.3:
                            color[i] = rng.randrange(K)
                    conflicts = count_conflicts(color)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_color = color[:]
                    no_improve = 0
                    conflict_verts = set()
                    for i in range(m):
                        for j in neigh[i]:
                            if color[i] == color[j]:
                                conflict_verts.add(i)
                                conflict_verts.add(j)
                    continue

                if not conflict_verts:
                    break

                # Pick a random vertex from conflict set
                v = rng.choice(list(conflict_verts))
                old_c = color[v]

                # Try each color, pick the one that minimizes new conflicts
                best_new_c = old_c
                best_delta = 0
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    # Count how many neighbors have new_c vs old_c
                    add = sum(1 for u in neigh[v] if color[u] == new_c)
                    rem = sum(1 for u in neigh[v] if color[u] == old_c)
                    delta = add - rem
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c

                if best_new_c == old_c:
                    # No improvement possible for this vertex, try random
                    if rng.random() < 0.5:
                        new_c = rng.randrange(K)
                        if new_c != old_c:
                            add = sum(1 for u in neigh[v] if color[u] == new_c)
                            rem = sum(1 for u in neigh[v] if color[u] == old_c)
                            delta = add - rem
                            color[v] = new_c
                            conflicts += delta
                            no_improve += 1
                    continue

                # Apply the best move
                add = sum(1 for u in neigh[v] if color[u] == best_new_c)
                rem = sum(1 for u in neigh[v] if color[u] == old_c)
                delta = add - rem
                color[v] = best_new_c
                conflicts += delta

                if delta < 0:
                    no_improve = 0
                else:
                    no_improve += 1

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color = color[:]

                # Update conflict_verts
                if conflicts < 100:
                    conflict_verts = set()
                    for i in range(m):
                        for j in neigh[i]:
                            if color[i] == color[j]:
                                conflict_verts.add(i)
                                conflict_verts.add(j)
                else:
                    # Just update locally
                    if old_c != best_new_c:
                        # Check if v is still in conflict
                        still_conflict = any(color[u] == best_new_c for u in neigh[v])
                        if not still_conflict:
                            conflict_verts.discard(v)
                        else:
                            conflict_verts.add(v)
                        # Neighbors might have changed status
                        for u in neigh[v]:
                            u_conflict = any(color[w] == color[u] for w in neigh[u])
                            if u_conflict:
                                conflict_verts.add(u)
                            else:
                                conflict_verts.discard(u)

            if best_conflicts == 0:
                return best_color
            return None

        # Try to reduce number of colors by removing one color at a time
        def try_remove_color(color, K, rng, time_limit):
            """Try to recolor all vertices with color K-1 to other colors."""
            if K <= 1:
                return color, K

            target_color = K - 1
            # Get vertices with target color
            verts_to_recolor = [i for i in range(m) if color[i] == target_color]
            if not verts_to_recolor:
                return color, K - 1

            # Try to recolor these vertices using local search
            # First, create a working copy
            work_color = color[:]

            # For each vertex, try to find a valid new color
            # Use a priority: vertices with fewer options first
            def options_for(v, c):
                """Number of valid colors for vertex v (excluding c)."""
                forbidden = set()
                for u in neigh[v]:
                    if c[u] != target_color or u not in verts_to_recolor:
                        forbidden.add(c[u])
                    elif u != v:
                        # u is also being recolored, so its color is uncertain
                        # But for now, treat it as potentially any color
                        pass
                available = 0
                for cc in range(K - 1):
                    if cc not in forbidden:
                        available += 1
                return available

            # Greedy: recolor vertices in order of fewest options
            order = sorted(verts_to_recolor, key=lambda v: options_for(v, work_color))

            success = True
            for v in order:
                forbidden = set()
                for u in neigh[v]:
                    if u not in verts_to_recolor:
                        forbidden.add(work_color[u])
                    else:
                        # u is being recolored, check if already recolored
                        if work_color[u] != target_color:
                            forbidden.add(work_color[u])

                found = False
                for cc in range(K - 1):
                    if cc not in forbidden:
                        work_color[v] = cc
                        found = True
                        break

                if not found:
                    # Try local search to free up a color
                    # This is where it gets complex; for now, fail
                    success = False
                    break

            if success:
                return work_color, K - 1
            return color, K

        # Main search: start with DSATUR, then try to reduce colors
        best_color = color0[:]
        best_K = K0
        rng = Random(seed * 7919 + 42)

        # Phase 1: Try to fix and reduce at current K
        for target_K in range(best_K, max(2, max_deg), -1):
            if time.time() > deadline - 2.0:
                break

            # Try multiple restarts at this K
            found = False
            max_restarts = 30
            for attempt in range(max_restarts):
                if time.time() > deadline - 1.0:
                    break

                if attempt == 0:
                    # Start from current best, clamp colors
                    trial = [min(c, target_K - 1) for c in best_color]
                elif attempt < 10:
                    # Perturb
                    trial = [min(c, target_K - 1) for c in best_color]
                    for i in range(m):
                        if rng.random() < 0.2:
                            trial[i] = rng.randrange(target_K)
                else:
                    # Random start
                    trial = [rng.randrange(target_K) for _ in range(m)]

                # Run local search
                remaining = deadline - 1.0 - time.time()
                if remaining < 0.5:
                    break
                time_slice = min(remaining / (max_restarts - attempt) * 0.8, 3.0)
                result = local_search_fix(trial, target_K, rng, time.time() + time_slice)

                if result is not None:
                    best_color = result
                    best_K = target_K
                    found = True
                    break

            if not found:
                # Can't achieve this K, try lower
                continue

        return compress(best_color)

    try:
        start = time.time()
        total_budget = 140.0  # leave 10s margin for 150s kill
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        if time.time() > start + 145.0:
            return out

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
