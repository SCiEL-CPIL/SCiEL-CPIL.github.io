"""Rewritten program: exact maximum independent set per colour class via branch-and-bound."""
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

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

    def build_conflict(edges, adj):
        n_edges = len(edges)
        # conflict[i] is a bitmask of edges that cannot share a colour with edge i
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # share a vertex?
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                # joined by an edge?
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def max_independent_set(conflict, available, time_limit=None, start_time=None):
        """Find a maximum independent set in the subgraph induced by 'available' bitmask.
        Uses branch-and-bound with bitset operations."""
        if start_time is None:
            start_time = time.time()
        if time_limit is None:
            time_limit = 5.0

        n = max(conflict) + 1 if conflict else 0
        # Precompute for each vertex its neighbours within the available set
        # We'll work with bitmasks of size up to ~480 bits (Python ints handle this)

        best_size = 0
        best_set = 0

        # Greedy initial solution: repeatedly pick vertex with min degree in remaining
        def greedy_mis(avail):
            nonlocal best_size, best_set
            s = 0
            count = 0
            while avail:
                # Pick vertex with fewest neighbours in avail
                best_v = -1
                best_deg = float('inf')
                tmp = avail
                while tmp:
                    v = (tmp & -tmp).bit_length() - 1
                    tmp &= tmp - 1
                    deg = (conflict[v] & avail).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                s |= (1 << best_v)
                count += 1
                avail &= ~(conflict[best_v] | (1 << best_v))
            if count > best_size:
                best_size = count
                best_set = s
            return s

        greedy_mis(available)

        # Branch and bound
        # Use a recursive approach with memoization on small sets
        def bbs(avail, depth):
            nonlocal best_size, best_set
            if time.time() - start_time > time_limit:
                return
            if avail == 0:
                return
            # Upper bound: |avail|
            upper = avail.bit_count()
            if upper <= best_size:
                return
            # Pick a vertex to branch on
            # Heuristic: pick vertex with max degree in avail for better pruning
            v = -1
            max_deg = -1
            tmp = avail
            while tmp:
                cv = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                deg = (conflict[cv] & avail).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = cv
            if v < 0:
                return

            # Branch 1: include v
            new_avail = avail & ~(conflict[v] | (1 << v))
            bbs(new_avail, depth + 1)

            # Branch 2: exclude v
            new_avail2 = avail & ~(1 << v)
            bbs(new_avail2, depth + 1)

        # For larger sets, use a more efficient approach:
        # Process vertices in order, using a stack-based DFS
        def efficient_bbs(avail):
            nonlocal best_size, best_set
            if time.time() - start_time > time_limit:
                return
            if avail == 0:
                return
            if avail.bit_count() <= best_size:
                return

            # Try to extend greedily first
            greedy_mis(avail)
            if time.time() - start_time > time_limit:
                return
            if avail.bit_count() <= best_size:
                return

            # Pick branching vertex
            v = -1
            max_deg = -1
            tmp = avail
            while tmp:
                cv = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                deg = (conflict[cv] & avail).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = cv
            if v < 0:
                return

            # Include v
            avail_inc = avail & ~(conflict[v] | (1 << v))
            efficient_bbs(avail_inc)

            # Exclude v
            avail_exc = avail & ~(1 << v)
            efficient_bbs(avail_exc)

        # Use the efficient version for the main search
        if available.bit_count() > 30:
            efficient_bbs(available)
        else:
            bbs(available, 0)

        return best_set

    def greedy_color(edges, conflict, order):
        """Greedy coloring with given order."""
        n_edges = len(edges)
        colour = [0] * n_edges
        for idx in order:
            taken = 0
            cm = conflict[idx]
            while cm:
                j = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                taken |= (1 << colour[j])
            c = 0
            while (taken >> c) & 1:
                c += 1
            colour[idx] = c
        return colour

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)

        # Compute degree in conflict graph for ordering
        degs = [conflict[i].bit_count() for i in range(n_edges)]

        # Initial greedy coloring (descending degree order)
        order = sorted(range(n_edges), key=lambda i: -degs[i])
        colour = greedy_color(edges, conflict, order)

        # Iterative exact MIS coloring
        remaining = (1 << n_edges) - 1  # all edges
        colour = [-1] * n_edges
        color_idx = 0

        while remaining and time.time() < start_time + time_budget:
            # Find maximum independent set in remaining
            mis = max_independent_set(conflict, remaining,
                                      time_limit=min(3.0, time_budget - (time.time() - start_time)),
                                      start_time=time.time())
            if mis == 0:
                # Fallback: just pick one edge
                e = (remaining & -remaining).bit_length() - 1
                mis = 1 << e

            # Assign this color class
            tmp = mis
            while tmp:
                e = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                colour[e] = color_idx

            remaining &= ~mis
            color_idx += 1

            # If time is running out, fall back to greedy for remaining
            if time.time() > start_time + time_budget - 5:
                break

        # Fill in any uncolored edges with greedy
        if remaining:
            uncolored = []
            tmp = remaining
            while tmp:
                e = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                uncolored.append(e)
            # Greedy on remaining
            for e in uncolored:
                taken = 0
                cm = conflict[e]
                while cm:
                    j = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if colour[j] >= 0:
                        taken |= (1 << colour[j])
                c = 0
                while (taken >> c) & 1:
                    c += 1
                colour[e] = c

        return colour

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Allocate time budget per instance
    total_budget = deadline - start_time
    per_instance = total_budget / 3.0

    for spec in specs:
        if time.time() > deadline - 3:
            break
        remaining_time = min(per_instance, deadline - time.time() - 2)
        if remaining_time < 5:
            remaining_time = 5
        colour = solve_instance(spec[0], spec[1], spec[2], remaining_time)
        out.append(colour)

    # Fallback for any missing
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflict = build_conflict(edges, adj)
        n_edges = len(edges)
        order = list(range(n_edges))
        colour = greedy_color(edges, conflict, order)
        out.append(colour)

    return out
