import time
import sys
import math
from random import Random
from collections import deque

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

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

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

        # Build conflict graph on edges
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

        deg = [len(neigh[i]) for i in range(m)]

        # Bitset representation of conflict graph for fast MIS
        # conflict_mask[i] is a Python int with bit j set if i conflicts with j
        conflict_mask = [0] * m
        for i in range(m):
            for j in neigh[i]:
                if j > i:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)

        def find_max_independent_set(available_mask, time_limit):
            """
            Find a maximum independent set in the subgraph induced by available_mask
            using branch-and-bound with bitsets.
            Returns the set of vertices as a list.
            """
            # If time is up, return a greedy solution
            if time.time() > time_limit:
                # Greedy: pick vertices with lowest degree in remaining
                remaining = [i for i in range(m) if (available_mask >> i) & 1]
                remaining.sort(key=lambda x: bin(conflict_mask[x] & available_mask).count('1'))
                result = []
                for v in remaining:
                    if (available_mask >> v) & 1:
                        result.append(v)
                        available_mask &= ~(1 << v)
                        # Remove all neighbors of v
                        for u in range(m):
                            if (available_mask >> u) & 1 and (conflict_mask[v] >> u) & 1:
                                available_mask &= ~(1 << u)
                return result

            best_set = []
            best_size = 0
            nodes_explored = 0

            # Order vertices by degree for better pruning
            # Use a simple heuristic: process vertices with highest degree first in branching
            vertex_order = list(range(m))
            vertex_order.sort(key=lambda x: -bin(conflict_mask[x] & available_mask).count('1'))

            def bb(current_set, current_mask, lower_bound):
                nonlocal best_set, best_size, nodes_explored

                if time.time() > time_limit:
                    return

                nodes_explored += 1

                # Upper bound: size of current set + number of available vertices
                avail_count = bin(current_mask).count('1')
                if len(current_set) + avail_count <= best_size:
                    return

                # If no more vertices, update best
                if current_mask == 0:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                    return

                # Simple upper bound using greedy coloring of the subgraph
                # For now, just use the count of available vertices as upper bound
                # We can improve this with a better bound

                # Choose a vertex to branch on
                # Pick vertex with highest degree in the remaining graph
                v = -1
                max_deg = -1
                for u in vertex_order:
                    if (current_mask >> u) & 1:
                        d = bin(conflict_mask[u] & current_mask).count('1')
                        if d > max_deg:
                            max_deg = d
                            v = u

                if v == -1:
                    return

                # Branch 1: include v in the independent set
                new_mask = current_mask & ~(1 << v)
                # Remove all neighbors of v
                for u in range(m):
                    if (new_mask >> u) & 1 and (conflict_mask[v] >> u) & 1:
                        new_mask &= ~(1 << u)
                current_set.append(v)
                bb(current_set, new_mask, lower_bound + 1)
                current_set.pop()

                # Branch 2: exclude v
                current_mask &= ~(1 << v)
                bb(current_set, current_mask, lower_bound)

            bb([], available_mask, 0)
            return best_set

        # Greedy initial coloring
        def greedy_color():
            color = [-1] * m
            for i in range(m):
                forbidden = set()
                for j in neigh[i]:
                    if color[j] != -1:
                        forbidden.add(color[j])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            return color

        color0 = greedy_color()
        K0 = max(color0) + 1

        rng = Random(seed * 7919 + 42)

        # Start with K0 colors and try to reduce
        current_color = color0[:]
        current_K = K0

        # Lower bound: maximum degree + 1 in conflict graph
        lb = max(deg) + 1 if deg else 1

        # Try to reduce K from K0 down to lb
        for target_K in range(current_K - 1, lb - 1, -1):
            if time.time() > deadline - 3.0:
                break

            success = False
            num_attempts = 20
            for attempt in range(num_attempts):
                if time.time() > deadline - 3.0:
                    break

                # Try to find a valid target_K coloring using MIS-based approach
                # This is a heuristic: greedily build color classes as maximum independent sets
                trial_color = [-1] * m
                remaining = (1 << m) - 1  # All vertices available
                k_used = 0

                for k in range(target_K):
                    if time.time() > deadline - 3.0:
                        break
                    if remaining == 0:
                        break

                    # Find a large independent set in the remaining graph
                    # Use a greedy heuristic for speed
                    avail_list = [i for i in range(m) if (remaining >> i) & 1]
                    if not avail_list:
                        break

                    # Sort by degree in remaining subgraph (ascending - pick low degree first)
                    avail_list.sort(key=lambda x: bin(conflict_mask[x] & remaining).count('1'))

                    color_class = []
                    temp_remaining = remaining
                    for v in avail_list:
                        if (temp_remaining >> v) & 1:
                            color_class.append(v)
                            temp_remaining &= ~(1 << v)
                            # Remove neighbors
                            for u in range(m):
                                if (temp_remaining >> u) & 1 and (conflict_mask[v] >> u) & 1:
                                    temp_remaining &= ~(1 << u)

                    for v in color_class:
                        trial_color[v] = k

                    remaining &= ~((1 << len(color_class)) - 1) if not color_class else 0
                    # Actually, we need to remove the color_class from remaining properly
                    for v in color_class:
                        remaining &= ~(1 << v)

                    k_used += 1

                # Check if we colored all vertices
                if all(c >= 0 for c in trial_color):
                    # Verify it's valid
                    valid = True
                    for i in range(m):
                        for j in neigh[i]:
                            if j > i and trial_color[i] == trial_color[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current_color = trial_color
                        current_K = target_K
                        success = True
                        break

            if not success:
                break

        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 130.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 40.0)
            share = max(share, 1.0)
            instance_deadline = start + (time.time() - start) + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
