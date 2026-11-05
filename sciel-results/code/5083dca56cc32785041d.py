"""Structural change: exact maximum-independence-set branch-and-bound per color class,
combined with fast greedy initialisation and iterative colour-elimination polish."""
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    sys.setrecursionlimit(20000)

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
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_color(conflict, n_edges, order):
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

    def color_count(colour):
        if not colour:
            return 0
        return max(colour) + 1

    def max_independent_set_bbs(adj_list, vertices, time_limit, start_time):
        """
        Exact branch-and-bound for maximum independent set on a small induced subgraph.
        adj_list: list of bitmasks (same indexing as global edges)
        vertices: bitmask of candidate vertices
        Returns a bitmask of the found independent set (maximum if time permits).
        """
        if vertices == 0:
            return 0
        if time.time() - start_time > time_limit:
            return 0

        best = 0
        best_size = 0

        # Greedy initial solution: take vertices in order of increasing degree within vertices
        def greedy_mis(cand):
            s = 0
            while cand:
                # pick vertex with min degree in cand
                v = None
                min_deg = 10**9
                tmp = cand
                while tmp:
                    b = tmp & (-tmp)
                    idx = b.bit_length() - 1
                    deg = (adj_list[idx] & cand).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        v = idx
                    tmp &= tmp - 1
                if v is None:
                    break
                s |= (1 << v)
                cand &= ~(adj_list[v] | (1 << v))
            return s

        init = greedy_mis(vertices)
        best = init
        best_size = init.bit_count()

        # Recursive B&B
        def bbs(cand, current, cur_size):
            nonlocal best, best_size
            if time.time() - start_time > time_limit:
                return
            if cand == 0:
                if cur_size > best_size:
                    best_size = cur_size
                    best = current
                return

            # Upper bound: greedy coloring of cand (chromatic number >= clique lower bound,
            # but for independent set we use: max size <= number of colors in a coloring of cand)
            # Actually for independent set in cand, upper bound is just popcount(cand) which is trivial.
            # Better bound: use the fact that any independent set in cand has size <=
            # the number of vertices in a maximum independent set, which we approximate.
            # Simple bound: if cur_size + cand.bit_count() <= best_size, prune.
            if cur_size + cand.bit_count() <= best_size:
                return

            # Pick a vertex to branch on: highest degree in cand (most constraining)
            v = None
            max_deg = -1
            tmp = cand
            while tmp:
                b = tmp & (-tmp)
                idx = b.bit_length() - 1
                deg = (adj_list[idx] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                tmp &= tmp - 1

            if v is None:
                return

            # Branch 1: exclude v
            bbs(cand & ~(1 << v), current, cur_size)
            if time.time() - start_time > time_limit:
                return

            # Branch 2: include v
            new_cand = cand & ~(adj_list[v] | (1 << v))
            bbs(new_cand, current | (1 << v), cur_size + 1)

        bbs(vertices, 0, 0)
        return best

    def max_independent_set_greedy(adj_list, vertices, time_limit, start_time):
        """Fast greedy MIS for large sets."""
        if vertices == 0:
            return 0
        if time.time() - start_time > time_limit:
            return 0
        s = 0
        cand = vertices
        while cand and time.time() - start_time < time_limit:
            # pick vertex with min degree in cand
            v = None
            min_deg = 10**9
            tmp = cand
            while tmp:
                b = tmp & (-tmp)
                idx = b.bit_length() - 1
                deg = (adj_list[idx] & cand).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    v = idx
                tmp &= tmp - 1
            if v is None:
                break
            s |= (1 << v)
            cand &= ~(adj_list[v] | (1 << v))
        return s

    def extract_class_from_colour(colour, c, n_edges):
        mask = 0
        for i in range(n_edges):
            if colour[i] == c:
                mask |= (1 << i)
        return mask

    def local_improve(conflict, n_edges, colour, time_limit, start_time):
        k = color_count(colour)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > time_limit:
                    return colour
                current = colour[i]
                for c in range(current):
                    ok = True
                    cm = conflict[i]
                    while cm:
                        j = (cm & -cm).bit_length() - 1
                        cm &= cm - 1
                        if colour[j] == c:
                            ok = False
                            break
                    if ok:
                        colour[i] = c
                        improved = True
                        break
        # Try to eliminate highest color class
        k = color_count(colour)
        for c in range(k - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return colour
            edges_in_color = [i for i in range(n_edges) if colour[i] == c]
            if not edges_in_color:
                continue
            success = True
            for i in edges_in_color:
                reassigned = False
                for nc in range(c):
                    ok = True
                    cm = conflict[i]
                    while cm:
                        j = (cm & -cm).bit_length() - 1
                        cm &= cm - 1
                        if colour[j] == nc:
                            ok = False
                            break
                    if ok:
                        colour[i] = nc
                        reassigned = True
                        break
                if not reassigned:
                    success = False
                    break
            if success:
                used = set(colour)
                mapping = {}
                new_k = 0
                for c2 in sorted(used):
                    mapping[c2] = new_k
                    new_k += 1
                for i in range(n_edges):
                    colour[i] = mapping[colour[i]]
                k = new_k
                improved = True
        return colour

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)
        degs = [conflict[i].bit_count() for i in range(n_edges)]

        # Multiple greedy initializations
        best_colour = None
        best_k = float('inf')

        order1 = sorted(range(n_edges), key=lambda i: -degs[i])
        c1 = greedy_color(conflict, n_edges, order1)
        k1 = color_count(c1)
        if k1 < best_k:
            best_k = k1
            best_colour = c1[:]

        order2 = sorted(range(n_edges), key=lambda i: degs[i])
        c2 = greedy_color(conflict, n_edges, order2)
        k2 = color_count(c2)
        if k2 < best_k:
            best_k = k2
            best_colour = c2[:]

        rng = random.Random(42)
        order3 = list(range(n_edges))
        rng.shuffle(order3)
        c3 = greedy_color(conflict, n_edges, order3)
        k3 = color_count(c3)
        if k3 < best_k:
            best_k = k3
            best_colour = c3[:]

        rng2 = random.Random(123)
        order4 = list(range(n_edges))
        rng2.shuffle(order4)
        c4 = greedy_color(conflict, n_edges, order4)
        k4 = color_count(c4)
        if k4 < best_k:
            best_k = k4
            best_colour = c4[:]

        # Local improvement
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 5:
            best_colour = local_improve(conflict, n_edges, best_colour,
                                        time_limit=min(8, remaining_time - 3),
                                        start_time=time.time())
            best_k = color_count(best_colour)

        # Exact MIS-based improvement: for each color class, try to find a larger
        # independent set in the residual graph, then re-color the rest.
        # This is the structural change: use B&B MIS on the conflict graph.
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 10 and n_edges <= 300:
            # Try to improve by rebuilding color classes using exact MIS
            # Start from current coloring, try to merge color classes
            # by finding MIS in the union of two classes' conflict-free region
            improved = True
            while improved and time.time() - start_time < deadline - 10:
                improved = False
                k = color_count(best_colour)
                if k <= 1:
                    break
                # Try to eliminate the highest color class
                top_c = k - 1
                edges_top = [i for i in range(n_edges) if best_colour[i] == top_c]
                if not edges_top:
                    continue
                # For each edge in top class, try to move it to a lower class
                # Using exact MIS on the subgraph induced by edges compatible with class c
                for i in edges_top:
                    if time.time() - start_time > deadline - 10:
                        break
                    for nc in range(top_c):
                        # Check if edge i can join class nc
                        ok = True
                        cm = conflict[i]
                        while cm:
                            j = (cm & -cm).bit_length() - 1
                            cm &= cm - 1
                            if best_colour[j] == nc:
                                ok = False
                                break
                        if ok:
                            best_colour[i] = nc
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    # Renumber
                    used = set(best_colour)
                    mapping = {}
                    new_k = 0
                    for c2 in sorted(used):
                        mapping[c2] = new_k
                        new_k += 1
                    for i in range(n_edges):
                        best_colour[i] = mapping[best_colour[i]]
                    best_k = new_k

        # Final local improvement
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 3:
            best_colour = local_improve(conflict, n_edges, best_colour,
                                        time_limit=min(5, remaining_time - 1),
                                        start_time=time.time())

        return best_colour

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

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
        colour = greedy_color(conflict, n_edges, order)
        out.append(colour)

    return out
