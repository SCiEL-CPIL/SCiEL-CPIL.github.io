"""Strong edge colouring: greedy with multiple orderings + local search (swap + move)."""
import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        """Build the strong edge-colouring conflict graph.
        Two edges conflict if they share a vertex or are joined by an edge.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # edge index
        edge_set = set(edges)
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}

        # conflict[i] = set of j != i such that edges[i] and edges[j] conflict
        conflict = [set() for _ in range(m)]

        for i, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for w in adj[u]:
                e2 = (min(u, w), max(u, w))
                if e2 in edge_set:
                    j = edge_idx[e2]
                    if j != i:
                        conflict[i].add(j)
            for w in adj[v]:
                e2 = (min(v, w), max(v, w))
                if e2 in edge_set:
                    j = edge_idx[e2]
                    if j != i:
                        conflict[i].add(j)
            # Edges joined by an edge: for each neighbor w of u, and each edge (w, x), that edge conflicts with (u,v)
            # Actually "joined by an edge" means there exists an edge connecting an endpoint of one to an endpoint of the other.
            # If edges (u,v) and (w,x) are joined by an edge, it means some endpoint of (u,v) is adjacent to some endpoint of (w,x).
            # That is: u~w, u~x, v~w, or v~x.
            # We already covered shared vertices. Now cover the case where no shared vertex but there's an edge between them.
            for w in adj[u]:
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_set:
                        j = edge_idx[e2]
                        if j != i:
                            conflict[i].add(j)
            for w in adj[v]:
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_set:
                        j = edge_idx[e2]
                        if j != i:
                            conflict[i].add(j)

        return conflict, edge_idx

    def greedy_color(edges, conflict, order, rng):
        """Greedy first-fit in the given order. Returns colour assignment dict edge->colour."""
        m = len(edges)
        colour = [-1] * m
        # colour_classes[c] = set of edge indices with colour c
        colour_classes = defaultdict(set)
        # For each edge, track which colours are forbidden based on already-coloured neighbours
        # We'll use a simpler approach: for each edge in order, check all colours

        # Precompute: for each edge i, the set of conflicting edges
        # Use vertex-forbidden sets for speed
        # forbidden[i] = set of colours that edge i cannot take (based on already placed edges)
        # Actually, let's just do it straightforwardly but efficiently.

        for idx in order:
            e = edges[idx]
            # Find which colours are taken by conflicting already-coloured edges
            taken = set()
            for j in conflict[idx]:
                if colour[j] != -1:
                    taken.add(colour[j])
            # Assign the smallest available colour
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c
            colour_classes[c].add(idx)

        return colour, colour_classes

    def local_search(edges, conflict, colour, colour_classes, rng, max_iters=2000):
        """Try to reduce colour count via moves and swaps."""
        m = len(edges)
        max_colour = max(colour) + 1 if m > 0 else 0
        improved = True
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try to move an edge from a higher colour class to a lower one
            for c in range(max_colour - 1, 0, -1):
                if not colour_classes[c]:
                    continue
                # Try each edge in this colour class
                for idx in list(colour_classes[c]):
                    # Check if we can move this edge to any colour < c
                    taken = set()
                    for j in conflict[idx]:
                        if colour[j] != -1 and colour[j] != c:
                            taken.add(colour[j])
                    # Also, the edge itself has colour c, so we need a colour not in taken and != c
                    for c2 in range(c):
                        if c2 not in taken:
                            # Move idx from c to c2
                            colour_classes[c].discard(idx)
                            colour[idx] = c2
                            colour_classes[c2].add(idx)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Try swap: two edges in different colour classes swap colours
            # This is more expensive, so do it less frequently
            if not improved and iters % 5 == 0:
                # Try to find a swap that reduces max colour or merges classes
                # Simple approach: for each pair of colours c1 < c2, try to find edges i in c1, j in c2
                # such that after swap, both are valid.
                # This is expensive, so limit the search.
                found_swap = False
                for c1 in range(max_colour):
                    if not colour_classes[c1]:
                        continue
                    for c2 in range(c1 + 1, max_colour):
                        if not colour_classes[c2]:
                            continue
                        # Try a few random pairs
                        list1 = list(colour_classes[c1])
                        list2 = list(colour_classes[c2])
                        if len(list1) > 10:
                            list1 = rng.sample(list1, 10)
                        if len(list2) > 10:
                            list2 = rng.sample(list2, 10)
                        for i in list1:
                            for j in list2:
                                # Check if swapping is valid
                                # After swap: i gets c2, j gets c1
                                # i must not conflict with any edge in c2 (other than j, which will leave c2)
                                # j must not conflict with any edge in c1 (other than i, which will leave c1)
                                ok = True
                                for k in conflict[i]:
                                    if colour[k] == c2 and k != j:
                                        ok = False
                                        break
                                if ok:
                                    for k in conflict[j]:
                                        if colour[k] == c1 and k != i:
                                            ok = False
                                            break
                                if ok:
                                    # Perform swap
                                    colour_classes[c1].discard(i)
                                    colour_classes[c2].discard(j)
                                    colour[i] = c2
                                    colour[j] = c1
                                    colour_classes[c2].add(i)
                                    colour_classes[c1].add(j)
                                    improved = True
                                    found_swap = True
                                    break
                            if found_swap:
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    # Update max_colour
                    max_colour = max(colour) + 1 if m > 0 else 0

            # Update max_colour
            max_colour = max(colour) + 1 if m > 0 else 0

        return colour, colour_classes, max_colour

    def solve_instance(n, d, seed, deadline):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        conflict, edge_idx = build_conflict(edges, n)

        best_colour = None
        best_max_colour = m  # worst case

        # Try multiple random orderings
        base_order = list(range(m))

        # First, try a few deterministic orderings
        orderings = [
            base_order,  # natural order
            sorted(base_order, key=lambda i: len(conflict[i])),  # least constrained first
            sorted(base_order, key=lambda i: -len(conflict[i])),  # most constrained first
        ]

        rng = random.Random(seed)

        for ordering in orderings:
            if time.time() > deadline:
                break
            colour, colour_classes = greedy_color(edges, conflict, ordering, rng)
            max_c = max(colour) + 1 if m > 0 else 0

            # Local search
            colour, colour_classes, max_c = local_search(edges, conflict, colour, colour_classes, rng, max_iters=500)

            if max_c < best_max_colour:
                best_max_colour = max_c
                best_colour = colour[:]

        # Try more random orderings if time permits
        attempts = 0
        while time.time() < deadline - 5.0:
            order = base_order[:]
            rng.shuffle(order)
            colour, colour_classes = greedy_color(edges, conflict, order, rng)
            max_c = max(colour) + 1 if m > 0 else 0

            colour, colour_classes, max_c = local_search(edges, conflict, colour, colour_classes, rng, max_iters=300)

            if max_c < best_max_colour:
                best_max_colour = max_c
                best_colour = colour[:]
            attempts += 1

        if best_colour is None:
            best_colour = [0] * m

        return [best_colour[edge_idx[e]] for e in edges]

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 10.0:
            # Not enough time for remaining instances; use a quick greedy
            edges = gen_instance(*spec)
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            conflict, edge_idx = build_conflict(edges, spec[0])
            colour, _ = greedy_color(edges, conflict, list(range(m)), random.Random(spec[2]))
            results.append([colour[edge_idx[e]] for e in edges])
        else:
            results.append(solve_instance(spec[0], spec[1], spec[2], DEADLINE - 10.0))

    return results
