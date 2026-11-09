"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = 20
    num_colors = 20

    # A color class must be a matching (no two edges of same color share a vertex),
    # and for any two same-colored disjoint edges (u,v) and (x,y), the 4-cycle
    # formed with cross edges must not be completed.
    #
    # Greedy: add edges in some order, for each edge try colors; a color c is valid
    # for (u,v) if:
    #   1) no neighbor of u or v already uses color c (matching property)
    #   2) for every existing edge (x,y) with color c, it is NOT the case that
    #      both cross-edges (v,x)&(y,u) are present, and NOT both (v,y)&(x,u) present.

    def build_graph(edge_order, rng):
        """Build a valid coloring greedily over the given edge order."""
        # color_map: (u,v) with u<v -> color
        color_map = {}
        # For fast checks:
        # adj: set of neighbors
        adj = [set() for _ in range(n)]
        # color_edges[c]: list of (u,v) edges with color c (u<v)
        color_edges = [set() for _ in range(num_colors)]
        # edge_present: set of (u,v) u<v
        edge_present = set()

        for (u, v) in edge_order:
            if u > v:
                u, v = v, u
            if (u, v) in edge_present:
                continue
            # Try colors in a randomized order
            colors = list(range(num_colors))
            rng.shuffle(colors)
            chosen = -1
            for c in colors:
                # 1) matching property: no neighbor of u or v has color c
                ok = True
                for nb in adj[u]:
                    if color_map[(min(u, nb), max(u, nb))] == c:
                        ok = False
                        break
                if not ok:
                    continue
                for nb in adj[v]:
                    if color_map[(min(v, nb), max(v, nb))] == c:
                        ok = False
                        break
                if not ok:
                    continue
                # 2) 4-cycle completion check with existing color-c edges
                for (x, y) in color_edges[c]:
                    # (x,y) is disjoint from (u,v) by matching property (no shared vertex)
                    # 4-cycle u-v-x-y-u needs (v,x) and (y,u)
                    # 4-cycle u-v-y-x-u needs (v,y) and (x,u)
                    # Check if either cross pair is fully present
                    e_vx = (min(v, x), max(v, x))
                    e_yu = (min(y, u), max(y, u))
                    if e_vx in edge_present and e_yu in edge_present:
                        ok = False
                        break
                    e_vy = (min(v, y), max(v, y))
                    e_xu = (min(x, u), max(x, u))
                    if e_vy in edge_present and e_xu in edge_present:
                        ok = False
                        break
                if ok:
                    chosen = c
                    break
            if chosen >= 0:
                color_map[(u, v)] = chosen
                edge_present.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                color_edges[chosen].add((u, v))

        result = []
        for (u, v), c in color_map.items():
            result.append([u, v, c])
        return result

    def validate(edges):
        """Full validation: every 4-cycle is rainbow. Returns True if valid."""
        color_map = {}
        for u, v, c in edges:
            if u > v:
                u, v = v, u
            if (u, v) in color_map:
                return False  # duplicate edge
            color_map[(u, v)] = c
        adj = [set() for _ in range(n)]
        for (u, v) in color_map:
            adj[u].add(v)
            adj[v].add(u)
        # Enumerate all 4-cycles: a<b<c<d, check all 3 pairings
        for a in range(n):
            for b in range(a + 1, n):
                if b not in adj[a]:
                    continue
                for c in range(b + 1, n):
                    if c not in adj[b]:
                        continue
                    for d in range(c + 1, n):
                        if d not in adj[c]:
                            continue
                        # Need a closing edge back to a
                        # 4-cycle a-b-c-d-a: need (d,a)
                        if d in adj[a]:
                            cols = [color_map[(a, b)], color_map[(b, c)],
                                    color_map[(c, d)], color_map[(a, d)]]
                            if len(set(cols)) != 4:
                                return False
        return True

    # Generate edge orders with different strategies
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def make_orders(rng):
        orders = []
        # Random order
        o = list(all_edges)
        rng.shuffle(o)
        orders.append(o)
        # Reverse random
        o2 = list(o)
        o2.reverse()
        orders.append(o2)
        # Sorted by some heuristic: prefer edges that are "spread out"
        # Use a random permutation of vertices to define a natural order
        perm = list(range(n))
        rng.shuffle(perm)
        pos = {v: i for i, v in enumerate(perm)}
        o3 = sorted(all_edges, key=lambda e: (pos[e[0]] + pos[e[1]]) % n)
        orders.append(o3)
        # Another permutation
        perm2 = list(range(n))
        rng.shuffle(perm2)
        pos2 = {v: i for i, v in enumerate(perm2)}
        o4 = sorted(all_edges, key=lambda e: abs(pos2[e[0]] - pos2[e[1]]))
        orders.append(o4)
        return orders

    best_edges = []
    best_len = 0
    rng = random.Random(12345)

    # First, try a few deterministic-ish orders to get a baseline quickly
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        orders = make_orders(rng)
        for order in orders:
            edges = build_graph(order, rng)
            if len(edges) > best_len:
                # Validate before accepting
                if validate(edges):
                    best_edges = edges
                    best_len = len(edges)
                    if best_len >= 160:
                        break
        if best_len >= 160:
            break

    # Final validation of best
    if best_edges and not validate(best_edges):
        # Should not happen, but safety
        return []

    return best_edges
