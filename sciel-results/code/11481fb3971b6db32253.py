import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

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

    def build_conflict(edges, n):
        m = len(edges)
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # conflict[i] = set of j that conflict with edge i
        conflict = [set() for _ in range(m)]
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        for i, (u, v) in enumerate(edges):
            # All edges sharing u or v, or joined by an edge
            # Neighbors of u and v (including u,v themselves for sharing)
            candidates = set(adj[u]) | set(adj[v]) | {u, v}
            for w in candidates:
                for x in adj[w]:
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_set:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
        return conflict

    def dsatur_color(conflict, m):
        color = [-1] * m
        neighbor_colors = [set() for _ in range(m)]
        uncolored = set(range(m))
        degree = [len(conflict[i]) for i in range(m)]

        for _ in range(m):
            if time.time() > deadline:
                break
            best_v = -1
            best_key = (-1, -1)
            for v in uncolored:
                sat = len(neighbor_colors[v])
                key = (sat, degree[v])
                if key > best_key:
                    best_key = key
                    best_v = v
            if best_v == -1:
                break
            used = neighbor_colors[best_v]
            c = 0
            while c in used:
                c += 1
            color[best_v] = c
            uncolored.discard(best_v)
            for w in conflict[best_v]:
                if w in uncolored:
                    neighbor_colors[w].add(c)

        for v in uncolored:
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            for w in conflict[v]:
                neighbor_colors[w].add(c)
        return color

    def greedy_color(conflict, m, order):
        color = [-1] * m
        neighbor_colors = [set() for _ in range(m)]
        for v in order:
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            for w in conflict[v]:
                neighbor_colors[w].add(c)
        return color

    def try_recolor(color, conflict, v, target):
        """Try to recolor vertex v to target color. Returns True if success."""
        if color[v] == target:
            return True
        # Check if target is available
        for w in conflict[v]:
            if color[w] == target:
                return False
        # Recolor
        color[v] = target
        return True

    def local_improve(color, conflict, m, max_time):
        """Iteratively try to reduce colors by recoloring and merging."""
        color = list(color)
        current_colors = max(color) + 1
        rng = Random(12345)

        while time.time() < max_time and current_colors > 0:
            improved = False
            # Try to remove the highest color
            top_color = current_colors - 1
            # Find vertices with top color
            top_vertices = [i for i in range(m) if color[i] == top_color]
            rng.shuffle(top_vertices)

            for v in top_vertices:
                if time.time() > max_time:
                    break
                # Try to recolor v to any lower color
                recolored = False
                for c in range(top_color):
                    if try_recolor(color, conflict, v, c):
                        recolored = True
                        improved = True
                        break
                if not recolored:
                    # Try k-exchange: find a vertex w with a color that v could take
                    # and v's color that w could take
                    for w in conflict[v]:
                        if color[w] < top_color and color[w] != -1:
                            cw = color[w]
                            # Can w take v's color (top_color)?
                            can_w_take = True
                            for x in conflict[w]:
                                if color[x] == top_color and x != v:
                                    can_w_take = False
                                    break
                            if can_w_take:
                                # Do the swap
                                color[v] = cw
                                color[w] = top_color
                                improved = True
                                break

            # After trying to recolor, check if top color is gone
            if any(color[i] == top_color for i in range(m)):
                if not improved:
                    # Try color merging: merge two colors
                    break
            else:
                current_colors -= 1

            # Also try random recoloring passes to free up colors
            if current_colors > 1 and time.time() < max_time:
                # Try to merge colors by recoloring all vertices of color A to color B
                # Find the smallest color class
                from collections import defaultdict
                color_classes = defaultdict(list)
                for i in range(m):
                    color_classes[color[i]].append(i)
                # Try merging the largest color into another
                if len(color_classes) > 1:
                    colors_sorted = sorted(color_classes.keys())
                    # Try to recolor all of color c1 to c2
                    c1 = colors_sorted[0]
                    for c2 in colors_sorted[1:]:
                        if time.time() > max_time:
                            break
                        # Check if all vertices of c1 can move to c2
                        # This is complex; skip for now
                        pass

        return color

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        conflict = build_conflict(edges, n)

        # DSATUR initial coloring
        color = dsatur_color(conflict, m)
        best = list(color)
        best_colors = max(best) + 1 if m > 0 else 0

        # Try greedy with degree ordering
        degree = [len(conflict[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: -degree[i])
        color_g = greedy_color(conflict, m, order)
        colors_g = max(color_g) + 1 if m > 0 else 0
        if colors_g < best_colors:
            best = color_g
            best_colors = colors_g

        # Try a few random orderings
        rng = Random(42)
        for trial in range(30):
            if time.time() > deadline:
                break
            order = list(range(m))
            rng.shuffle(order)
            color_r = greedy_color(conflict, m, order)
            colors_r = max(color_r) + 1 if m > 0 else 0
            if colors_r < best_colors:
                best = color_r
                best_colors = colors_r

        # Local improvement
        best = local_improve(best, conflict, m, deadline - 5)
        best_colors = max(best) + 1 if m > 0 else 0

        return best

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            # Not enough time; use a quick greedy
            edges = instance(*spec)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            conflict = build_conflict(edges, spec[0])
            degree = [len(conflict[i]) for i in range(m)]
            order = sorted(range(m), key=lambda i: -degree[i])
            out.append(greedy_color(conflict, m, order))
        else:
            out.append(solve_instance(*spec))

    return out
