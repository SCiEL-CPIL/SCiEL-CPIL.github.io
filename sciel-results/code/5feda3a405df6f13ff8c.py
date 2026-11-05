"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using incremental local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0

    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    def edge_index(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adjacency(colors, n):
        """Build adjacency lists per color: adj[c][v] = set of neighbors."""
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        return adj

    def count_c6_in_adj(adj, n):
        """Count C6s in a single color's adjacency. Returns count."""
        count = 0
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1):
                        continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3):
                                continue
                        # Check v4 -> v5 -> v0
                        common = adj[v4] & adj[v0]
                        for v5 in common:
                            if v5 not in (v0, v1, v2, v3, v4):
                                count += 1
        return count

    def total_mono_c6(adj, n):
        """Total monochromatic C6s across all 4 colors."""
        total = 0
        for c in range(4):
            total += count_c6_in_adj(adj[c], n)
        return total

    def delta_c6_count(adj, n, u, v, old_color, new_color):
        """
        Compute the change in monochromatic C6 count when edge (u,v) changes
        from old_color to new_color.
        A C6 is affected if it uses edge (u,v) in old_color (lost) or new_color (gained).
        """
        delta = 0
        # Remove C6s that used (u,v) in old_color
        delta -= _count_c6s_through_edge(adj[old_color], n, u, v)
        # Add C6s that use (u,v) in new_color
        delta += _count_c6s_through_edge(adj[new_color], n, u, v)
        return delta

    def _count_c6s_through_edge(adj, n, u, v):
        """Count C6s in this adjacency that use edge (u,v)."""
        # A C6 using (u,v): u-v-a-b-c-d-u or u-v-d-c-b-a-u (same cycle)
        # Fix: path of length 5 from u to v through 4 intermediate vertices
        # u - v - a - b - c - d - u
        # So we need: a in adj[v], b in adj[a]\{v}, c in adj[b]\{a,v}, d in adj[c]\{b,a,v}, d in adj[u]\{c}
        # And d != u,v,a,b,c
        count = 0
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[a]:
                if b in (u, v, a):
                    continue
                for c in adj[b]:
                    if c in (u, v, a, b):
                        continue
                    for d in adj[c]:
                        if d in (u, v, a, b, c):
                            continue
                        if u in adj[d]:
                            count += 1
        return count

    def local_search_incremental(colors, n, max_iters=50000):
        """Local search with incremental C6 counting."""
        num_edges = n * (n - 1) // 2
        colors_arr = np.array(colors, dtype=np.int32)
        adj = build_adjacency(colors_arr.tolist(), n)
        
        current_count = total_mono_c6(adj, n)
        if current_count == 0:
            return colors_arr.tolist(), 0

        T = 2.0
        no_improve = 0
        max_no_improve = 5000

        # Track which edges are in monochromatic C6s for weighted selection
        # We'll use a simple approach: periodically rebuild a set of "hot" edges
        hot_edges = set()

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
            if current_count == 0:
                return colors_arr.tolist(), 0

            # Rebuild hot edges periodically
            if it % 200 == 0:
                hot_edges = _find_hot_edges(adj, n, current_count)

            # Select edge: prefer hot edges
            if hot_edges and random.random() < 0.7:
                ei = random.choice(list(hot_edges))
            else:
                ei = random.randint(0, num_edges - 1)

            ei = int(ei)
            old_color = int(colors_arr[ei])
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue

            # Find (u, v) for edge index ei
            u, v = _edge_to_vertices(ei, n)

            # Compute delta
            delta = delta_c6_count(adj, n, u, v, old_color, new_color)

            if delta <= 0 or random.random() < np.exp(-delta / max(0.1, T)):
                # Accept: update adjacency
                adj[old_color][u].discard(v)
                adj[old_color][v].discard(u)
                adj[new_color][u].add(v)
                adj[new_color][v].add(u)
                colors_arr[ei] = new_color
                current_count += delta
                no_improve = 0
            else:
                no_improve += 1

            T *= 0.9999

        return colors_arr.tolist(), current_count

    def _find_hot_edges(adj, n, count):
        """Find edges that participate in monochromatic C6s."""
        hot = set()
        for c in range(4):
            for u in range(n):
                for v in adj[c][u]:
                    if v > u:  # only count once
                        ei = edge_index(u, v, n)
                        if _count_c6s_through_edge(adj[c], n, u, v) > 0:
                            hot.add(ei)
        return hot

    def _edge_to_vertices(ei, n):
        """Convert edge index to (u, v) with u < v."""
        # Find i such that cumulative edges up to i-1 <= ei < cumulative up to i
        # Edges from vertex i: (i, i+1), ..., (i, n-1) -> (n-1-i) edges
        # Cumulative before vertex i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Solve for i: i*(n-1) - i*(i-1)//2 <= ei
        # Binary search
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cum = mid * (n - 1) - mid * (mid - 1) // 2
            if cum <= ei:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        cum_before = i * (n - 1) - i * (i - 1) // 2
        j = i + 1 + (ei - cum_before)
        return i, j

    # Try n from 17 down
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break

        num_edges = n * (n - 1) // 2
        remaining = deadline - time.time()

        # Number of restarts based on remaining time
        if n >= 16:
            max_restarts = max(5, int(remaining * 2))
            max_iters = 30000
        else:
            max_restarts = max(3, int(remaining * 3))
            max_iters = 20000

        found = False
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break

            colors = [random.randint(0, 3) for _ in range(num_edges)]
            colors, count = local_search_incremental(colors, n, max_iters=max_iters)

            if count == 0:
                best_n = n
                best_colors = colors
                found = True
                break

        if found:
            continue
        else:
            if n == 17:
                continue
            else:
                break

    # Fallback
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    return {'n': best_n, 'colors': best_colors}
