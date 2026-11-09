import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

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

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []

        # Build incompatibility as bitmask lists (m up to ~200, fits in Python int)
        incompat = [0] * m  # bitmask of incompatible edges for each edge i

        for i in range(m):
            u1, v1 = edges[i]
            mask = 0
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if edges i and j are incompatible
                # Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask |= (1 << j)
                    incompat[j] |= (1 << i)
                    continue
                # Joined by an edge?
                if (u2 in adj[u1] or v2 in adj[u1] or
                    u2 in adj[v1] or v2 in adj[v1]):
                    mask |= (1 << j)
                    incompat[j] |= (1 << i)
            incompat[i] = mask

        # Greedy coloring with DSATUR-like ordering
        # Order edges by descending incompatibility degree
        degs = [bin(incompat[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: -degs[x])

        colors = [-1] * m
        color_count = 0

        for v in order:
            # Find smallest color not used by incompatible neighbors
            used = 0
            mask = incompat[v]
            # Extract colors of neighbors
            for j in range(m):
                if (mask >> j) & 1:
                    if colors[j] >= 0:
                        used |= (1 << colors[j])
            c = 0
            while (used >> c) & 1:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1

        # Local search: try to eliminate colors one by one
        # For each color (from highest to lowest, or smallest class first),
        # try to merge it into another color, resolving conflicts.

        def try_eliminate_color(target, source):
            """Try to eliminate color 'target' by merging into 'source'.
            Returns True if successful."""
            nonlocal color_count

            # Get edges with target color
            target_edges = [i for i in range(m) if colors[i] == target]
            if not target_edges:
                return True

            # Check conflicts: target_edges that are incompatible with source-colored edges
            source_mask = 0
            for i in range(m):
                if colors[i] == source:
                    source_mask |= (1 << i)

            conflicts = []
            for i in target_edges:
                if incompat[i] & source_mask:
                    conflicts.append(i)

            if not conflicts:
                # No conflicts, just recolor
                for i in target_edges:
                    colors[i] = source
                # Update color_count
                if target == color_count - 1:
                    # Check if target is still used
                    for i in range(m):
                        if colors[i] == target:
                            break
                    else:
                        color_count -= 1
                return True

            # Save state
            old_colors = colors[:]
            old_cc = color_count

            # Move all target edges to source first
            for i in target_edges:
                colors[i] = source

            # Now resolve conflicts: move conflicting edges to other colors
            resolved = True
            for i in conflicts:
                # Find valid colors for i
                # Cannot use source (clashes with source edges)
                # Cannot use target (being eliminated, but might still have edges)
                # Actually target is being eliminated, so we should avoid it
                used = 0
                mask = incompat[i]
                for j in range(m):
                    if (mask >> j) & 1:
                        if colors[j] >= 0:
                            used |= (1 << colors[j])
                # Exclude source and target
                used |= (1 << source)
                if target < color_count:
                    used |= (1 << target)

                # Find smallest valid color
                c = 0
                while (used >> c) & 1:
                    c += 1
                colors[i] = c
                if c + 1 > color_count:
                    color_count = c + 1

            # Check if we reduced or kept color count
            max_used = -1
            for i in range(m):
                if colors[i] > max_used:
                    max_used = colors[i]
            new_cc = max_used + 1

            if new_cc <= old_cc:
                color_count = new_cc
                return True
            else:
                # Revert
                colors[:] = old_colors
                color_count = old_cc
                return False

        # Try to eliminate colors
        # Strategy: repeatedly try to eliminate the color with the fewest edges
        # into a random other color
        rng = random.Random(12345)
        improved = True
        max_iters = 5000
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1

            # Get color counts
            color_counts = defaultdict(int)
            for c in colors:
                color_counts[c] += 1

            if len(color_counts) <= 1:
                break

            # Sort colors by count ascending
            sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
            target = sorted_colors[0]  # Eliminate the smallest color class

            # Try merging into each other color
            other_colors = [c for c in color_counts if c != target]
            rng.shuffle(other_colors)

            for source in other_colors:
                if time.time() > deadline - 5.0:
                    break
                if try_eliminate_color(target, source):
                    improved = True
                    break

        # Final polish: try to reduce further by checking all pairs
        # Also try swapping colors to free up a color class
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            improved = False
            color_counts = defaultdict(int)
            for c in colors:
                color_counts[c] += 1

            if len(color_counts) <= 1:
                break

            # Try eliminating each color into each other
            colors_list = sorted(color_counts.keys(), key=lambda x: color_counts[x])
            for target in colors_list:
                if time.time() > deadline - 5.0:
                    break
                other_colors = [c for c in color_counts if c != target]
                rng.shuffle(other_colors)
                for source in other_colors:
                    if time.time() > deadline - 5.0:
                        break
                    if try_eliminate_color(target, source):
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Return best effort
            break

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        colors = solve_instance(edges, adj)
        out.append(colors)

    # If we ran out of time, fill remaining with empty lists (shouldn't happen)
    while len(out) < 3:
        out.append([])

    return out
