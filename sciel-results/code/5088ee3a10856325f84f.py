import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s limit

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # ---- Precompute all 4-cycles and per-edge impact matrix ----
    four_cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(a, d))
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(c, d), edge_idx(a, c))
        e3 = (edge_idx(a, c), edge_idx(b, c), edge_idx(b, d), edge_idx(a, d))
        four_cycles.append(e1)
        four_cycles.append(e2)
        four_cycles.append(e3)

    NC = len(four_cycles)
    # impact[e] = boolean vector over cycles: True where cycle e participates
    impact = [None] * NUM_EDGES
    for e in range(NUM_EDGES):
        impact[e] = np.zeros(NC, dtype=bool)
    for ci, cyc in enumerate(four_cycles):
        for e in cyc:
            impact[e][ci] = True

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for cyc in four_cycles:
            if len(set(colors[e] for e in cyc)) == 4:
                return False
        return True

    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        mask = impact[edge]
        if not mask.any():
            return True
        sub = np.array([colors[e] for e in range(NUM_EDGES) if mask[e]])
        # a cycle is violated iff its other 3 edges have 3 distinct colors all != new_color
        other = sub[sub != new_color]
        if other.size == 3 and len(set(other.tolist())) == 3:
            return False
        return True

    # ---- Structural base: partition into 4 groups of 4, color by min group ----
    def base_coloring():
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = min(i // 4, j // 4)
        return colors

    best_colors = base_coloring()
    best_count = count_distinct(best_colors)

    random.seed(777)

    def try_introduce_new_colors(colors, budget):
        """Greedy: repeatedly pick an edge and give it an unused color if valid."""
        used = set(colors)
        count = len(used)
        # precompute list of edges grouped by current color for consolidation awareness
        while time.time() < budget:
            unused = [c for c in range(MAX_COLOR) if c not in used]
            if not unused:
                break
            random.shuffle(unused)
            # pick a random edge
            edge = random.randint(0, NUM_EDGES - 1)
            improved = False
            for nc in unused:
                if try_change(colors, edge, nc):
                    colors[edge] = nc
                    used.add(nc)
                    count += 1
                    improved = True
                    break
            if not improved:
                # try a few more random edges before giving up on this pass
                for _ in range(20):
                    edge = random.randint(0, NUM_EDGES - 1)
                    for nc in unused:
                        if try_change(colors, edge, nc):
                            colors[edge] = nc
                            used.add(nc)
                            count += 1
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
        return count

    # ---- Consolidation: recolor edges to existing colors to free up capacity ----
    def consolidate(colors, budget):
        """For edges whose color is unique, try to move them to a more common
        color (so the unique color can be reused elsewhere)."""
        while time.time() < budget:
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            # find edges with rare colors
            rare_edges = []
            for e in range(NUM_EDGES):
                if color_counts[colors[e]] <= 2:
                    rare_edges.append(e)
            if not rare_edges:
                break
            random.shuffle(rare_edges)
            moved = False
            for edge in rare_edges:
                if time.time() > budget:
                    break
                old = colors[edge]
                # candidate target colors: more common colors
                candidates = [c for c in color_counts if c != old and color_counts[c] > 1]
                random.shuffle(candidates)
                for nc in candidates:
                    if try_change(colors, edge, nc):
                        colors[edge] = nc
                        color_counts[old] -= 1
                        color_counts[nc] = color_counts.get(nc, 0) + 1
                        moved = True
                        break
            if not moved:
                break
        return count_distinct(colors)

    # ---- Main search loop: alternate introduce + consolidate ----
    for it in range(400):
        if time.time() > deadline - 3:
            break
        current = best_colors[:]
        # introduce new colors
        count = try_introduce_new_colors(current, time.time() + 1.5)
        # consolidate to free capacity, then introduce more
        count = consolidate(current, time.time() + 1.0)
        count = try_introduce_new_colors(current, time.time() + 1.0)
        if count > best_count:
            best_count = count
            best_colors = current[:]

    # Final verification
    if not is_valid(best_colors):
        best_colors = base_coloring()

    return best_colors
