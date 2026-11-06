import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles and their edge lists
    four_cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]

        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
            for e in cycle_edges:
                others = [x for x in cycle_edges if x != e]
                edge_to_cycles[e].append(others)

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True

    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        for others in edge_to_cycles[edge]:
            cycle_colors = set()
            cycle_colors.add(new_color)
            for e in others:
                cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True

    # Base: modular coloring as initial valid solution
    def modular_coloring(a, b, p):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = (i * a + j * b) % p
        return colors

    random.seed(42)

    # Find a good base coloring
    best_colors = None
    best_count = 0
    candidates = []
    for p in [17, 19, 23, 29, 31]:
        for a in range(1, p):
            for b in range(1, p):
                if a != b:
                    candidates.append((a, b, p))
    random.shuffle(candidates)

    for (a, b, p) in candidates[:30]:
        if time.time() > deadline - 15:
            break
        colors = modular_coloring(a, b, p)
        if is_valid(colors):
            cnt = count_distinct(colors)
            if cnt > best_count:
                best_count = cnt
                best_colors = colors[:]

    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_count = 1

    # Phase 1: Constraint-guided greedy color introduction
    used_colors = set(best_colors)
    unused_colors = [c for c in range(MAX_COLOR) if c not in used_colors]

    # Order edges by how many constraints they participate in (most constrained first)
    edge_constraint_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    edges_order = sorted(range(NUM_EDGES), key=lambda e: edge_constraint_count[e])

    for edge in edges_order:
        if time.time() > deadline - 12:
            break
        if not unused_colors:
            break

        # Try to assign a new color to this edge
        random.shuffle(unused_colors)
        for new_color in unused_colors:
            if try_change(best_colors, edge, new_color):
                best_colors[edge] = new_color
                unused_colors.remove(new_color)
                best_count += 1
                break

    # Phase 2: Randomized local search with constraint propagation
    for restart in range(50):
        if time.time() > deadline - 5:
            break

        current = best_colors[:]
        current_count = best_count
        used = set(current)
        unused = [c for c in range(MAX_COLOR) if c not in used]

        if not unused:
            break

        # Iteratively try to introduce new colors
        for step in range(200):
            if time.time() > deadline - 5:
                break
            if not unused:
                break

            # Pick a random edge
            edge = random.randint(0, NUM_EDGES - 1)

            # Get valid colors for this edge
            valid_new_colors = []
            for c in unused:
                if try_change(current, edge, c):
                    valid_new_colors.append(c)

            if valid_new_colors:
                # Pick a random valid color
                new_color = random.choice(valid_new_colors)
                current[edge] = new_color
                unused.remove(new_color)
                current_count += 1

                if current_count > best_count:
                    best_count = current_count
                    best_colors = current[:]

            # Also try swapping colors to free up space for new ones
            if random.random() < 0.3:
                # Try a swap: pick two edges and swap their colors
                if len(unused) > 0:
                    e1 = random.randint(0, NUM_EDGES - 1)
                    e2 = random.randint(0, NUM_EDGES - 1)
                    if e1 != e2:
                        c1 = current[e1]
                        c2 = current[e2]
                        if c1 != c2:
                            # Check if swap is valid
                            current[e1] = c2
                            current[e2] = c1
                            if is_valid(current):
                                # Try to add a new color after swap
                                if unused:
                                    e3 = random.randint(0, NUM_EDGES - 1)
                                    for nc in unused:
                                        if try_change(current, e3, nc):
                                            current[e3] = nc
                                            unused.remove(nc)
                                            current_count += 1
                                            if current_count > best_count:
                                                best_count = current_count
                                                best_colors = current[:]
                                            break
                            else:
                                current[e1] = c1
                                current[e2] = c2

    # Final verification
    if not is_valid(best_colors):
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i

    return best_colors
