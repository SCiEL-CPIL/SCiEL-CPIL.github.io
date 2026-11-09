import time
import random
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a in range(N - 3):
    for b in range(a + 1, N - 2):
        for c in range(b + 1, N - 1):
            for d in range(c + 1, N):
                _CYCLES.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a)))
                _CYCLES.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a)))
                _CYCLES.append((_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a)))

NUM_CYCLES = len(_CYCLES)  # 5460

# Precompute for each edge, which cycles it belongs to
_EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for _ci, _cycle in enumerate(_CYCLES):
    for _e in _cycle:
        _EDGE_CYCLES[_e].append(_ci)


def construct():
    """
    Construct a Gallai coloring of K_16 with 20 distinct edge colors,
    then attempt to improve via fast local search.
    """
    start_time = time.time()
    DEADLINE = start_time + 55.0  # safety margin before 60s kill

    # --- Build initial Gallai coloring via recursive partition tree ---
    # Partition:
    #   L0=[0,1,2], L1=[3,4,5], L2=[6,7,8], L3=[9,10,11], L4=[12,13,14], L5=[15]
    #   Node A: L0 vs L1          -> color 15
    #   Node B: L2 vs L3          -> color 16
    #   Node C: (L0+L1) vs (L2+L3)-> color 17
    #   Node D: L4 vs L5          -> color 18
    #   Root: (L0..L3) vs (L4+L5) -> color 19
    # Leaf internals: 15 distinct colors (0..14)
    # Total: 20 distinct colors

    colors = [0] * NUM_EDGES
    color_counter = 0

    def get_new_color():
        nonlocal color_counter
        c = color_counter
        color_counter += 1
        return c

    def color_all_distinct(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                colors[_edge_idx(u, v)] = get_new_color()

    def color_node(part1, part2):
        c = get_new_color()
        for u in part1:
            for v in part2:
                colors[_edge_idx(u, v)] = c

    L0 = [0, 1, 2]
    L1 = [3, 4, 5]
    L2 = [6, 7, 8]
    L3 = [9, 10, 11]
    L4 = [12, 13, 14]
    L5 = [15]

    for leaf in (L0, L1, L2, L3, L4, L5):
        if len(leaf) >= 2:
            color_all_distinct(leaf)

    color_node(L0, L1)
    color_node(L2, L3)
    color_node(L0 + L1, L2 + L3)
    color_node(L4, L5)
    color_node(L0 + L1 + L2 + L3, L4 + L5)

    # Track used colors incrementally
    used_colors = set(colors)
    best_count = len(used_colors)
    best_colors = colors[:]

    # --- Fast local search ---
    rng = random.Random(42)
    all_edges = list(range(NUM_EDGES))

    def can_take_color(edge, new_color):
        """Check if assigning new_color to edge keeps all cycles valid."""
        for ci in _EDGE_CYCLES[edge]:
            cycle = _CYCLES[ci]
            # Check: do the other 3 edges have 3 distinct colors?
            # If yes, new_color must match at least one of them.
            other_colors = []
            for e in cycle:
                if e != edge:
                    other_colors.append(colors[e])
            # If all 3 other colors are distinct, new_color must be one of them
            if len(set(other_colors)) == 3:
                if new_color not in other_colors:
                    return False
        return True

    # Phase 1: Try to add new colors to edges that currently share colors
    # Focus on edges in the partition tree's "cross" groups (they share colors)
    # Group edges by color to find candidates
    def get_color_groups():
        groups = {}
        for e in range(NUM_EDGES):
            c = colors[e]
            if c not in groups:
                groups[c] = []
            groups[c].append(e)
        return groups

    # Phase 1: Add new colors
    for it in range(20000):
        if time.time() > DEADLINE:
            break

        # Get color groups, pick a color with >= 2 edges
        groups = get_color_groups()
        multi_colors = [c for c, edges in groups.items() if len(edges) >= 2]

        if not multi_colors:
            break

        # Pick a random multi-color and a random edge from it
        target_color = rng.choice(multi_colors)
        edge = rng.choice(groups[target_color])

        # Try to assign a new unused color
        available = set(range(120)) - used_colors
        if not available:
            break

        # Try random available colors
        candidates = list(available)
        rng.shuffle(candidates)
        success = False
        for nc in candidates:
            if can_take_color(edge, nc):
                # Apply
                colors[edge] = nc
                used_colors.add(nc)
                best_count = len(used_colors)
                best_colors = colors[:]
                success = True
                break

    # Phase 2: More aggressive - try all edges, all available colors
    for it in range(50000):
        if time.time() > DEADLINE:
            break

        edge = rng.choice(all_edges)
        available = set(range(120)) - used_colors
        if not available:
            break

        current_color = colors[edge]
        # Try to find a new color that is valid
        candidates = list(available)
        rng.shuffle(candidates)
        for nc in candidates:
            if nc == current_color:
                continue
            if can_take_color(edge, nc):
                colors[edge] = nc
                used_colors.add(nc)
                best_count = len(used_colors)
                best_colors = colors[:]
                break

    # Phase 3: Try to free up colors by merging (recolor an edge to an existing color)
    # This can help if we have a "stuck" configuration where no new color fits,
    # but recoloring one edge to free up a constraint could allow another edge to get a new color.
    for it in range(30000):
        if time.time() > DEADLINE:
            break

        # Pick a random edge
        edge = rng.choice(all_edges)
        current_color = colors[edge]

        # Try recoloring to an existing color (not the current one)
        existing = list(used_colors)
        rng.shuffle(existing)
        for nc in existing:
            if nc == current_color:
                continue
            if can_take_color(edge, nc):
                # Apply and then try to add a new color somewhere
                colors[edge] = nc
                # Now try to add a new color to a random edge
                available = set(range(120)) - used_colors
                if available:
                    for e2 in rng.sample(all_edges, min(5, NUM_EDGES)):
                        candidates2 = list(available)
                        rng.shuffle(candidates2)
                        for nc2 in candidates2:
                            if can_take_color(e2, nc2):
                                colors[e2] = nc2
                                used_colors.add(nc2)
                                if len(used_colors) > best_count:
                                    best_count = len(used_colors)
                                    best_colors = colors[:]
                                break
                break

    # Final: return the best coloring found
    result = best_colors[:]
    return result


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
    print(f"Time: {t1 - t0:.2f}s")
