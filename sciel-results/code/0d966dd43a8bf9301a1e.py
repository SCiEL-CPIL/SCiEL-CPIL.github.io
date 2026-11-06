import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))

    # Convert cycles to edge indices
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))

    # For each edge, find which cycles it belongs to and its position in each
    # edge_cycle_info[e] = list of (cycle_idx, position)
    edge_cycle_info = [[] for _ in range(NUM_EDGES)]
    for ci, c in enumerate(cycle_indices):
        for pos, e in enumerate(c):
            edge_cycle_info[e].append((ci, pos))

    # For each edge, precompute the "other 3 edges" in each cycle it belongs to
    # edge_other_edges[e] = list of (other1, other2, other3) for each cycle
    edge_other_edges = [[] for _ in range(NUM_EDGES)]
    for ci, c in enumerate(cycle_indices):
        for pos, e in enumerate(c):
            others = [c[p] for p in range(4) if p != pos]
            edge_other_edges[e].append(tuple(others))

    # Greedy constructive algorithm
    def greedy_construct(seed):
        random.seed(seed)
        colors = [-1] * NUM_EDGES  # -1 means uncolored
        color_used = [False] * MAX_COLORS  # track which colors are used
        num_colors_used = 0

        # Order of edges to assign - try a few different orderings
        # Ordering 1: natural lexicographic
        # Ordering 2: random shuffle
        # We'll use random shuffle for diversity

        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)

        for e in edge_order:
            # Determine forbidden colors for this edge
            # A color c is forbidden if, when assigned to e, some cycle
            # containing e would have 4 distinct colors.
            # For a cycle (e, o1, o2, o3):
            # If o1, o2, o3 are all colored and distinct from each other,
            # then c cannot equal any of {colors[o1], colors[o2], colors[o3]}
            # Actually, the condition is: the 4 colors must NOT be all distinct.
            # So c is forbidden if colors[o1], colors[o2], colors[o3] are all
            # distinct AND c is different from all three of them.
            # Wait, let me re-think:
            # The cycle has edges e, o1, o2, o3.
            # If o1, o2, o3 are colored with values v1, v2, v3.
            # If v1, v2, v3 are all distinct, then for the 4-cycle to NOT be rainbow,
            # c must equal v1 OR v2 OR v3.
            # So c is forbidden if c != v1 AND c != v2 AND c != v3.
            # If any of o1, o2, o3 is uncolored, no constraint from this cycle yet.

            forbidden = set()
            for others in edge_other_edges[e]:
                o1, o2, o3 = others
                v1 = colors[o1]
                v2 = colors[o2]
                v3 = colors[o3]
                # If all three are colored
                if v1 != -1 and v2 != -1 and v3 != -1:
                    if v1 != v2 and v1 != v3 and v2 != v3:
                        # All three are distinct, so c must be one of them
                        # c is forbidden if c is NOT in {v1, v2, v3}
                        # We want to find the highest color NOT forbidden
                        # So we mark all colors except v1, v2, v3 as forbidden
                        # But that's expensive. Instead, we can just check candidates.
                        # Let's collect the constraint: c must be in {v1, v2, v3}
                        # We'll handle this differently.
                        pass

            # Better approach: for each candidate color (from high to low),
            # check if it's valid.
            # Since MAX_COLORS is only 120, this is cheap.

            # Find the highest available color that is valid
            # "Available" means either unused, or we allow reuse.
            # To maximize distinct colors, prefer unused colors, but allow reuse if needed.

            # Actually, to maximize the number of distinct colors, we should:
            # 1. Try to assign a new (unused) color first.
            # 2. If no unused color works, assign a used color that works.

            # Let's iterate from MAX_COLORS-1 down to 0
            assigned = False
            for c in range(MAX_COLORS - 1, -1, -1):
                # Check if c is valid for edge e
                valid = True
                for others in edge_other_edges[e]:
                    o1, o2, o3 = others
                    v1 = colors[o1]
                    v2 = colors[o2]
                    v3 = colors[o3]
                    # If any of the others is uncolored, no constraint from this cycle
                    if v1 == -1 or v2 == -1 or v3 == -1:
                        continue
                    # Check if {c, v1, v2, v3} has 4 distinct values
                    if c != v1 and c != v2 and c != v3 and v1 != v2 and v1 != v3 and v2 != v3:
                        valid = False
                        break

                if valid:
                    colors[e] = c
                    if not color_used[c]:
                        color_used[c] = True
                        num_colors_used += 1
                    assigned = True
                    break

            if not assigned:
                # No valid color found - this shouldn't happen if we allow reuse
                # but just in case, assign 0
                colors[e] = 0
                if not color_used[0]:
                    color_used[0] = True
                    num_colors_used += 1

        return colors, num_colors_used

    # Try multiple random orderings and return the best
    best_colors = None
    best_count = 0

    # Run multiple greedy constructions with different random seeds
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 1.0:
            break

        colors, count = greedy_construct(attempt)

        # Verify validity (safety check)
        colors_np = np.array(colors, dtype=np.int32)
        cycle_colors = colors_np[np.array(cycle_indices, dtype=np.int32)]
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        all_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        if np.any(all_distinct):
            # Invalid, skip
            continue

        if count > best_count:
            best_count = count
            best_colors = colors

    # Fallback: if no valid coloring found, return a simple valid one
    if best_colors is None:
        # Use a simple valid coloring: all edges color 0
        best_colors = [0] * NUM_EDGES
        best_count = 1

    return best_colors
