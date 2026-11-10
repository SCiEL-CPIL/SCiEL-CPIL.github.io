"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START
def construct():
    """
    Construct edge coloring of K_16 with no rainbow 4-cycle.
    Maximize number of distinct colors used.
    Structural approach: multi-start greedy construction with constraint tracking
    + perturbation-based local search (simulated annealing style).
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i*(i+1)//2 + (j - i - 1)

    # Precompute all 4-cycles (3 per 4-vertex subset)
    all_4sets = list(itertools.combinations(range(N), 4))
    constraints = []  # list of tuples of 4 edge indices
    for a, b, c, d in all_4sets:
        constraints.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        constraints.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        constraints.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

    num_constraints = len(constraints)

    # Build adjacency: for each edge, which constraints it belongs to
    edge_to_constraints = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(constraints):
        for e in cyc:
            edge_to_constraints[e].append(ci)

    # For each constraint, store the 4 edge indices
    # Precompute for fast checking
    constraint_edges = [list(c) for c in constraints]

    def count_distinct_colors(colors):
        return len(set(colors))

    def is_valid(colors):
        """Check if coloring has no rainbow 4-cycle."""
        for ci in range(num_constraints):
            c = constraint_edges[ci]
            if len(set(colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]])) == 4:
                return False
        return True

    def is_rainbow(colors, ci):
        c = constraint_edges[ci]
        return len(set(colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]])) == 4

    def check_edge_assignment(colors, e, new_col):
        """Check if assigning new_col to edge e creates any rainbow cycle."""
        old_col = colors[e]
        colors[e] = new_col
        valid = True
        for ci in edge_to_constraints[e]:
            c = constraint_edges[ci]
            if len(set(colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]])) == 4:
                valid = False
                break
        colors[e] = old_col
        return valid

    def greedy_construct(order, color_pool_size=120):
        """
        Greedy: assign colors to edges in given order.
        For each edge, try colors from color_pool_size down to 0,
        preferring unused colors. Skip colors that would create a rainbow cycle.
        """
        colors = [-1] * NUM_EDGES
        used_colors = set()

        for e in order:
            # Try to assign a new color first (from high to low for diversity)
            assigned = False
            # Prefer colors not yet used
            for col in range(color_pool_size - 1, -1, -1):
                if col in used_colors:
                    continue
                if check_edge_assignment(colors, e, col):
                    colors[e] = col
                    used_colors.add(col)
                    assigned = True
                    break
            if not assigned:
                # Must reuse an existing color
                for col in range(color_pool_size):
                    if col not in used_colors:
                        continue
                    if check_edge_assignment(colors, e, col):
                        colors[e] = col
                        assigned = True
                        break
            if not assigned:
                # Fallback: assign color 0 (shouldn't happen)
                colors[e] = 0
                used_colors.add(0)

        return colors, len(used_colors)

    def local_search_improve(colors, current_count, time_budget=3.0):
        """
        Local search: try to increase color count by:
        1. Directly assigning new colors to edges
        2. Two-edge swaps to free up constraints
        3. Perturbation: temporarily change colors, then repair
        """
        best = colors[:]
        best_count = current_count
        start = time.time()

        # Track which colors are used
        used = set(best)

        while time.time() - start < time_budget:
            improved = False

            # Phase 1: Try to assign a new color to a random edge
            for attempt in range(100):
                if time.time() - start > time_budget:
                    break
                e = random.randint(0, NUM_EDGES - 1)
                # Pick a color not in used
                available = [c for c in range(120) if c not in used]
                if not available:
                    break
                new_col = random.choice(available)
                if check_edge_assignment(best, e, new_col):
                    best[e] = new_col
                    used.add(new_col)
                    best_count += 1
                    improved = True
                    break

            # Phase 2: Two-edge swap to reorganize, then try new color
            if not improved:
                for attempt in range(50):
                    if time.time() - start > time_budget:
                        break
                    e1 = random.randint(0, NUM_EDGES - 1)
                    e2 = random.randint(0, NUM_EDGES - 1)
                    if e1 == e2:
                        continue
                    old1, old2 = best[e1], best[e2]
                    if old1 == old2:
                        continue
                    # Swap
                    best[e1], best[e2] = old2, old1
                    if not is_valid(best):
                        best[e1], best[e2] = old1, old2
                        continue
                    # Try to assign new color to some edge
                    available = [c for c in range(120) if c not in used]
                    if available:
                        new_col = random.choice(available)
                        e3 = random.randint(0, NUM_EDGES - 1)
                        if check_edge_assignment(best, e3, new_col):
                            best[e3] = new_col
                            used.add(new_col)
                            best_count += 1
                            improved = True
                            break
                    # Undo swap
                    best[e1], best[e2] = old1, old2

            # Phase 3: Perturbation - change 1-2 edges to random colors, then repair
            if not improved and best_count > 5:
                for attempt in range(20):
                    if time.time() - start > time_budget:
                        break
                    saved = best[:]
                    # Perturb: change 1-3 edges to random colors
                    num_perturb = random.randint(1, 3)
                    perturbed_edges = []
                    for _ in range(num_perturb):
                        e = random.randint(0, NUM_EDGES - 1)
                        perturbed_edges.append((e, best[e]))
                        best[e] = random.randint(0, 119)
                    # Repair: for each edge that now violates, try to fix
                    repair_ok = True
                    for e, old_c in perturbed_edges:
                        # Try to find a valid color for e
                        found = False
                        for col in sorted(range(120), key=lambda x: random.random()):
                            if check_edge_assignment(best, e, col):
                                best[e] = col
                                found = True
                                break
                        if not found:
                            repair_ok = False
                            break
                    if repair_ok and is_valid(best):
                        new_count = count_distinct_colors(best)
                        if new_count > best_count:
                            best_count = new_count
                            used = set(best)
                            improved = True
                            break
                    # Restore if not improved
                    if not improved:
                        best = saved[:]
                        used = set(best)

            if not improved:
                # Try a more aggressive perturbation
                if best_count > 3:
                    saved = best[:]
                    num_perturb = random.randint(3, 6)
                    perturbed_edges = []
                    for _ in range(num_perturb):
                        e = random.randint(0, NUM_EDGES - 1)
                        perturbed_edges.append((e, best[e]))
                        best[e] = random.randint(0, 119)
                    repair_ok = True
                    for e, old_c in perturbed_edges:
                        found = False
                        for col in sorted(range(120), key=lambda x: random.random()):
                            if check_edge_assignment(best, e, col):
                                best[e] = col
                                found = True
                                break
                        if not found:
                            repair_ok = False
                            break
                    if repair_ok and is_valid(best):
                        new_count = count_distinct_colors(best)
                        if new_count > best_count:
                            best_count = new_count
                            used = set(best)
                            improved = True
                    else:
                        best = saved[:]
                        used = set(best)

        return best, best_count

    # Multi-start: try different random edge orderings
    best_result = None
    best_count = 0

    # First, try a deterministic algebraic baseline for a quick start
    for K in [7, 11, 13, 16, 23, 31, 37, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = (i + j) % K
        cnt = count_distinct_colors(colors)
        if cnt > best_count and is_valid(colors):
            best_count = cnt
            best_result = colors[:]

    # Multi-start greedy with different random orderings
    num_starts = 30
    for s in range(num_starts):
        if time.time() > deadline - 5:
            break

        # Different edge orderings
        order = list(range(NUM_EDGES))
        random.shuffle(order)

        # Sometimes use degree-based ordering (edges between high-degree vertices first)
        if s % 3 == 1:
            # Sort by minimum endpoint
            order.sort(key=lambda e: (e // N, e))
        elif s % 3 == 2:
            # Sort by maximum endpoint
            order.sort(key=lambda e: (e % N, e))

        colors, cnt = greedy_construct(order)
        if cnt > best_count and is_valid(colors):
            best_count = cnt
            best_result = colors[:]

        # Local search improvement
        if cnt > best_count - 2 and time.time() < deadline - 5:
            t_budget = min(2.0, max(0.5, (deadline - 5 - time.time()) / max(1, num_starts - s)))
            improved_colors, improved_count = local_search_improve(colors, cnt, t_budget)
            if improved_count > best_count:
                best_count = improved_count
                best_result = improved_colors[:]

    # Final polish: local search on the best result
    if best_result is not None and time.time() < deadline - 3:
        t_budget = min(3.0, max(0.5, deadline - 3 - time.time()))
        final_colors, final_count = local_search_improve(best_result, best_count, t_budget)
        if final_count > best_count:
            best_count = final_count
            best_result = final_colors[:]

    if best_result is None:
        best_result = [0] * NUM_EDGES
        best_count = 1

    return best_result
# EVOLVE-BLOCK-END
