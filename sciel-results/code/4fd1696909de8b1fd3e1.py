import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = MAX_COLOR + 1  # 120

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _is_rainbow(cyc, colors):
    """Check if the cycle has 4 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def _validate(colors):
    """Full validation of a coloring."""
    for cyc in CYCLES:
        if _is_rainbow(cyc, colors):
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _greedy_seed(seed):
    """Build a valid coloring using greedy color introduction."""
    rng = random.Random(seed)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    color_counts = np.zeros(NUM_COLORS, dtype=np.int32)
    color_counts[0] = NUM_EDGES

    # Precompute edge_cycle_info for this run
    edge_cycle_info = []
    for e in range(NUM_EDGES):
        info = []
        for ci in EDGE_TO_CYCLES[e]:
            cycle = CYCLES[ci]
            for pos in range(4):
                if cycle[pos] == e:
                    info.append((ci, pos))
                    break
        edge_cycle_info.append(info)

    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos in edge_cycle_info[e]:
            cycle = CYCLES[ci]
            nc0, nc1, nc2, nc3 = (
                new_color if pos == 0 else colors_arr[cycle[0]],
                new_color if pos == 1 else colors_arr[cycle[1]],
                new_color if pos == 2 else colors_arr[cycle[2]],
                new_color if pos == 3 else colors_arr[cycle[3]],
            )
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True

    # Introduce colors 1..119 one at a time
    for new_c in range(1, MAX_COLOR + 1):
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        for e in edge_order:
            if _is_move_valid(e, new_c, colors):
                old = colors[e]
                colors[e] = new_c
                color_counts[old] -= 1
                color_counts[new_c] += 1
                break
    return colors, color_counts, edge_cycle_info

def _local_search(colors, color_counts, edge_cycle_info, deadline, rng):
    """Local search maximizing distinct colors while keeping validity."""
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    current_distinct = best_distinct

    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos in edge_cycle_info[e]:
            cycle = CYCLES[ci]
            nc0 = new_color if pos == 0 else colors_arr[cycle[0]]
            nc1 = new_color if pos == 1 else colors_arr[cycle[1]]
            nc2 = new_color if pos == 2 else colors_arr[cycle[2]]
            nc3 = new_color if pos == 3 else colors_arr[cycle[3]]
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True

    while time.time() < deadline:
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        improved = False

        for e in edge_order:
            if time.time() >= deadline:
                break
            old_color = colors[e]

            # Compute allowed colors via forbidden-set pruning
            allowed = None
            for ci, pos in edge_cycle_info[e]:
                cycle = CYCLES[ci]
                other_edges = [cycle[k] for k in range(4) if k != pos]
                other_colors = [colors[oe] for oe in other_edges]
                other_set = set(other_colors)
                if len(other_set) == 3:
                    if allowed is None:
                        allowed = set(other_set)
                    else:
                        allowed &= other_set

            if allowed is not None:
                candidates = [c for c in allowed if c != old_color]
                if not candidates:
                    continue
            else:
                # No hard constraint; prefer unused then under-represented colors
                unused = [c for c in range(NUM_COLORS) if color_counts[c] == 0]
                if unused:
                    rng.shuffle(unused)
                    candidates = unused[:15]
                else:
                    present = [c for c in range(NUM_COLORS) if color_counts[c] > 0]
                    present.sort(key=lambda c: color_counts[c])
                    candidates = present[:15]
                # Drop old color
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue

            new_color = rng.choice(candidates)

            if not _is_move_valid(e, new_color, colors):
                continue

            # Apply and update counts
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            colors[e] = new_color

            new_distinct = current_distinct
            if color_counts[old_color] == 0:
                new_distinct -= 1
            if color_counts[new_color] == 1:
                new_distinct += 1

            if new_distinct >= current_distinct:
                # Accept (improvement or tie)
                current_distinct = new_distinct
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    best_colors = colors.copy()
                    improved = True
            else:
                # Revert
                colors[e] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1

        if not improved:
            # Perturbation to escape local minimum
            for _ in range(rng.randint(2, 6)):
                e = rng.randint(0, NUM_EDGES - 1)
                old = colors[e]
                new = rng.randint(0, MAX_COLOR)
                if new == old:
                    continue
                if _is_move_valid(e, new, colors):
                    color_counts[old] -= 1
                    color_counts[new] += 1
                    colors[e] = new

    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard kill

    best_overall_colors = None
    best_overall_distinct = 0

    num_restarts = 0
    while time.time() < deadline - 3.0:
        seed = 1000 + num_restarts
        colors, color_counts, edge_cycle_info = _greedy_seed(seed)

        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        rng = random.Random(seed + 7777)

        # Small initial perturbation for diversity
        for _ in range(rng.randint(0, 5)):
            e = rng.randint(0, NUM_EDGES - 1)
            new_c = rng.randint(0, MAX_COLOR)
            old = colors[e]
            if new_c == old:
                continue
            # Quick validity check
            ok = True
            for ci, pos in edge_cycle_info[e]:
                cycle = CYCLES[ci]
                nc0 = new_c if pos == 0 else colors[cycle[0]]
                nc1 = new_c if pos == 1 else colors[cycle[1]]
                nc2 = new_c if pos == 2 else colors[cycle[2]]
                nc3 = new_c if pos == 3 else colors[cycle[3]]
                if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                    ok = False
                    break
            if ok:
                color_counts[old] -= 1
                color_counts[new_c] += 1
                colors[e] = new_c

        colors, distinct = _local_search(colors, color_counts, edge_cycle_info, local_deadline, rng)

        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()

        num_restarts += 1

    if best_overall_colors is None:
        # Fallback: all zeros is always valid
        best_overall_colors = np.zeros(NUM_EDGES, dtype=np.int32)

    # Final validation
    if not _validate(best_overall_colors):
        return [0] * NUM_EDGES

    return best_overall_colors.tolist()
