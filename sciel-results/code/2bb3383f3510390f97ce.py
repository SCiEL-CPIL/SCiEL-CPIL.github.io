import time
import random
import numpy as np
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# ---------- Precomputation ----------
def _build():
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1

    # For each 4-vertex subset, the 3 cycles (each a tuple of 4 edge indices)
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append((idx_map[(a, b)], idx_map[(b, c)], idx_map[(c, d)], idx_map[(d, a)]))
        cycles.append((idx_map[(a, b)], idx_map[(b, d)], idx_map[(c, d)], idx_map[(a, c)]))
        cycles.append((idx_map[(a, c)], idx_map[(b, c)], idx_map[(b, d)], idx_map[(a, d)]))

    cycles_np = np.array(cycles, dtype=np.int32)

    # edge -> list of (cycle_index, [other 3 edges])
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)

    edge_cycle_info = []
    for e in range(NEDGES):
        info = []
        for ci in edge_to_cycles[e]:
            cyc = cycles_np[ci]
            pos = int(np.where(cyc == e)[0][0])
            other = [int(cyc[k]) for k in range(4) if k != pos]
            info.append((ci, other))
        edge_cycle_info.append(info)

    return idx_map, cycles_np, edge_to_cycles, edge_cycle_info

IDX_MAP, CYCLES, EDGE_TO_CYCLES, EDGE_CYCLE_INFO = _build()
NUM_CYCLES = len(CYCLES)


def _full_valid(colors_np):
    cc = colors_np[CYCLES]
    s = np.sort(cc, axis=1)
    ok = (s[:, 0] == s[:, 1]) | (s[:, 1] == s[:, 2]) | (s[:, 2] == s[:, 3])
    return bool(np.all(ok))


def _distinct(colors_np):
    return int(np.count_nonzero(np.bincount(colors_np, minlength=MAX_COLOR) > 0))


# ---------- CSP-style constructive search ----------
def _allowed_colors_for_edge(colors, e, domain_cache):
    """Return a set of colors that keep validity for edge e, given current colors."""
    old = colors[e]
    colors[e] = -1  # sentinel, not a real color
    allowed = set(range(MAX_COLOR))
    for ci, other in EDGE_CYCLE_INFO[e]:
        o0, o1, o2 = other
        c0, c1, c2 = colors[o0], colors[o1], colors[o2]
        # The 4-cycle would be rainbow iff new color is distinct from c0,c1,c2
        # AND c0,c1,c2 are pairwise distinct.
        if c0 != c1 and c0 != c2 and c1 != c2:
            # new color must equal one of {c0,c1,c2} to avoid rainbow
            allowed &= {c0, c1, c2}
    colors[e] = old
    return allowed


def _constructive_search(seed, deadline):
    rng = random.Random(seed)
    colors = np.full(NEDGES, -1, dtype=np.int32)
    assigned = np.zeros(NEDGES, dtype=bool)
    color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    distinct = 0

    # We'll assign edges one by one.
    # To maximize distinct colors, prefer assigning unused colors.
    # MRV: pick unassigned edge with fewest allowed colors.
    # LCV: among allowed, prefer colors that constrain neighbors least (i.e., most used / or unused to grow distinct).

    unassigned = set(range(NEDGES))

    while unassigned and time.time() < deadline:
        # Compute allowed colors for all unassigned edges (cheap-ish)
        best_edge = None
        best_allowed = None
        best_len = MAX_COLOR + 1

        # To save time, sample a subset if many remain
        if len(unassigned) > 40:
            sample = rng.sample(list(unassigned), 40)
        else:
            sample = list(unassigned)

        for e in sample:
            allowed = _allowed_colors_for_edge(colors, e, None)
            # We must be able to assign at least one color
            if len(allowed) < best_len:
                best_len = len(allowed)
                best_edge = e
                best_allowed = allowed
                if best_len <= 1:
                    break

        if best_edge is None:
            # Fallback: pick random
            best_edge = rng.choice(list(unassigned))
            best_allowed = _allowed_colors_for_edge(colors, best_edge, None)

        if len(best_allowed) == 0:
            # Dead end: backtrack not implemented; return what we have (may be invalid)
            # Instead, try to recover by unassigning a random edge and reassigning
            # For simplicity, break and let caller handle.
            break

        # Choose color: prefer unused colors to maximize distinct
        unused_in_allowed = [c for c in best_allowed if color_counts[c] == 0]
        if unused_in_allowed:
            new_color = rng.choice(unused_in_allowed)
        else:
            # Prefer colors that are most common among allowed (least constraining)
            # Actually, to keep validity easy, prefer colors that appear in many cycles' other edges
            # Simple heuristic: pick the color in best_allowed that is most frequent in the whole coloring
            freqs = {c: color_counts[c] for c in best_allowed}
            # Pick highest frequency (most "safe")
            new_color = max(best_allowed, key=lambda c: (color_counts[c], rng.random()))

        colors[best_edge] = new_color
        assigned[best_edge] = True
        unassigned.discard(best_edge)
        color_counts[new_color] += 1
        if color_counts[new_color] == 1:
            distinct += 1

    # If we didn't assign all edges, fill the rest with a safe valid coloring
    if not np.all(assigned):
        # Fallback: use a known valid base coloring and blend
        base = _base_valid_coloring()
        for e in range(NEDGES):
            if not assigned[e]:
                colors[e] = base[e]
                if color_counts[colors[e]] == 0:
                    distinct += 1
                color_counts[colors[e]] += 1
        assigned = np.ones(NEDGES, dtype=bool)

    return colors, distinct


def _base_valid_coloring():
    """A simple valid coloring: split into two 8-cliques."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            colors[idx] = 1 if (i >= 8 and j >= 8) else 0
    return colors


def _improve_by_recoloring(colors, deadline, rng):
    """After constructive search, do a few passes of recoloring to boost distinct count."""
    colors = colors.copy()
    color_counts = np.bincount(colors, minlength=MAX_COLOR)
    distinct = int(np.sum(color_counts > 0))
    best_colors = colors.copy()
    best_distinct = distinct

    while time.time() < deadline:
        improved = False
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)
        for e in edge_order:
            if time.time() >= deadline:
                break
            old = colors[e]
            allowed = _allowed_colors_for_edge(colors, e, None)
            # Prefer unused colors
            unused = [c for c in allowed if c != old and color_counts[c] == 0]
            if unused:
                new = rng.choice(unused)
                colors[e] = new
                color_counts[old] -= 1
                color_counts[new] += 1
                if color_counts[old] == 0:
                    distinct -= 1
                if color_counts[new] == 1:
                    distinct += 1
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()
                improved = True
        if not improved:
            break

    return best_colors, best_distinct


def _algebraic_coloring(K):
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            colors[IDX_MAP[(i, j)]] = (i + j) % K
    return colors


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    rng = random.Random(12345)

    best_colors = None
    best_distinct = 0

    # 1) Try algebraic colorings as strong seeds
    for K in range(2, 30):
        c = _algebraic_coloring(K)
        if _full_valid(c):
            d = _distinct(c)
            if d > best_distinct:
                best_distinct = d
                best_colors = c.copy()

    # 2) Multi-start constructive search with different seeds
    num_starts = 0
    while time.time() < deadline - 10.0:
        seed = rng.randint(0, 10**9)
        local_deadline = min(time.time() + 5.0, deadline - 8.0)
        colors, distinct = _constructive_search(seed, local_deadline)

        # Validate
        if not _full_valid(colors):
            continue

        # Improve
        colors, distinct = _improve_by_recoloring(colors, min(time.time() + 3.0, deadline - 6.0), rng)

        if distinct > best_distinct:
            best_distinct = distinct
            best_colors = colors.copy()

        num_starts += 1
        # Stop early if we've hit a high bar
        if best_distinct >= 50:
            break

    # 3) Final polish: try to swap in unused colors
    if best_colors is not None and time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        best_colors, best_distinct = _improve_by_recoloring(best_colors, polish_deadline, rng)

    if best_colors is None:
        best_colors = _base_valid_coloring()

    return best_colors.tolist()
