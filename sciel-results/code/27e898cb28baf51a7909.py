"""Hypergraph Ramsey R(4,5;3): Simulated Annealing with incremental energy updates
and multi-start strategy targeting n=34 directly."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _t_idx(a, b, c, n):
    """Lexicographic index of triple (a,b,c) among all triples of n vertices."""
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
    """Build all structural data for n vertices."""
    triples = list(combinations(range(n), 3))
    m = len(triples)

    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    n_quads = len(quads)

    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    n_pents = len(pents)

    # Inverse maps: for each triple, which quads/pents contain it
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    return triples, m, quads, pents, tri_quads, tri_pents


def _initial_energy(colors, quads, pents):
    """Compute initial quad_red and pent_blue counts."""
    m = len(colors)
    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    return quad_red, pent_blue, red_k4, blue_k5


def _sa_search(colors_init, quads, pents, tri_quads, tri_pents, time_budget, seed=0):
    """
    Simulated Annealing with incremental energy updates.
    Energy = (# red K4) + (# blue K5). We minimize this.
    """
    rng = np.random.RandomState(seed)
    colors = np.array(colors_init, dtype=np.int8)
    m = len(colors)
    t_end = time.time() + time_budget

    quad_red, pent_blue, red_k4, blue_k5 = _initial_energy(colors, quads, pents)
    energy = red_k4 + blue_k5

    # Pre-convert tri_quads and tri_pents to numpy arrays for speed
    tri_quads_np = [np.array(qs, dtype=np.int32) if qs else np.array([], dtype=np.int32) for qs in tri_quads]
    tri_pents_np = [np.array(ps, dtype=np.int32) if ps else np.array([], dtype=np.int32) for ps in tri_pents]

    best_energy = energy
    best_colors = colors.copy()
    best_red_k4 = red_k4
    best_blue_k5 = blue_k5

    # SA parameters
    T_init = max(energy, 5.0)
    T_min = 0.01
    alpha = 0.995  # cooling rate per step
    T = T_init
    step = 0
    stall = 0
    MAX_STALL = 3000

    while time.time() < t_end:
        # Pick a random triple to flip
        t = rng.randint(m)
        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1

        aq = tri_quads_np[t]
        ap = tri_pents_np[t]

        dq = 0
        if len(aq) > 0:
            old_q = quad_red[aq]
            new_q = old_q + delta_red
            old_contrib = np.maximum(old_q - 3, 0)
            new_contrib = np.maximum(new_q - 3, 0)
            dq = int(np.sum(new_contrib - old_contrib))

        dp = 0
        if len(ap) > 0:
            old_p = pent_blue[ap]
            new_p = old_p + delta_blue
            old_contrib = np.maximum(old_p - 9, 0)
            new_contrib = np.maximum(new_p - 9, 0)
            dp = int(np.sum(new_contrib - old_contrib))

        delta_E = dq + dp

        # Metropolis acceptance
        if delta_E <= 0:
            accept = True
        else:
            if T > T_min:
                accept = (rng.random() < np.exp(-delta_E / T))
            else:
                accept = False

        if accept:
            colors[t] = new
            if len(aq) > 0:
                quad_red[aq] += delta_red
            if len(ap) > 0:
                pent_blue[ap] += delta_blue
            energy += delta_E

            if energy < best_energy:
                best_energy = energy
                best_colors = colors.copy()
                best_red_k4 = int(np.sum(quad_red == 4))
                best_blue_k5 = int(np.sum(pent_blue == 10))
                stall = 0
            else:
                stall += 1
        else:
            stall += 1

        # Cool down
        if step % 10 == 0:
            T *= alpha

        step += 1

        # Periodic exact recount to guard against drift
        if step % 10000 == 0:
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5

        # Restart on stagnation
        if stall > MAX_STALL:
            # Perturb: flip 3-10 random triples
            n_flip = rng.randint(3, 10)
            for _ in range(n_flip):
                t2 = rng.randint(m)
                colors[t2] = 1 - colors[t2]
                delta_red2 = 1 if colors[t2] == 0 else -1
                delta_blue2 = 1 if colors[t2] == 1 else -1
                aq2 = tri_quads_np[t2]
                ap2 = tri_pents_np[t2]
                if len(aq2) > 0:
                    quad_red[aq2] += delta_red2
                if len(ap2) > 0:
                    pent_blue[ap2] += delta_blue2
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            T = max(energy, 3.0)
            stall = 0

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    if red_k4 + blue_k5 < best_energy:
        return best_colors.tolist(), red_k4, blue_k5
    return best_colors.tolist(), best_red_k4, best_blue_k5


def _find_valid(n, time_budget, seed=0):
    """Try multiple random initializations with SA to find valid coloring on n vertices."""
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)
    t_end = _START + time_budget
    attempts = 0

    # Try different red probabilities for diversity
    p_red_list = [0.3, 0.35, 0.4, 0.25, 0.45, 0.32, 0.38]

    while time.time() < t_end:
        attempts += 1
        remaining = t_end - time.time()
        if remaining < 2.0:
            break

        # Allocate time per attempt
        sa_time = min(remaining * 0.4, 8.0)
        if attempts > 5:
            sa_time = min(remaining * 0.5, 10.0)

        p_red = p_red_list[attempts % len(p_red_list)] + random.uniform(-0.05, 0.05)
        p_red = max(0.1, min(0.6, p_red))
        colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()

        colors, r4, b5 = _sa_search(
            colors, quads, pents, tri_quads, tri_pents,
            sa_time, seed=seed + attempts * 7)

        if r4 == 0 and b5 == 0:
            return colors

    return None


def _try_extend(n_base, colors_base, n_target, time_budget, seed=0):
    """
    Try to extend a valid coloring on n_base vertices to n_target vertices.
    New vertices get random colors for their new triples, then SA refines.
    """
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n_target)

    # Initialize: for triples entirely in [0, n_base), use base coloring
    # For triples involving new vertices, use random
    colors = np.zeros(m, dtype=np.int8)
    base_idx = 0
    for a, b, c in combinations(range(n_base), 3):
        idx = _t_idx(a, b, c, n_target)
        colors[idx] = colors_base[base_idx]
        base_idx += 1

    # Fill remaining (triples with at least one vertex >= n_base) randomly
    for i in range(m):
        a, b, c = triples[i]
        if c >= n_base:
            colors[i] = rng.randint(0, 2) if False else 0  # placeholder

    # Actually fill properly
    rng = np.random.RandomState(seed)
    for i in range(m):
        a, b, c = triples[i]
        if c >= n_base:
            colors[i] = rng.randint(0, 2)

    t_end = _START + time_budget
    attempts = 0
    while time.time() < t_end:
        attempts += 1
        remaining = t_end - time.time()
        if remaining < 2.0:
            break
        sa_time = min(remaining * 0.5, 10.0)
        colors, r4, b5 = _sa_search(
            colors.tolist(), quads, pents, tri_quads, tri_pents,
            sa_time, seed=seed + attempts * 13)
        if r4 == 0 and b5 == 0:
            return colors.tolist()
        # Perturb for next attempt
        n_flip = rng.randint(10, 30)
        for _ in range(n_flip):
            colors[rng.randint(m)] = 1 - colors[rng.randint(m)]

    return None


def construct():
    deadline = _START + _DEADLINE
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Strategy: try n=34 first (the target), then work down if needed
    # This is fundamentally different from the current approach which tries small n first

    # Phase 1: Directly attempt n=34 with multiple SA runs
    if time.time() < deadline - 30.0:
        remaining = deadline - time.time()
        colors = _find_valid(34, min(remaining, 60.0), seed=42)
        if colors is not None:
            best_n = 34
            best_colors = colors
            return {'n': best_n, 'colors': best_colors}

    # Phase 2: Try n=33
    if best_n < 33 and time.time() < deadline - 20.0:
        remaining = deadline - time.time()
        colors = _find_valid(33, min(remaining, 40.0), seed=100)
        if colors is not None:
            best_n = 33
            best_colors = colors

    # Phase 3: Try n=32
    if best_n < 32 and time.time() < deadline - 15.0:
        remaining = deadline - time.time()
        colors = _find_valid(32, min(remaining, 30.0), seed=200)
        if colors is not None:
            best_n = 32
            best_colors = colors

    # Phase 4: Try n=31
    if best_n < 31 and time.time() < deadline - 10.0:
        remaining = deadline - time.time()
        colors = _find_valid(31, min(remaining, 20.0), seed=300)
        if colors is not None:
            best_n = 31
            best_colors = colors

    # Phase 5: Try n=30
    if best_n < 30 and time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        colors = _find_valid(30, min(remaining, 10.0), seed=400)
        if colors is not None:
            best_n = 30
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
