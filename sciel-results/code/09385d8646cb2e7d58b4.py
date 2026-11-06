"""Hypergraph Ramsey R(4,5;3): Simulated Annealing Ensemble with vectorized deltas.
Strategy: many short SA restarts, each with its own temperature schedule.
Energy = (# red K4) + (# blue K5). Vectorized delta computation per flip.
"""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _build_data(n):
    """Build all structural data needed for fast SA."""
    triples = list(combinations(range(n), 3))
    m = len(triples)

    # Precompute triple index lookup: (a,b,c) -> flat index
    # Use combinatorial number system
    def t_idx(a, b, c):
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt

    # Build quads: each quad is 4 triple indices
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
        quads.append(idxs)
    n_quads = len(quads)

    # Build pents: each pent is 10 triple indices
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(t_idx(t[0], t[1], t[2]))
        pents.append(idxs)
    n_pents = len(pents)

    # For each triple, which quads and pents it belongs to
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    # Convert to numpy arrays for speed
    tri_quads_np = np.array([qs if qs else [-1] for qs in tri_quads], dtype=np.int32)
    tri_pents_np = np.array([ps if ps else [-1] for ps in tri_pents], dtype=np.int32)

    # Remove -1 sentinels: create a mask and compact
    # Actually, let's just keep them and mask during computation
    # Better: use variable-length approach with pre-allocated arrays

    return triples, m, quads, pents, tri_quads, tri_pents, tri_quads_np, tri_pents_np, n_quads, n_pents


def _initial_counts(colors, quads, pents):
    """Compute quad_red counts and pent_blue counts."""
    colors = np.array(colors, dtype=np.int8)
    quad_red = np.zeros(len(quads), dtype=np.int16)
    for qi, idxs in enumerate(quads):
        quad_red[qi] = np.sum(colors[np.array(idxs)] == 0)
    pent_blue = np.zeros(len(pents), dtype=np.int16)
    for pi, idxs in enumerate(pents):
        pent_blue[pi] = np.sum(colors[np.array(idxs)] == 1)
    return quad_red, pent_blue


def _energy(quad_red, pent_blue):
    return int(np.sum(quad_red == 4)) + int(np.sum(pent_blue == 10))


def _sa_run(colors_init, quads, pents, tri_quads_np, tri_pents_np,
            n_quads, n_pents, time_budget, seed):
    """Single SA run with vectorized delta computation."""
    rng = np.random.RandomState(seed)
    m = len(colors_init)
    colors = np.array(colors_init, dtype=np.int8).copy()
    quad_red, pent_blue = _initial_counts(colors, quads, pents)
    E = _energy(quad_red, pent_blue)

    best_E = E
    best_colors = colors.copy()

    # Temperature schedule: geometric cooling
    T0 = 2.0
    Tend = 0.01
    # Estimate steps: ~1ms per step for n<=34, so ~5000 steps in 5s
    # We'll just run until time budget
    t_end = time.time() + time_budget

    step = 0
    while time.time() < t_end:
        step += 1
        # Cool temperature
        T = T0 * (Tend / T0) ** (step / max(1, int(time_budget * 2000)))

        # Pick random triple to flip
        t = rng.randint(m)
        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1

        # Compute delta E using precomputed memberships
        aq = tri_quads_np[t]
        ap = tri_pents_np[t]

        dE = 0

        # Quad contributions
        if len(aq) > 0 and aq[0] != -1:
            # Filter out -1
            valid = aq[aq != -1]
            if len(valid) > 0:
                old_q = quad_red[valid]
                new_q = old_q + delta_red
                # Contribution: 1 if count==4, 0 otherwise
                old_c = (old_q == 4).astype(np.int32)
                new_c = (new_q == 4).astype(np.int32)
                dE += int(np.sum(new_c - old_c))

        # Pent contributions
        if len(ap) > 0 and ap[0] != -1:
            valid = ap[ap != -1]
            if len(valid) > 0:
                old_p = pent_blue[valid]
                new_p = old_p + delta_blue
                old_c = (old_p == 10).astype(np.int32)
                new_c = (new_p == 10).astype(np.int32)
                dE += int(np.sum(new_c - old_c))

        # Accept/reject
        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            colors[t] = new
            if len(aq) > 0 and aq[0] != -1:
                valid = aq[aq != -1]
                if len(valid) > 0:
                    quad_red[valid] += delta_red
            if len(ap) > 0 and ap[0] != -1:
                valid = ap[ap != -1]
                if len(valid) > 0:
                    pent_blue[valid] += delta_blue
            E += dE
            if E < best_E:
                best_E = E
                best_colors = colors.copy()
                if best_E == 0:
                    return best_colors.tolist(), 0
        else:
            pass

    return best_colors.tolist(), best_E


def _sa_ensemble(n, time_budget, seed=0):
    """Run multiple SA restarts, return best result."""
    triples, m, quads, pents, tri_quads, tri_pents, tri_quads_np, tri_pents_np, n_quads, n_pents = _build_data(n)

    t_end = time.time() + time_budget
    best_colors = None
    best_E = float('inf')

    # Multiple restarts with different seeds and init strategies
    restart_count = 0
    while time.time() < t_end:
        restart_count += 1
        remaining = t_end - time.time()
        if remaining < 1.0:
            break
        sa_time = min(remaining * 0.25, 4.0)  # each SA gets up to 4s

        # Diverse initialization
        if restart_count % 3 == 1:
            # Random init
            colors_init = (np.random.RandomState(seed + restart_count).rand(m) < 0.3).astype(np.int8).tolist()
        elif restart_count % 3 == 2:
            # Biased random (more red to avoid blue K5)
            p_red = 0.4 + 0.2 * np.random.RandomState(seed + restart_count).random()
            colors_init = (np.random.RandomState(seed + restart_count + 1000).rand(m) < p_red).astype(np.int8).tolist()
        else:
            # Structured: based on vertex parity
            colors_init = []
            for a, b, c in triples:
                if (a ^ b ^ c) % 2 == 0:
                    colors_init.append(0)
                else:
                    colors_init.append(1)

        colors, E = _sa_run(colors_init, quads, pents, tri_quads_np, tri_pents_np,
                            n_quads, n_pents, sa_time, seed + restart_count * 7)
        if E < best_E:
            best_E = E
            best_colors = colors
            if best_E == 0:
                return best_colors

    return best_colors


def _try_extend(n_base, colors_base, n_target, time_budget):
    """Try to extend a valid coloring on n_base vertices to n_target by adding vertices.
    New triples involving new vertices are colored by SA."""
    if n_base >= n_target:
        return colors_base

    m_target = n_target * (n_target - 1) * (n_target - 2) // 6
    m_base = len(colors_base)

    # Start with base colors + random for new triples
    rng = np.random.RandomState(42)
    colors = list(colors_base) + [int(rng.random() < 0.3) for _ in range(m_target - m_base)]

    # Run SA on the full n_target instance
    result = _sa_ensemble(n_target, time_budget, seed=999)
    return result


def construct():
    deadline = _START + _DEADLINE - 3.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Strategy: try increasing n from small to large, using SA ensemble
    # For each n, if we find a valid coloring, keep it and try to extend
    for n in range(10, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate more time for larger n (harder problems)
        alloc = min(remaining * 0.7, 30.0) if n <= 25 else min(remaining * 0.5, 40.0)

        colors = _sa_ensemble(n, alloc, seed=100 + n)
        if colors is not None:
            # Verify validity
            # Quick check: build data and count violations
            triples, m, quads, pents, _, _, _, _, _, _ = _build_data(n)
            quad_red, pent_blue = _initial_counts(colors, quads, pents)
            if _energy(quad_red, pent_blue) == 0:
                best_n = n
                best_colors = colors
            else:
                # Not valid, but maybe close; try a few more restarts
                if remaining > 10.0:
                    colors2 = _sa_ensemble(n, min(remaining * 0.3, 20.0), seed=500 + n)
                    if colors2 is not None:
                        quad_red2, pent_blue2 = _initial_counts(colors2, quads, pents)
                        if _energy(quad_red2, pent_blue2) == 0:
                            best_n = n
                            best_colors = colors2

    return {'n': best_n, 'colors': best_colors}
