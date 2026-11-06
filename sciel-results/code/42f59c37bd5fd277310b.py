"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0  # seconds; return with margin


def _triples(n):
    """List of triples (a,b,c), a<b<c, in lexicographic order."""
    return list(combinations(range(n), 3))


def _triples_to_idx(n):
    """Map (a,b,c) -> index in lexicographic triple list."""
    triples = _triples(n)
    m = len(triples)
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx, m


def _build_quads(n, triples):
    """For each 4-set, list of its 4 triple indices. Returns list of lists."""
    m = len(triples)
    # Precompute triple index via formula for speed
    def t_idx(a, b, c):
        # lexicographic index of (a,b,c) among triples of n
        # count triples (x,y,z) with x<a, or x==a and y<b, or x==a,y==b,z<c
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
        quads.append(idxs)
    return quads


def _build_pents(n, triples):
    """For each 5-set, list of its 10 triple indices."""
    def t_idx(a, b, c):
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    pents = []
    for p in combinations(range(n), 5):
        a, b, c, d, e = p
        idxs = []
        for t in combinations(p, 3):
            idxs.append(t_idx(*t))
        pents.append(idxs)
    return pents


def _count_violations(colors, quads, pents):
    """Count red K4 (all 4 red) and blue K5 (all 10 blue)."""
    red_k4 = 0
    for idxs in quads:
        s = 0
        for i in idxs:
            s += colors[i] == 0
        if s == 4:
            red_k4 += 1
    blue_k5 = 0
    for idxs in pents:
        s = 0
        for i in idxs:
            s += colors[i] == 1
        if s == 10:
            blue_k5 += 1
    return red_k4, blue_k5


def _local_search(colors, quads, pents, time_budget):
    """Flip triples to reduce violations. Returns (colors, red_k4, blue_k5)."""
    colors = colors.copy()
    m = len(colors)
    # Build for each triple, the list of (quad_idx, sign) and (pent_idx, sign) it affects
    # sign: +1 if flipping to red increases red count, etc. We'll just recompute affected counts.
    # For speed, maintain quad_red_counts and pent_blue_counts arrays.
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
    total_v = red_k4 + blue_k5
    if total_v == 0:
        return colors, red_k4, blue_k5

    # Precompute for each triple the quads and pents it belongs to
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    t_end = time.time() + time_budget
    order = list(range(m))
    random.shuffle(order)
    improved = True
    while improved and time.time() < t_end:
        improved = False
        for t in order:
            if time.time() >= t_end:
                break
            cur = colors[t]
            # Try flipping t
            new = 1 - cur
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            # Compute change in violations
            dq = 0
            for qi in tri_quads[t]:
                old = quad_red[qi]
                new_q = old + delta_red
                if old == 4:
                    dq -= 1
                if new_q == 4:
                    dq += 1
            dp = 0
            for pi in tri_pents[t]:
                old = pent_blue[pi]
                new_p = old + delta_blue
                if old == 10:
                    dp -= 1
                if new_p == 10:
                    dp += 1
            if dq + dp < 0:
                # Apply flip
                colors[t] = new
                for qi in tri_quads[t]:
                    quad_red[qi] += delta_red
                for pi in tri_pents[t]:
                    pent_blue[pi] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                improved = True
        # reshuffle occasionally
        random.shuffle(order)
    return colors, red_k4, blue_k5


def _random_valid(n, time_budget):
    """Try to find a valid colouring on n vertices via random init + local search."""
    triples, idx, m = _triples_to_idx(n)
    quads = _build_quads(n, triples)
    pents = _build_pents(n, triples)
    attempts = 0
    while time.time() < _START + time_budget:
        attempts += 1
        # Random init biased
        p_red = 0.35
        colors = (np.random.rand(m) < p_red).astype(np.int8)
        colors = colors.tolist()
        colors, r4, b5 = _local_search(colors, quads, pents, 2.0)
        if r4 == 0 and b5 == 0:
            return colors
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0  # safety margin
    best_n = 9
    best_colors = None
    # Seed with the known small valid construction
    from itertools import combinations as _c
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in _c(range(n0), 3)]
    best_n = n0

    # Try to grow n from 10 upward, using time budget
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        # Estimate time per n
        remaining = deadline - time.time()
        if n <= 20:
            budget = min(remaining, 15.0)
        elif n <= 28:
            budget = min(remaining, 25.0)
        else:
            budget = min(remaining, 35.0)
        if budget < 3.0:
            break
        colors = _random_valid(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
        # If we found valid at n, try n+1 with remaining time
        if time.time() >= deadline:
            break

    return {'n': best_n, 'colors': best_colors}
