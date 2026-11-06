"""Hypergraph Ramsey R(4,5;3): deterministic GF(2^k) algebraic seed + violation-driven
stochastic local search (with restarts) to maximise n for a 2-colouring of triples
with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0


def _t_idx(a, b, c, n):
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
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


def _gf2k_coloring(k):
    """
    Deterministic algebraic colouring on n = 2^k vertices.
    Each vertex is a field element of GF(2^k). For a triple {a,b,c},
    colour red (0) iff the sum a+b+c (field addition) is a non-zero square,
    otherwise blue (1). This is a structured, low-violation starting point.
    """
    n = 1 << k
    # Build GF(2^k) using a fixed primitive polynomial per k
    polys = {
        1: 0b11,   # x+1 (GF(2))
        2: 0b111,  # x^2+x+1
        3: 0b10011,  # x^4+x+1
        4: 0b100011,  # x^5+x^2+1
        5: 0b100010011,  # x^6+x+1
        6: 0b100001001,  # x^7+x^3+1
    }
    # We only need addition (XOR) and a "square" test. In GF(2^k), every element
    # is a square (the Frobenius map x->x^2 is a bijection), so instead we use
    # a trace-based colouring: red iff Tr(a+b+c) == 0, where Tr is the field trace.
    # Trace: Tr(x) = x + x^2 + x^4 + x^8 + ... (mod the field, summed over k terms).
    p = polys.get(k, 0b10011)
    # Precompute powers of 2 for trace: 2^0, 2^1, 2^2, ..., 2^(k-1) as field elements
    # Field multiplication by 2 (x) via reduction
    def mul2(x):
        y = x << 1
        if y & (1 << k):
            y ^= p
        return y & ((1 << k) - 1)

    # trace(x) = sum_{i=0}^{k-1} x^(2^i); compute iteratively
    def trace(x):
        t = 0
        cur = x
        for _ in range(k):
            t ^= (cur & 1)  # parity of constant bits
            cur = mul2(cur)
        return t

    colors = []
    for a, b, c in combinations(range(n), 3):
        s = a ^ b ^ c  # field addition is XOR
        if trace(s) == 0:
            colors.append(0)  # red
        else:
            colors.append(1)  # blue
    return n, colors


def _local_search_violation_driven(colors, quads, pents, tri_quads, tri_pents,
                                   time_budget, seed=0):
    """
    Stochastic local search that only flips triples belonging to currently
    violating red K4 or blue K5. Restarts on stagnation.
    """
    rng = np.random.RandomState(seed)
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    t_end = time.time() + time_budget

    quad_red, pent_blue, red_k4, blue_k5 = _initial_energy(colors, quads, pents)
    energy = red_k4 + blue_k5

    tri_quads_np = [np.array(qs, dtype=np.int32) if qs else np.array([], dtype=np.int32) for qs in tri_quads]
    tri_pents_np = [np.array(ps, dtype=np.int32) if ps else np.array([], dtype=np.int32) for ps in tri_pents]

    best_energy = energy
    best_colors = colors.copy()
    stall = 0
    MAX_STALL = 2000

    while time.time() < t_end:
        if red_k4 == 0 and blue_k5 == 0:
            return colors.tolist(), 0, 0

        # Build candidate set: triples in violating quads/pents
        cand = set()
        for qi, idxs in enumerate(quads):
            if quad_red[qi] == 4:
                cand.update(idxs)
        for pi, idxs in enumerate(pents):
            if pent_blue[pi] == 10:
                cand.update(idxs)

        if not cand:
            # No violating structures but energy says otherwise? Force random flip.
            t = rng.randint(m)
        else:
            cand_list = list(cand)
            t = cand_list[rng.randint(len(cand_list))]

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

        # Accept if it reduces energy, or with small probability if neutral
        if delta_E <= 0:
            colors[t] = new
            if len(aq) > 0:
                quad_red[aq] += delta_red
            if len(ap) > 0:
                pent_blue[ap] += delta_blue
            energy += delta_E
            if energy < best_energy:
                best_energy = energy
                best_colors = colors.copy()
                stall = 0
            else:
                stall += 1
        else:
            # Occasionally accept a worsening move to escape local minima
            if rng.random() < 0.05:
                colors[t] = new
                if len(aq) > 0:
                    quad_red[aq] += delta_red
                if len(ap) > 0:
                    pent_blue[ap] += delta_blue
                energy += delta_E
                stall += 1
            else:
                stall += 1

        # Recompute exact counts periodically
        if stall % 500 == 0:
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5

        # Stagnation: restart with perturbation
        if stall > MAX_STALL:
            # Perturb: flip 5-15 random triples
            n_flip = rng.randint(5, 15)
            for _ in range(n_flip):
                t2 = rng.randint(m)
                colors[t2] = 1 - colors[t2]
                # Update counts
                delta_red = 1 if colors[t2] == 0 else -1
                delta_blue = 1 if colors[t2] == 1 else -1
                aq2 = tri_quads_np[t2]
                ap2 = tri_pents_np[t2]
                if len(aq2) > 0:
                    quad_red[aq2] += delta_red
                if len(ap2) > 0:
                    pent_blue[ap2] += delta_blue
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            stall = 0

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    if red_k4 + blue_k5 < best_energy:
        return colors.tolist(), red_k4, blue_k5
    return best_colors.tolist(), best_energy, best_energy


def _find_valid(n, time_budget, seed=0):
    """Try to find a valid colouring on n vertices."""
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)

    t_end = _START + time_budget
    attempts = 0

    # Try algebraic seed first if n is a power of 2
    k = None
    if n > 1:
        logn = n.bit_length() - 1
        if (1 << logn) == n:
            k = logn

    if k is not None:
        _, alg_colors = _gf2k_coloring(k)
        colors, r4, b5 = _local_search_violation_driven(
            alg_colors, quads, pents, tri_quads, tri_pents,
            min(time_budget * 0.3, 5.0), seed=seed)
        if r4 == 0 and b5 == 0:
            return colors

    # Then try several random inits
    while time.time() < t_end:
        attempts += 1
        remaining = t_end - time.time()
        if remaining < 1.0:
            break
        sa_time = min(remaining * 0.3, 5.0)
        p_red = 0.25 + 0.25 * random.random()
        colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()
        colors, r4, b5 = _local_search_violation_driven(
            colors, quads, pents, tri_quads, tri_pents,
            sa_time, seed=seed + attempts)
        if r4 == 0 and b5 == 0:
            return colors
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try algebraic seeds for powers of 2 up to 32
    for k in range(3, 7):
        n = 1 << k
        if n > 34:
            break
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        colors = _find_valid(n, min(remaining, 30.0), seed=100 + k)
        if colors is not None:
            best_n = n
            best_colors = colors

    # Now try to extend beyond the best algebraic n, up to 34
    for n in range(best_n + 1, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        colors = _find_valid(n, min(remaining, 40.0), seed=200 + n)
        if colors is not None:
            best_n = n
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
