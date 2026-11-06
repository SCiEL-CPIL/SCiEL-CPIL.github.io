import numpy as np
import time
import itertools
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def build_structure(n):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(quad[j] for j in range(4) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)
        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        return M, triples, quad_triples, quint_triples, nwords, quad_masks, quint_masks

    def init_coloring(n, method, seed):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        rng = np.random.RandomState(seed)
        colors = np.zeros(M, dtype=np.int8)
        if method == 'mod3':
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        elif method == 'mod5':
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + 2*b + 3*c) % 5 < 3 else 1
        elif method == 'xor':
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if ((a ^ b) ^ c) % 2 == 0 else 1
        elif method == 'product':
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a * b + b * c + c * a) % 3 == 0 else 1
        elif method == 'random_biased':
            colors = rng.randint(0, 2, size=M, dtype=np.int8)
            mask = rng.random(M) < 0.65
            colors[mask] = 0
        else:
            colors = rng.randint(0, 2, size=M, dtype=np.int8)
        return colors

    def build_bitsets(colors, nwords):
        M = len(colors)
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for i in range(M):
            bit = np.uint64(1) << np.uint64(i & 63)
            w = i >> 6
            if colors[i] == 0:
                red_set[w] |= bit
            else:
                blue_set[w] |= bit
        return red_set, blue_set

    def get_violations(red_set, blue_set, quad_masks, quint_masks):
        viol_quads = np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]
        viol_quints = np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]
        return viol_quads, viol_quints

    def count_violations(red_set, blue_set, quad_masks, quint_masks):
        r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
        b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
        return int(r4) + int(b5)

    def flip_edge(colors, red_set, blue_set, idx, new_color):
        old = colors[idx]
        if old == new_color:
            return
        bit = np.uint64(1) << np.uint64(idx & 63)
        w = idx >> 6
        if old == 0:
            red_set[w] &= ~bit
        else:
            blue_set[w] &= ~bit
        colors[idx] = new_color
        if new_color == 0:
            red_set[w] |= bit
        else:
            blue_set[w] |= bit

    def greedy_solve(n, time_limit, seed=0):
        rng = np.random.RandomState(seed)
        M, triples, quad_triples, quint_triples, nwords, quad_masks, quint_masks = build_structure(n)
        methods = ['mod3', 'mod5', 'xor', 'product', 'random_biased']
        best_energy = float('inf')
        best_colors = None

        for midx, method in enumerate(methods):
            if time.time() > start_time + time_limit:
                break
            colors = init_coloring(n, method, seed + midx * 100)
            red_set, blue_set = build_bitsets(colors, nwords)
            energy = count_violations(red_set, blue_set, quad_masks, quint_masks)

            for it in range(80000):
                if time.time() > start_time + time_limit:
                    break
                if energy == 0:
                    if energy < best_energy:
                        best_energy = 0
                        best_colors = colors.copy()
                    return best_colors.tolist()

                viol_quads, viol_quints = get_violations(red_set, blue_set, quad_masks, quint_masks)
                if len(viol_quads) + len(viol_quints) == 0:
                    break

                if len(viol_quads) > 0:
                    q = viol_quads[rng.randint(len(viol_quads))]
                    tidxs = quad_triples[q]
                    red_edges = [int(t) for t in tidxs if colors[t] == 0]
                    if red_edges:
                        best_delta = float('inf')
                        best_edge = red_edges[0]
                        for e in red_edges:
                            old_c = colors[e]
                            flip_edge(colors, red_set, blue_set, e, 1)
                            new_energy = count_violations(red_set, blue_set, quad_masks, quint_masks)
                            delta = new_energy - energy
                            flip_edge(colors, red_set, blue_set, e, old_c)
                            if delta < best_delta:
                                best_delta = delta
                                best_edge = e
                        flip_edge(colors, red_set, blue_set, best_edge, 1)
                        energy += best_delta
                elif len(viol_quints) > 0:
                    q = viol_quints[rng.randint(len(viol_quints))]
                    tidxs = quint_triples[q]
                    blue_edges = [int(t) for t in tidxs if colors[t] == 1]
                    if blue_edges:
                        best_delta = float('inf')
                        best_edge = blue_edges[0]
                        for e in blue_edges[:4]:
                            old_c = colors[e]
                            flip_edge(colors, red_set, blue_set, e, 0)
                            new_energy = count_violations(red_set, blue_set, quad_masks, quint_masks)
                            delta = new_energy - energy
                            flip_edge(colors, red_set, blue_set, e, old_c)
                            if delta < best_delta:
                                best_delta = delta
                                best_edge = e
                        flip_edge(colors, red_set, blue_set, best_edge, 0)
                        energy += best_delta

                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors.copy()

            if best_energy == 0:
                return best_colors.tolist()

        return None

    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        per_n_budget = min(remaining - 5.0, 28.0)
        if per_n_budget <= 1.0:
            break
        found = None
        for seed in [0, 42, 123]:
            if time.time() > deadline - 5.0:
                break
            t_limit = min(per_n_budget / 1.5, deadline - time.time() - 5.0)
            if t_limit <= 0.5:
                break
            res = greedy_solve(n, t_limit, seed=seed)
            if res is not None:
                found = res
                break
        if found is not None:
            best_n = n
            best_colors = found
        else:
            break

    if best_n < 10:
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    return {'n': best_n, 'colors': best_colors}
