import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 120s kill

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    break
            else:
                return False
        return True

    def build_structures(n):
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        # Bitsets for fast violation counting
        quad_bitsets = []
        for qt in quad_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quad_bitsets.append(bs)

        quint_bitsets = []
        for qt in quint_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quint_bitsets.append(bs)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
                quad_bitsets, quint_bitsets)

    def count_violations(red_mask, quad_bitsets, quint_bitsets):
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
        return rk4, bk5

    def local_search(n, init_colors, deadline):
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = build_structures(n)

        color_arr = np.array(init_colors, dtype=np.int8)
        red_mask = 0
        for i in range(num_triples):
            if color_arr[i] == 0:
                red_mask |= (1 << i)

        rk4, bk5 = count_violations(red_mask, quad_bitsets, quint_bitsets)
        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        # Cache of violating structures for guided selection
        viol_quads = set()
        viol_quints = set()
        for qi in range(num_quads):
            if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                viol_quads.add(qi)
        for qi in range(num_quints):
            if (red_mask & quint_bitsets[qi]) == 0:
                viol_quints.add(qi)

        max_iters = 300000
        temp_start = 2.0
        temp_end = 0.001

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip, biased toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.45:
                # Pick from violating red K4s
                for _ in range(20):
                    if viol_quads:
                        qi = random.choice(list(viol_quads))
                        qt = quad_triples_idx[qi]
                        if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                            tidx = qt[random.randint(0, 3)]
                            break
            elif bk5 > 0 and r < 0.85:
                # Pick from violating blue K5s
                for _ in range(20):
                    if viol_quints:
                        qi = random.choice(list(viol_quints))
                        qt = quint_triples_idx[qi]
                        if (red_mask & quint_bitsets[qi]) == 0:
                            tidx = qt[random.randint(0, 9)]
                            break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]

            # Compute delta_rk4: quads where other 3 are red
            delta_rk4 = 0
            for qi in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                others_all_red = True
                for p in range(4):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 0:
                        others_all_red = False
                        break
                if others_all_red:
                    if old_color == 0:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1

            # Compute delta_bk5: quints where other 9 are blue
            delta_bk5 = 0
            for qi in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                others_all_blue = True
                for p in range(10):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 1:
                        others_all_blue = False
                        break
                if others_all_blue:
                    if old_color == 1:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                new_color = 1 - old_color
                color_arr[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5
                if rk4 < 0:
                    rk4 = 0
                if bk5 < 0:
                    bk5 = 0

                # Maintain violation caches
                if delta_rk4 < 0:
                    for qi in triple_in_quads[tidx]:
                        qt = quad_triples_idx[qi]
                        if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                            viol_quads.add(qi)
                        elif qi in viol_quads:
                            viol_quads.discard(qi)
                if delta_rk4 > 0:
                    for qi in triple_in_quads[tidx]:
                        if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                            viol_quads.add(qi)
                if delta_bk5 < 0:
                    for qi in triple_in_quints[tidx]:
                        qt = quint_triples_idx[qi]
                        if (red_mask & quint_bitsets[qi]) == 0:
                            viol_quints.add(qi)
                        elif qi in viol_quints:
                            viol_quints.discard(qi)
                if delta_bk5 > 0:
                    for qi in triple_in_quints[tidx]:
                        if (red_mask & quint_bitsets[qi]) == 0:
                            viol_quints.add(qi)

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return None, False

    # Try n from large down to 10; stop when a valid coloring is found
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break

        num_triples = len(list(combinations(range(n), 3)))

        # Multiple random restarts with varying red probability
        for restart in range(15):
            if time.time() > deadline - 3:
                break

            # Balance red/blue so neither color is too dense
            p_red = 0.15 + 0.01 * restart
            init_colors = np.random.binomial(1, p_red, num_triples).tolist()

            colors, ok = local_search(n, init_colors, deadline)
            if ok and colors is not None:
                # Verify before returning
                if is_valid(n, colors):
                    best_n = n
                    best_colors = colors
                    return {'n': n, 'colors': colors}

    # Fallback
    return {'n': best_n, 'colors': best_colors}
