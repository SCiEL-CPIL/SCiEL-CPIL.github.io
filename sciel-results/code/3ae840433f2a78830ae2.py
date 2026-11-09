"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        # Precompute structures
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # For each quad (4-set), the 4 triples it contains
        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        # For each quint (5-set), the 10 triples it contains
        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        # Bitset representation for fast all-equal checks
        # red_sets: for each quad, bitset of its 4 triple indices
        # We'll use Python integers as bitsets for speed
        quad_red_bitsets = []
        for qt in quad_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quad_red_bitsets.append(bs)

        quint_blue_bitsets = []
        for qt in quint_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quint_blue_bitsets.append(bs)

        # Maintain: red_mask[i] = 1 if triple i is red (color 0)
        # For a quad to be all-red: all 4 bits in red_mask must be 1
        # For a quint to be all-blue: all 10 bits in (not red_mask) must be 1

        found = False
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Initialize coloring
            if restart == 0:
                color_arr = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                color_arr = np.array([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            elif restart == 3:
                # Random with bias toward red
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 0
            elif restart == 4:
                # Random with bias toward blue
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 1
            else:
                # Perturb modular
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.15
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]

            # Build red_mask as a Python int (bit i = 1 if triple i is red)
            red_mask = 0
            for i in range(num_triples):
                if color_arr[i] == 0:
                    red_mask |= (1 << i)

            # Count violations using bitset operations
            def count_violations(rm):
                rk4 = 0
                for qbs in quad_red_bitsets:
                    if (rm & qbs) == qbs:
                        rk4 += 1
                # Blue K5: all 10 triples blue means none are red in that quint
                bk5 = 0
                for qbs in quint_blue_bitsets:
                    if (rm & qbs) == 0:
                        bk5 += 1
                return rk4, bk5

            rk4, bk5 = count_violations(red_mask)

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search with simulated annealing
            max_iters = 50000
            temp_start = 1.5
            temp_end = 0.005
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Choose a triple to flip - bias toward violating structures
                tidx = -1
                r = random.random()
                if rk4 > 0 and r < 0.6:
                    # Pick from a violating red K4
                    for _ in range(50):
                        qi = random.randint(0, num_quads - 1)
                        qt = quad_triples_idx[qi]
                        if all(color_arr[t] == 0 for t in qt):
                            tidx = qt[random.randint(0, 3)]
                            break
                elif bk5 > 0 and r < 0.9:
                    # Pick from a violating blue K5
                    for _ in range(50):
                        qi = random.randint(0, num_quints - 1)
                        qt = quint_triples_idx[qi]
                        if all(color_arr[t] == 1 for t in qt):
                            tidx = qt[random.randint(0, 9)]
                            break

                if tidx == -1:
                    tidx = random.randint(0, num_triples - 1)

                old_color = color_arr[tidx]

                # Compute delta_rk4 and delta_bk5 incrementally
                delta_rk4 = 0
                for (qi, pos) in triple_in_quads[tidx]:
                    qt = quad_triples_idx[qi]
                    others_red = True
                    for p in range(4):
                        if p == pos:
                            continue
                        if color_arr[qt[p]] != 0:
                            others_red = False
                            break
                    if old_color == 0 and others_red:
                        delta_rk4 -= 1  # removing this red breaks the K4
                    elif old_color == 1 and others_red:
                        delta_rk4 += 1  # adding this red creates a K4

                delta_bk5 = 0
                for (qi, pos) in triple_in_quints[tidx]:
                    qt = quint_triples_idx[qi]
                    others_blue = True
                    for p in range(10):
                        if p == pos:
                            continue
                        if color_arr[qt[p]] != 1:
                            others_blue = False
                            break
                    if old_color == 1 and others_blue:
                        delta_bk5 -= 1  # removing this blue breaks the K5
                    elif old_color == 0 and others_blue:
                        delta_bk5 += 1  # adding this blue creates a K5

                old_total = rk4 + bk5
                new_total = (rk4 + delta_rk4) + (bk5 + delta_bk5)
                delta = new_total - old_total

                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    # Accept the flip
                    new_color = 1 - old_color
                    color_arr[tidx] = new_color
                    if new_color == 0:
                        red_mask |= (1 << tidx)
                    else:
                        red_mask &= ~(1 << tidx)
                    rk4 += delta_rk4
                    bk5 += delta_bk5

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_arr.tolist()
            # Quick verification
            if is_valid(target_n, best_colors):
                pass
            else:
                best_n = 9
                best_colors = mod_color(9, 3)

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
