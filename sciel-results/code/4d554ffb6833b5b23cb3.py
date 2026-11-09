"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit

    best_n = 9
    # Fallback valid coloring: color by (a+b+c) mod 2
    best_colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # We'll try to find valid colorings for increasing n
    # Use bitset-accelerated local search with incremental violation counting

    for target_n in range(10, 35):
        if time.time() > deadline - 5:
            break

        found = False
        # Multiple random restarts
        for restart in range(200):
            if time.time() > deadline - 3:
                break

            # Initialize: random coloring
            colors = np.random.randint(0, 2, size=(target_n, target_n, target_n), dtype=np.int8)
            # Only upper triangle matters; set lower to 0 (won't be read)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors[i, j, :] = 0
                    colors[i, :, j] = 0
                    colors[:, i, j] = 0

            # Set valid entries
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    for k in range(j + 1, target_n):
                        colors[i, j, k] = np.random.randint(0, 2)

            # Build index mapping
            triple_indices = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    for k in range(j + 1, target_n):
                        triple_indices.append((i, j, k))
            num_triples = len(triple_indices)

            # Precompute: for each vertex, list of triples containing it
            # For red K4: for each pair (i,j), list of pairs (k,l) with k,l > j such that (i,j,k),(i,j,l),(i,k,l),(j,k,l)
            # Actually let's use a different approach:
            # For each 4-set, check if all 4 triples are red
            # For each 5-set, check if all 10 triples are blue

            # Incremental counting approach:
            # red_k4_count: number of 4-sets where all 4 triples are red
            # blue_k5_count: number of 5-sets where all 10 triples are blue

            # Precompute all 4-sets and their 4 triples
            all_quads = list(combinations(range(target_n), 4))
            quad_triples = []
            for quad in all_quads:
                a, b, c, d = quad
                quad_triples.append([(a,b,c), (a,b,d), (a,c,d), (b,c,d)])

            all_quints = list(combinations(range(target_n), 5))
            quint_triples = []
            for quint in all_quints:
                quint_triples.append(list(combinations(quint, 3)))

            # Build triple -> index map for O(1) lookup
            triple_to_idx = {}
            for idx, (a, b, c) in enumerate(triple_indices):
                triple_to_idx[(a, b, c)] = idx

            # Current colors as flat list for easy access
            color_flat = np.array([colors[a, b, c] for (a, b, c) in triple_indices], dtype=np.int8)

            # Count initial violations
            def count_violations(cf):
                rk4 = 0
                for qt in quad_triples:
                    if all(cf[triple_to_idx[t]] == 0 for t in qt):
                        rk4 += 1
                bk5 = 0
                for qt in quint_triples:
                    if all(cf[triple_to_idx[t]] == 1 for t in qt):
                        bk5 += 1
                return rk4, bk5

            rk4, bk5 = count_violations(color_flat)

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search: flip triples to reduce violations
            max_iters = 50000
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                # Choose a triple to flip: pick from a violating structure
                if rk4 > 0:
                    # Find a random red K4
                    # Pick random quad until we find one
                    for _ in range(50):
                        qi = random.randint(0, len(quad_triples) - 1)
                        qt = quad_triples[qi]
                        if all(color_flat[triple_to_idx[t]] == 0 for t in qt):
                            # Flip one random triple in this quad
                            ti = random.randint(0, 3)
                            triple = qt[ti]
                            tidx = triple_to_idx[triple]
                            break
                    else:
                        ti = random.randint(0, num_triples - 1)
                        tidx = ti
                else:
                    # Find a random blue K5
                    for _ in range(50):
                        qi = random.randint(0, len(quint_triples) - 1)
                        qt = quint_triples[qi]
                        if all(color_flat[triple_to_idx[t]] == 1 for t in qt):
                            ti = random.randint(0, 9)
                            triple = qt[ti]
                            tidx = triple_to_idx[triple]
                            break
                    else:
                        tidx = random.randint(0, num_triples - 1)

                # Compute effect of flipping tidx
                old_color = color_flat[tidx]
                new_color = 1 - old_color

                # For each quad containing this triple, check if flipping changes rk4
                # A quad contributes to rk4 iff all 4 of its triples are red (color 0)
                # If we flip a triple from 0->1: quad was red only if all 4 were 0; now it's not
                # If we flip a triple from 1->0: quad becomes red only if the other 3 are already 0

                # We need to know which quads contain this triple
                # Precompute: triple -> list of (quad_idx, position_in_quad)
                if not hasattr(construct, '_triple_quad_map'):
                    construct._triple_quad_map = {}
                    for qi, qt in enumerate(quad_triples):
                        for pos, t in enumerate(qt):
                            if t not in construct._triple_quad_map:
                                construct._triple_quad_map[t] = []
                            construct._triple_quad_map[t].append((qi, pos))

                if not hasattr(construct, '_triple_quint_map'):
                    construct._triple_quint_map = {}
                    for qi, qt in enumerate(quint_triples):
                        for pos, t in enumerate(qt):
                            if t not in construct._triple_quint_map:
                                construct._triple_quint_map[t] = []
                            construct._triple_quint_map[t].append((qi, pos))

                # Count change in rk4
                delta_rk4 = 0
                for (qi, pos) in construct._triple_quad_map.get(triple_indices[tidx], []):
                    qt = quad_triples[qi]
                    # Check other 3 triples
                    others_red = True
                    for p in range(4):
                        if p == pos:
                            continue
                        if color_flat[triple_to_idx[qt[p]]] != 0:
                            others_red = False
                            break
                    if old_color == 0 and others_red:
                        # Was all-red, now not
                        delta_rk4 -= 1
                    elif old_color == 1 and others_red:
                        # Was not all-red, now is
                        delta_rk4 += 1

                # Count change in bk5
                delta_bk5 = 0
                for (qi, pos) in construct._triple_quint_map.get(triple_indices[tidx], []):
                    qt = quint_triples[qi]
                    others_blue = True
                    for p in range(10):
                        if p == pos:
                            continue
                        if color_flat[triple_to_idx[qt[p]]] != 1:
                            others_blue = False
                            break
                    if old_color == 1 and others_blue:
                        # Was all-blue, now not
                        delta_bk5 -= 1
                    elif old_color == 0 and others_blue:
                        # Was not all-blue, now is
                        delta_bk5 += 1

                new_rk4 = rk4 + delta_rk4
                new_bk5 = bk5 + delta_bk5

                # Accept if it doesn't increase total violations, or with some probability
                old_total = rk4 + bk5
                new_total = new_rk4 + new_bk5
                if new_total <= old_total or random.random() < 0.1:
                    color_flat[tidx] = new_color
                    rk4 = new_rk4
                    bk5 = new_bk5

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_flat.tolist()
            # Reset cached maps for next n
            if hasattr(construct, '_triple_quad_map'):
                del construct._triple_quad_map
            if hasattr(construct, '_triple_quint_map'):
                del construct._triple_quint_map

    # Final verification
    if best_n > 9:
        # Verify the solution
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]

        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            # Fallback to n=9
            best_n = 9
            best_colors = [0 if (a + b + c) % 2 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
