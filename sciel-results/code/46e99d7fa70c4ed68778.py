"""Combined CSP + fast-delta local search for R(4,5;3) hypergraph coloring."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_indices(n):
        """Build triple index and K4/K5 structures."""
        triples = get_triples(n)
        num_triples = len(triples)
        tri_idx = {t: i for i, t in enumerate(triples)}

        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                tri_idx[(a, b, c)], tri_idx[(a, b, d)],
                tri_idx[(a, c, d)], tri_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(tri_idx[triple])
            k5s.append(tuple(idxs))

        return triples, num_triples, tri_idx, k4s, k5s

    def build_reverse(num_triples, k4s, k5s):
        """Reverse index: triple_idx -> list of clique indices."""
        t2k4 = [[] for _ in range(num_triples)]
        t2k5 = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for t in idxs:
                t2k4[t].append(i)
        for i, idxs in enumerate(k5s):
            for t in idxs:
                t2k5[t].append(i)
        return t2k4, t2k5

    def count_violations(n, colors, tri_idx):
        """Count red K4 and blue K5 violations."""
        red_v = 0
        blue_v = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[tri_idx[(a,b,c)]] == 0 and colors[tri_idx[(a,b,d)]] == 0 and
                colors[tri_idx[(a,c,d)]] == 0 and colors[tri_idx[(b,c,d)]] == 0):
                red_v += 1
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[tri_idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_v += 1
        return red_v, blue_v

    def is_valid(n, colors, tri_idx):
        rv, bv = count_violations(n, colors, tri_idx)
        return rv == 0 and bv == 0

    def incremental_csp_search(n, time_limit, seed=0):
        """
        CSP approach: build coloring vertex by vertex.
        When adding vertex v, color all triples involving v.
        Use forward checking: after coloring a triple, check if any
        K4 becomes all-red or K5 becomes all-blue.
        """
        random.seed(seed)
        np.random.seed(seed)

        triples = get_triples(n)
        num_triples = len(triples)
        tri_idx = {t: i for i, t in enumerate(triples)}

        # Precompute which triples involve each vertex
        vertex_triples = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(triples):
            vertex_triples[a].append(i)
            vertex_triples[b].append(i)
            vertex_triples[c].append(i)

        # Precompute K4s and K5s
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                tri_idx[(a,b,c)], tri_idx[(a,b,d)],
                tri_idx[(a,c,d)], tri_idx[(b,c,d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(tri_idx[triple])
            k5s.append(tuple(idxs))

        t2k4, t2k5 = build_reverse(num_triples, k4s, k5s)

        best_colors = None
        best_cost = float('inf')
        end_time = start_time + time_limit

        # Try multiple random initializations and local search
        for attempt in range(20):
            if time.time() > end_time:
                break

            # Initialize with a heuristic
            if attempt % 3 == 0:
                colors = np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
            elif attempt % 3 == 1:
                colors = np.array([0 if (a * 2 + b * 3 + c * 5) % 7 < 3 else 1 for a, b, c in triples], dtype=np.int8)
            else:
                colors = np.array([np.random.randint(0, 2) for _ in range(num_triples)], dtype=np.int8)

            # Compute initial cost
            red_count = 0
            for idxs in k4s:
                if all(colors[i] == 0 for i in idxs):
                    red_count += 1
            blue_count = 0
            for idxs in k5s:
                if all(colors[i] == 1 for i in idxs):
                    blue_count += 1
            current_cost = red_count + blue_count

            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()

            if current_cost == 0:
                return colors, True

            # Local search with fast delta computation
            temp = 2.0
            cool_rate = 0.9995
            min_temp = 0.02
            no_improve = 0
            max_no_improve = 8000

            while time.time() < end_time:
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color

                # Fast delta computation
                delta = 0

                for k4_idx in t2k4[t]:
                    idxs = k4s[k4_idx]
                    red_before = sum(1 for i in idxs if colors[i] == 0)
                    if old_color == 0:
                        red_after = red_before - 1
                    else:
                        red_after = red_before + 1

                    if red_before == 4 and red_after < 4:
                        delta -= 1
                    elif red_before < 4 and red_after == 4:
                        delta += 1

                for k5_idx in t2k5[t]:
                    idxs = k5s[k5_idx]
                    blue_before = sum(1 for i in idxs if colors[i] == 1)
                    if old_color == 1:
                        blue_after = blue_before - 1
                    else:
                        blue_after = blue_before + 1

                    if blue_before == 10 and blue_after < 10:
                        delta -= 1
                    elif blue_before < 10 and blue_after == 10:
                        delta += 1

                new_cost = current_cost + delta

                if new_cost < current_cost:
                    accept = True
                    no_improve = 0
                elif new_cost == current_cost:
                    accept = np.random.rand() < 0.2
                    no_improve += 1
                else:
                    if temp > min_temp:
                        accept = np.random.rand() < np.exp(-new_cost / max(temp, 0.01))
                    else:
                        accept = False
                    no_improve += 1

                if accept:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        no_improve = 0
                        if best_cost == 0:
                            return best_colors, True

                temp *= cool_rate

                if no_improve > max_no_improve and temp > min_temp:
                    temp = max(0.5, temp * 1.5)
                    no_improve = 0

        return best_colors, (best_cost == 0)

    def try_algebraic(n):
        """Try algebraic coloring with primes."""
        triples = get_triples(n)
        num_triples = len(triples)
        tri_idx = {t: i for i, t in enumerate(triples)}

        candidates = []
        for p in range(max(n, 11), min(n + 40, 500)):
            if p < 2:
                continue
            is_prime = True
            for d in range(2, int(p**0.5) + 1):
                if p % d == 0:
                    is_prime = False
                    break
            if is_prime:
                candidates.append(p)

        best_colors = None
        best_score = float('inf')

        for p in candidates:
            if time.time() > deadline - 5:
                break

            qr = set()
            for x in range(p):
                qr.add((x * x) % p)
            qr.discard(0)

            colors = []
            for (a, b, c) in triples:
                prod = ((a - b) * (b - c) * (c - a)) % p
                if prod == 0:
                    colors.append(1)
                elif prod in qr:
                    colors.append(0)
                else:
                    colors.append(1)

            rv, bv = count_violations(n, colors, tri_idx)
            score = rv * 100 + bv

            if score < best_score:
                best_score = score
                best_colors = colors

            if score == 0:
                return colors

        return best_colors

    # Main search strategy
    best_n = 0
    best_colors = None

    # Start with known small valid construction
    n_start = 9
    triples_9 = get_triples(n_start)
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]
    best_n = n_start

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        # Try algebraic first (fast)
        colors_alg = try_algebraic(target_n)
        if colors_alg is not None:
            tri_idx = {t: i for i, t in enumerate(get_triples(target_n))}
            if is_valid(target_n, colors_alg, tri_idx):
                best_n = target_n
                best_colors = colors_alg
                continue

        # Try CSP + local search
        time_budget = min(8.0, (deadline - time.time()) * 0.4)
        if time_budget < 1.0:
            break

        colors, valid = incremental_csp_search(target_n, time_budget, seed=target_n)
        if valid:
            best_n = target_n
            best_colors = colors
            continue
        else:
            # Save best partial if no valid found
            if best_n == 0:
                best_n = target_n
                best_colors = colors
            break

    # Final verification
    if best_n > 0:
        triples_final = get_triples(best_n)
        if len(best_colors) != len(triples_final):
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_final]
        tri_idx_final = {t: i for i, t in enumerate(triples_final)}
        if not is_valid(best_n, best_colors, tri_idx_final):
            # Fallback to known valid
            best_n = 9
            triples_9 = get_triples(9)
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_9]

    return {'n': best_n, 'colors': list(best_colors)}
# EVOLVE-BLOCK-END
