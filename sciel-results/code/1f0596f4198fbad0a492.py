import time
import random
import math
import sys

# Fixed seed for reproducibility across evaluations
random.seed(42)


def construct():
    start = time.time()
    deadline = start + 112  # leave 8 s safety margin under 120 s

    def cycle_edge_indices(n, edge_map):
        """Return list of lists of edge indices for all C6 cycles in K_n."""
        from itertools import combinations, permutations

        cyc = []
        for verts in combinations(range(n), 6):
            v0 = verts[0]
            others = list(verts[1:])
            # 60 distinct 6-cycles: permutations of others where v1 < v5
            for perm in permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                vs = [v0, v1, v2, v3, v4, v5]
                edges = []
                for k in range(6):
                    a = vs[k]
                    b = vs[(k + 1) % 6]
                    if a > b:
                        a, b = b, a
                    edges.append(edge_map[(a, b)])
                cyc.append(edges)
        return cyc

    def solve_sa(n, tlimit, edge_map, cyc, edge_to_cyc, n_cycles):
        """Simulated annealing for C6-free 4-colouring of K_n. Returns colors or None."""
        ne = n * (n - 1) // 2

        def count_mono(colors):
            # cycle_counts[c][i] = count of colour i in cycle c
            cc = [[0, 0, 0, 0] for _ in range(n_cycles)]
            for ci, ce in enumerate(cyc):
                for e in ce:
                    cc[ci][colors[e]] += 1
            mc = 0
            for c in cc:
                for x in c:
                    if x == 6:
                        mc += 1
                        break
            return mc, cc

        # Try up to 150 restarts
        max_restarts = 150
        for r in range(max_restarts):
            if time.time() > tlimit:
                return None

            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(ne)]
            mono, cc = count_mono(colors)
            if mono == 0:
                return colors

            best_mono = mono
            T = 2.0
            T_min = 0.001
            cooling = 0.9995
            no_improve = 0

            # Greedy descent first
            while mono > 0 and no_improve < 10000 and time.time() < tlimit:
                # Pick an edge in a monochromatic cycle
                # Find a mono cycle
                mono_ci = -1
                for ci in range(n_cycles):
                    c = cc[ci]
                    for x in c:
                        if x == 6:
                            mono_ci = ci
                            break
                    if mono_ci >= 0:
                        break
                if mono_ci < 0:
                    break  # no mono cycle? shouldn't happen if mono>0

                e = random.choice(cyc[mono_ci])
                oc = colors[e]
                best_nc = -1
                best_d = 999
                for nc in range(4):
                    if nc == oc:
                        continue
                    d = 0
                    for ci in edge_to_cyc[e]:
                        c = cc[ci]
                        if c[oc] == 6:
                            d -= 1
                        if c[nc] == 5:
                            d += 1
                    if d < best_d:
                        best_d = d
                        best_nc = nc
                # Always accept if improves; else try SA
                if best_d < 0:
                    colors[e] = best_nc
                    for ci in edge_to_cyc[e]:
                        cc[ci][oc] -= 1
                        cc[ci][best_nc] += 1
                    mono += best_d
                    if mono < best_mono:
                        best_mono = mono
                        no_improve = 0
                    else:
                        no_improve += 1
                elif best_d == 0 and random.random() < 0.3:
                    colors[e] = best_nc
                    for ci in edge_to_cyc[e]:
                        cc[ci][oc] -= 1
                        cc[ci][best_nc] += 1
                    no_improve += 1
                else:
                    no_improve += 1
                    if no_improve > 5000:
                        break

            if mono == 0:
                return colors

            # SA phase
            for step in range(500000):
                if time.time() > tlimit:
                    return None
                if mono == 0:
                    return colors

                e = random.randint(0, ne - 1)
                oc = colors[e]
                nc = random.choice([c for c in range(4) if c != oc])

                d = 0
                for ci in edge_to_cyc[e]:
                    c = cc[ci]
                    if c[oc] == 6:
                        d -= 1
                    if c[nc] == 5:
                        d += 1

                if d < 0 or random.random() < math.exp(-d / T):
                    colors[e] = nc
                    for ci in edge_to_cyc[e]:
                        cc[ci][oc] -= 1
                        cc[ci][nc] += 1
                    mono += d
                    if mono < best_mono:
                        best_mono = mono
                        no_improve = 0
                    else:
                        no_improve += 1
                    T = max(T_min, T * cooling)
                else:
                    no_improve += 1
                    T = max(T_min, T * cooling)

                if no_improve > 30000:
                    break

        return None

    def solve_greedy_then_sa(n, tlimit):
        """Try greedy construction first, then SA if needed."""
        ne = n * (n - 1) // 2
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1

        cyc = cycle_edge_indices(n, edge_map)
        n_cycles = len(cyc)

        edge_to_cyc = [[] for _ in range(ne)]
        for ci, ce in enumerate(cyc):
            for e in ce:
                edge_to_cyc[e].append(ci)

        # Greedy construction: assign edges one by one, picking colour that minimises
        # immediate mono-C6 creation
        colors = [None] * ne
        cc = [[0, 0, 0, 0] for _ in range(n_cycles)]
        mono = 0

        for e in range(ne):
            best_c = 0
            best_d = 999
            for c in range(4):
                d = 0
                for ci in edge_to_cyc[e]:
                    if cc[ci][c] == 5:
                        d += 1
                if d < best_d:
                    best_d = d
                    best_c = c
            colors[e] = best_c
            for ci in edge_to_cyc[e]:
                cc[ci][best_c] += 1
            if any(x == 6 for x in cc) is False:
                pass
            # count mono cycles
            # Actually we need to recount; let's just do a full recount after
            pass

        # Recount
        mono = 0
        for ci in range(n_cycles):
            for x in cc[ci]:
                if x == 6:
                    mono += 1
                    break

        if mono == 0:
            return colors

        # SA phase
        T = 2.0
        T_min = 0.001
        cooling = 0.9995
        best_mono = mono
        no_improve = 0

        while mono > 0 and no_improve < 200000 and time.time() < tlimit:
            e = random.randint(0, ne - 1)
            oc = colors[e]
            nc = random.choice([c for c in range(4) if c != oc])
            d = 0
            for ci in edge_to_cyc[e]:
                if cc[ci][oc] == 6:
                    d -= 1
                if cc[ci][nc] == 5:
                    d += 1
            if d < 0 or random.random() < math.exp(-d / T):
                colors[e] = nc
                for ci in edge_to_cyc[e]:
                    cc[ci][oc] -= 1
                    cc[ci][nc] += 1
                mono += d
                if mono < best_mono:
                    best_mono = mono
                    no_improve = 0
                else:
                    no_improve += 1
                T = max(T_min, T * cooling)
            else:
                no_improve += 1
                T = max(T_min, T * cooling)

        if mono == 0:
            return colors
        return None

    def solve_multi(n, tlimit):
        """Multiple strategies: greedy, random+SA, multiple restarts."""
        ne = n * (n - 1) // 2
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1

        cyc = cycle_edge_indices(n, edge_map)
        n_cycles = len(cyc)

        edge_to_cyc = [[] for _ in range(ne)]
        for ci, ce in enumerate(cyc):
            for e in ce:
                edge_to_cyc[e].append(ci)

        # Strategy 1: Greedy
        res = solve_greedy_then_sa(n, min(time.time() + 20, tlimit))
        if res:
            return res

        # Strategy 2: Multiple random+SA restarts
        res = solve_sa(n, tlimit, edge_map, cyc, edge_to_cyc, n_cycles)
        if res:
            return res

        return None

    # Try n from 19 down
    for n, alloc in [(19, 45), (18, 40), (17, 25), (16, 5), (15, 3)]:
        if time.time() > deadline - 3:
            break
        tlimit = time.time() + alloc
        if tlimit > deadline - 3:
            tlimit = deadline - 3

        res = solve_multi(n, tlimit)
        if res:
            return {'n': n, 'colors': res}

    # Fallback: n=9 with a deterministic valid coloring
    # Use a known structure: round-robin tournament coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            # Colour by distance mod 4, but ensure no C6
            # For K9, use the known 4-colouring
            colors.append(d % 4)

    # Verify quickly
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1

    valid = True
    from itertools import combinations, permutations
    for verts in combinations(range(n), 6):
        v0 = verts[0]
        others = list(verts[1:])
        for perm in permutations(others):
            v1, v2, v3, v4, v5 = perm
            if v1 > v5:
                continue
            vs = [v0, v1, v2, v3, v4, v5]
            c0 = None
            mono = True
            for k in range(6):
                a = vs[k]
                b = vs[(k + 1) % 6]
                if a > b:
                    a, b = b, a
                e = edge_map[(a, b)]
                if c0 is None:
                    c0 = colors[e]
                elif colors[e] != c0:
                    mono = False
                    break
            if mono:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Just return a random n=6 coloring (always valid since no C6 in K6? 
        # Actually K6 has C6. So use n=5.)
        n = 5
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]

    return {'n': n, 'colors': colors}
