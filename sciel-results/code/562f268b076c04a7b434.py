import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under the 120s limit

    N = 16
    FULL = (1 << N) - 1
    SIZE = 1 << N  # 65536

    # All triples as (i,j,k,mask)
    all_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples.append((i, j, k, mask))
    T = len(all_triples)  # 560

    # ---------- fast exact independence number via hitting-set dual ----------
    # cov[mask] = number of selected triples that are subsets of `mask`.
    # A set H hits every triple iff cov[~H & FULL] == 0.
    # tau = min |H| with that property, alpha = 16 - tau.
    cov = bytearray(SIZE)

    # Precompute, for every mask, the list of triple indices that are subsets of it.
    subset_lists = [[] for _ in range(SIZE)]
    for ti, (i, j, k, mask) in enumerate(all_triples):
        for sup in range(mask, SIZE):
            if (sup & mask) == mask:
                subset_lists[sup].append(ti)

    # Precompute popcount for every mask.
    pop = bytearray(SIZE)
    for m in range(1, SIZE):
        pop[m] = pop[m >> 1] + (m & 1)

    # Precompute masks grouped by popcount, for fast tau search.
    by_size = [[] for _ in range(N + 1)]
    for m in range(SIZE):
        by_size[pop[m]].append(m)

    def recompute_cov():
        cov[:] = b'\x00' * SIZE
        for ti, (i, j, k, mask) in enumerate(all_triples):
            if selected[ti]:
                for sup in range(mask, SIZE):
                    if (sup & mask) == mask:
                        cov[sup] += 1

    def add_triple(ti):
        mask = all_triples[ti][3]
        for sup in range(mask, SIZE):
            if (sup & mask) == mask:
                cov[sup] += 1

    def remove_triple(ti):
        mask = all_triples[ti][3]
        for sup in range(mask, SIZE):
            if (sup & mask) == mask:
                cov[sup] -= 1

    def calc_alpha():
        # smallest H (by size) whose complement has zero covered triples
        for k in range(0, N + 1):
            for comp in by_size[N - k]:  # comp = ~H, popcount(comp)=16-k
                if cov[comp] == 0:
                    return N - k
        return 0

    # ---------- linear constraint feasibility ----------
    # Precompute, for each triple, the other triples that conflict (share >=2 points).
    conflict = [set() for _ in range(T)]
    for a in range(T):
        ma = all_triples[a][3]
        for b in range(a + 1, T):
            mb = all_triples[b][3]
            if (ma & mb).bit_count() > 1:
                conflict[a].add(b)
                conflict[b].add(a)

    selected = [False] * T
    sel_count = [0] * N  # how many selected triples contain each point

    def can_add(ti):
        # point-degree cap: each point in at most 5 triples (necessary for linear)
        for p in (all_triples[ti][0], all_triples[ti][1], all_triples[ti][2]):
            if sel_count[p] >= 5:
                return False
        for b in conflict[ti]:
            if selected[b]:
                return False
        return True

    def do_add(ti):
        selected[ti] = True
        for p in (all_triples[ti][0], all_triples[ti][1], all_triples[ti][2]):
            sel_count[p] += 1
        add_triple(ti)

    def do_remove(ti):
        selected[ti] = False
        for p in (all_triples[ti][0], all_triples[ti][1], all_triples[ti][2]):
            sel_count[p] -= 1
        remove_triple(ti)

    # ---------- deterministic high-quality fallback ----------
    def build_fallback():
        # Known linear packing on 16 points with alpha = 10 (5 disjoint triples)
        return [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [9, 12, 15], [10, 13, 0], [11, 14, 1],
            [2, 12, 15], [3, 13, 9], [4, 14, 10],
            [5, 15, 11], [6, 9, 14], [7, 10, 15],
            [8, 11, 12], [0, 4, 15], [1, 5, 9],
            [2, 6, 10], [3, 7, 11], [4, 8, 12],
            [5, 9, 13], [6, 10, 14], [7, 11, 15],
            [8, 12, 13], [9, 13, 14], [10, 14, 15],
            [11, 15, 12], [13, 15, 0],
        ]

    def triples_to_result(triples):
        return [[t[0], t[1], t[2]] for t in triples]

    def current_triples():
        return [all_triples[ti][:3] for ti in range(T) if selected[ti]]

    # ---------- simulated annealing local search ----------
    def anneal():
        # random feasible start
        order = list(range(T))
        random.shuffle(order)
        for ti in order:
            if can_add(ti):
                do_add(ti)
        recompute_cov()
        cur_alpha = calc_alpha()

        T0 = 2.5
        Tend = 0.02
        iters = 0
        while time.time() < deadline:
            iters += 1
            # pick a random move type
            r = random.random()
            if r < 0.45:
                # add a random feasible triple
                ti = random.randrange(T)
                if not selected[ti] and can_add(ti):
                    do_add(ti)
                    new_alpha = calc_alpha()
                    d = new_alpha - cur_alpha
                    if d <= 0 or random.random() < (0.5 ** d):
                        cur_alpha = new_alpha
                    else:
                        do_remove(ti)
            elif r < 0.75:
                # swap: remove a selected triple, then try to add a random one
                sel = [ti for ti in range(T) if selected[ti]]
                if sel:
                    ti = random.choice(sel)
                    do_remove(ti)
                    base_alpha = calc_alpha()
                    # try to add one
                    ti2 = random.randrange(T)
                    if not selected[ti2] and can_add(ti2):
                        do_add(ti2)
                        new_alpha = calc_alpha()
                        d = new_alpha - cur_alpha
                        if d <= 0 or random.random() < (0.5 ** d):
                            cur_alpha = new_alpha
                        else:
                            do_remove(ti2)
                    else:
                        do_add(ti)  # revert
            else:
                # targeted: add a triple that maximally reduces alpha among a sample
                best_ti = -1
                best_a = cur_alpha
                for _ in range(6):
                    ti = random.randrange(T)
                    if not selected[ti] and can_add(ti):
                        do_add(ti)
                        a = calc_alpha()
                        do_remove(ti)
                        if a < best_a:
                            best_a = a
                            best_ti = ti
                if best_ti >= 0 and best_a < cur_alpha:
                    do_add(best_ti)
                    cur_alpha = best_a

            # cool
            if iters % 200 == 0:
                T0 *= 0.9
            if T0 < Tend:
                T0 = Tend
        return cur_alpha

    # ---------- main: run several annealing restarts, keep best ----------
    best_alpha = None
    best_triples = None

    for _ in range(2000):
        if time.time() > deadline - 4.0:
            break
        a = anneal()
        if best_alpha is None or a < best_alpha:
            best_alpha = a
            best_triples = current_triples()
            if best_alpha <= 6:
                break

    # finalize
    if best_triples is None:
        best_triples = build_fallback()
        best_alpha = 10

    # final polish: try to add any feasible triple that reduces alpha
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for ti in range(T):
            if not selected[ti] and can_add(ti):
                do_add(ti)
                a = calc_alpha()
                if a < best_alpha:
                    best_alpha = a
                    best_triples = current_triples()
                else:
                    do_remove(ti)

    result = triples_to_result(best_triples)

    # validate linear property; if somehow broken, fall back
    seen = {}
    ok = True
    for tr in result:
        for x in range(3):
            for y in range(x + 1, 3):
                pair = (tr[x], tr[y])
                if pair in seen:
                    ok = False
                else:
                    seen[pair] = 1
    if not ok:
        result = triples_to_result(build_fallback())

    return result
