"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

    N = 256
    size = N - 1  # numbers 2..256, index i = number i+2

    # ---------- Compatibility / conflict precomputation ----------
    # conflict[i][j] = True if lcm(i+2, j+2) <= 256 (cannot coexist)
    conflict = [[False] * size for _ in range(size)]
    for i in range(size):
        a = i + 2
        for j in range(i + 1, size):
            b = j + 2
            g = gcd(a, b)
            if a * b // g <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    # weights as Fractions
    weights = [Fraction(1, i + 2) for i in range(size)]
    # float weights for fast bound arithmetic
    wfloat = [1.0 / (i + 2) for i in range(size)]

    # conflict sets
    conflict_sets = [set() for _ in range(size)]
    for i in range(size):
        cs = conflict_sets[i]
        row = conflict[i]
        for j in range(size):
            if row[j]:
                cs.add(j)

    # ---------- Prime-power slot decomposition for tight upper bound ----------
    # For each number, compute its "prime-power signature": the set of primes
    # p for which p^v_p(x) is a "dominant" factor (>= 2). Two numbers conflict
    # on prime p if including both would make p-part of lcm exceed 256 budget.
    # We use a simpler, valid upper bound: for each prime p, partition numbers by
    # v_p. A clique can contain at most one number from the "top" p-adic tier
    # that alone forces lcm>256 with any other top-tier member.
    # We build a greedy-coloring bound on the *compatibility* graph instead,
    # which is the proven strong bound (clique weight <= sum of color-class max).

    def greedy_color_bound(candidates, selected_fixed):
        """
        Upper bound on additional weight we can add from `candidates`
        (indices not conflicting with each other and with selected_fixed).
        Uses greedy coloring of the compatibility graph restricted to candidates:
        each color class is a clique in the compatibility graph, so at most one
        per color class can be chosen. Bound = sum over color classes of the max
        weight in that class.
        """
        if not candidates:
            return 0.0
        cand = list(candidates)
        # order by descending weight for tighter coloring
        cand.sort(key=lambda i: -wfloat[i])
        color_class_max = []
        color_of = {}
        for idx in cand:
            # find a color where idx is compatible with all already in that color
            placed = False
            for c in range(len(color_class_max)):
                ok = True
                # check compatibility with members of color c
                # we track members per color
                for m in color_members[c]:
                    if conflict[idx][m]:
                        ok = False
                        break
                if ok:
                    color_of[idx] = c
                    color_members[c].append(idx)
                    if wfloat[idx] > color_class_max[c]:
                        color_class_max[c] = wfloat[idx]
                    placed = True
                    break
            if not placed:
                color_of[idx] = len(color_class_max)
                color_members.append([idx])
                color_class_max.append(wfloat[idx])
        return sum(color_class_max)

    # We need color_members as a closure var; rebuild each call.
    def color_bound(candidates):
        cand = list(candidates)
        cand.sort(key=lambda i: -wfloat[i])
        color_members = []
        color_class_max = []
        for idx in cand:
            placed = False
            for c in range(len(color_members)):
                ok = True
                for m in color_members[c]:
                    if conflict[idx][m]:
                        ok = False
                        break
                if ok:
                    color_members[c].append(idx)
                    if wfloat[idx] > color_class_max[c]:
                        color_class_max[c] = wfloat[idx]
                    placed = True
                    break
            if not placed:
                color_members.append([idx])
                color_class_max.append(wfloat[idx])
        return sum(color_class_max)

    # ---------- Branch and bound over the compatibility graph (clique) ----------
    # We want a clique in the compatibility graph (max reciprocal sum).
    # Use B&B: at each node, bound = current_weight + color_bound(available).

    best_set = []
    best_score = Fraction(0)
    best_score_f = 0.0

    def branch_and_bound(available, current_weight_f, current_set, deadline):
        """available: list/set of candidate indices (all mutually compatible not yet checked)."""
        nonlocal best_set, best_score, best_score_f
        if time.time() > deadline:
            return
        if not available:
            if current_weight_f > best_score_f:
                best_score_f = current_weight_f
                best_set = list(current_set)
                best_score = sum(weights[i] for i in current_set)
            return
        # bound
        bound = current_weight_f + color_bound(available)
        if bound <= best_score_f + 1e-12:
            return
        # pick a pivot (highest weight)
        avail_list = list(available)
        avail_list.sort(key=lambda i: -wfloat[i])
        pivot = avail_list[0]
        # branch: include pivot
        new_set = current_set + [pivot]
        new_avail = [x for x in avail_list[1:] if not conflict[pivot][x]]
        branch_and_bound(new_avail, current_weight_f + wfloat[pivot], new_set, deadline)
        # branch: exclude pivot
        branch_and_bound(avail_list[1:], current_weight_f, current_set, deadline)

    # Run B&B from the full set, but seed with a good initial solution first.

    # ---------- Initial solution via multi-start greedy + local search ----------
    def can_add(selected_set, idx):
        return not (conflict_sets[idx] & selected_set)

    def greedy(order):
        selected = set()
        for idx in order:
            if can_add(selected, idx):
                selected.add(idx)
        return selected

    def score_of(selected_indices):
        s = Fraction(0)
        for idx in selected_indices:
            s += weights[idx]
        return s

    def local_search(selected_set, max_iters=30000, deadline=deadline):
        selected = set(selected_set)
        current_score = score_of(selected)
        best_score_ls = current_score
        best_set_ls = set(selected)
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # 1-swap
            for rem in list(selected):
                rem_weight = weights[rem]
                others = selected - {rem}
                for add in range(size):
                    if add in selected:
                        continue
                    if conflict_sets[add] & others:
                        continue
                    new_score = current_score - rem_weight + weights[add]
                    if new_score > current_score:
                        current_score = new_score
                        selected = others | {add}
                        if current_score > best_score_ls:
                            best_score_ls = current_score
                            best_set_ls = set(selected)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # 2-swap
                sel_list = list(selected)
                n_sel = len(sel_list)
                found = False
                for i in range(n_sel):
                    if time.time() > deadline:
                        break
                    rem1 = sel_list[i]
                    for j in range(i + 1, n_sel):
                        rem2 = sel_list[j]
                        base = selected - {rem1, rem2}
                        candidates = []
                        for add in range(size):
                            if add in selected:
                                continue
                            if conflict_sets[add] & base:
                                continue
                            candidates.append(add)
                        for ci in range(len(candidates)):
                            if time.time() > deadline:
                                break
                            add1 = candidates[ci]
                            for cj in range(ci + 1, len(candidates)):
                                add2 = candidates[cj]
                                if conflict[add1][add2]:
                                    continue
                                new_score = (current_score
                                              - weights[rem1] - weights[rem2]
                                              + weights[add1] + weights[add2])
                                if new_score > current_score:
                                    current_score = new_score
                                    selected = base | {add1, add2}
                                    if current_score > best_score_ls:
                                        best_score_ls = current_score
                                        best_set_ls = set(selected)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            if not improved:
                break
        return best_set_ls

    # Diverse orders
    conf_counts = [len(conflict_sets[i]) for i in range(size)]
    orders = [
        list(range(size)),
        list(range(size - 1, -1, -1)),
        sorted(range(size), key=lambda i: -weights[i]),
        sorted(range(size), key=lambda i: conf_counts[i]),
        sorted(range(size), key=lambda i: -weights[i] / (conf_counts[i] + 1)),
        sorted(range(size), key=lambda i: (-weights[i], conf_counts[i])),
        sorted(range(size), key=lambda i: (conf_counts[i], -weights[i])),
    ]
    import random
    for seed in (42, 123, 999, 7, 2024):
        rng = random.Random(seed)
        o = list(range(size))
        rng.shuffle(o)
        orders.append(o)

    # Get best initial
    init_best = set()
    init_best_score = Fraction(0)
    for order in orders:
        if time.time() > deadline:
            break
        gs = greedy(order)
        if not gs:
            continue
        ls = local_search(gs, max_iters=8000, deadline=deadline)
        sc = score_of(ls)
        if sc > init_best_score:
            init_best_score = sc
            init_best = ls

    if init_best:
        best_set = list(init_best)
        best_score = init_best_score
        best_score_f = sum(wfloat[i] for i in init_best)

    # ---------- Now run B&B to try to beat the initial ----------
    if time.time() < deadline - 5:
        # Run B&B on the full candidate set; it will improve best_* if it finds better.
        # To make B&B tractable, order the search by weight and rely on the
        # color bound for pruning. Start from full set.
        full_avail = list(range(size))
        branch_and_bound(full_avail, 0.0, [], deadline - 2)

    # ---------- Final local polish on the B&B result ----------
    if best_set and time.time() < deadline - 1:
        polished = local_search(best_set, max_iters=10000, deadline=deadline - 0.5)
        sc = score_of(polished)
        if sc > best_score:
            best_score = sc
            best_set = list(polished)

    # ---------- Convert to values ----------
    result = [i + 2 for i in best_set]
    if not result:
        result = [256]

    # ---------- Validate ----------
    def is_valid(vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                if a * b // gcd(a, b) <= 256:
                    return False
        return True

    if not is_valid(result):
        # Fallback to a known-valid singleton
        result = [256]

    return sorted(result)
# EVOLVE-BLOCK-END
