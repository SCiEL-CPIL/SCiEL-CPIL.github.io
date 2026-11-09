"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    N = 256
    vertices = list(range(1, N + 1))
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(2, N + 1) if x % 2 == 0]

    # ---- Precompute conflict structure (edge (a,b) if (a*b) % (a+b) == 0) ----
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i].add(j)
                conflict[j].add(i)

    # For each even e, the list of odds it conflicts with (used to decide swaps)
    even_odd_conf = {e: [o for o in odds if o in conflict[e]] for e in evens}

    # ---- Fast feasibility helpers ----
    def can_add(v, chosen):
        return not (conflict[v] & chosen)

    def complete(chosen):
        """Greedy fill: add every still-feasible vertex (order-independent result)."""
        for v in vertices:
            if v not in chosen and can_add(v, chosen):
                chosen.add(v)
        return chosen

    # ---- Core local search: 1-for-1 and 2-for-1 swaps + completion ----
    def local_search(initial, max_iters=4000):
        current = set(initial)
        complete(current)
        best = set(current)
        best_size = len(current)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False

            # 1-for-1: remove r, add a single vertex that becomes feasible
            for r in list(current):
                if time.time() > deadline:
                    break
                current.discard(r)
                for v in vertices:
                    if v in current:
                        continue
                    if can_add(v, current):
                        current.add(v)
                        improved = True
                        break
                if not improved:
                    current.add(r)  # restore
                if improved:
                    break

            # 2-for-1: remove two, add one vertex that is now feasible
            if not improved:
                rem_list = list(current)
                for i in range(len(rem_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(rem_list)):
                        r1, r2 = rem_list[i], rem_list[j]
                        current.discard(r1)
                        current.discard(r2)
                        for v in vertices:
                            if v in current:
                                continue
                            if can_add(v, current):
                                current.add(v)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    # restore removed vertices (keep only those that fit)
                    for r in rem_list:
                        if r not in current and can_add(r, current):
                            current.add(r)

            # 1-for-2: remove one, add two vertices that become feasible
            if not improved:
                for r in list(current):
                    if time.time() > deadline:
                        break
                    current.discard(r)
                    added = []
                    for v in vertices:
                        if v in current:
                            continue
                        if can_add(v, current):
                            current.add(v)
                            added.append(v)
                            if len(added) >= 2:
                                break
                    if len(added) >= 2:
                        improved = True
                    else:
                        for v in added:
                            current.discard(v)
                        current.add(r)  # restore
                    if improved:
                        break

            # completion pass to exploit any newly opened feasibility
            if improved:
                complete(current)

            if len(current) > best_size:
                best = set(current)
                best_size = len(current)

            if not improved:
                break
        return best

    def greedy_order(order):
        chosen = set()
        for v in order:
            if can_add(v, chosen):
                chosen.add(v)
        return chosen

    def shuffle_inplace(order):
        for i in range(len(order) - 1, 0, -1):
            j = random.randint(0, i)
            order[i], order[j] = order[j], order[i]

    best_overall = set()
    best_size = 0

    def update_best(cand):
        nonlocal best_overall, best_size
        if len(cand) > best_size:
            best_overall = set(cand)
            best_size = len(cand)

    # ---- Baseline: all odds (guaranteed valid) ----
    baseline = set(odds)
    complete(baseline)
    update_best(baseline)

    # ---- Randomized greedy restarts with diverse orderings ----
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        random.seed(restart * 7919 + 13)

        if restart % 4 == 0:
            order = vertices[:]
            shuffle_inplace(order)
        elif restart % 4 == 1:
            # fewest conflicts first (least constrained), perturbed
            order = sorted(vertices, key=lambda v: len(conflict[v]))
            shuffle_inplace(order)
        elif restart % 4 == 2:
            # most constrained first, perturbed
            order = sorted(vertices, key=lambda v: -len(conflict[v]))
            shuffle_inplace(order)
        else:
            # odds first (baseline), then evens in random order
            order = odds + evens[:]
            shuffle_inplace(order[len(odds):])

        greedy_set = greedy_order(order)
        update_best(greedy_set)

        # local search from this starting point
        if time.time() < deadline - 2.0:
            ls_result = local_search(greedy_set, max_iters=3000)
            update_best(ls_result)

    # ---- Local search from the odd baseline ----
    if time.time() < deadline - 4.0:
        update_best(local_search(set(odds), max_iters=5000))

    # ---- Perturbation restarts from the current best (escape local optima) ----
    if time.time() < deadline - 4.0:
        for _ in range(8):
            if time.time() > deadline - 2.0:
                break
            trial = set(best_overall)
            k = random.randint(2, 5)
            to_remove = random.sample(list(trial), min(k, len(trial)))
            for t in to_remove:
                trial.discard(t)
            update_best(local_search(trial, max_iters=2500))

    # ---- Final high-precision polish pass on the best set ----
    if time.time() < deadline - 1.0:
        update_best(local_search(best_overall, max_iters=6000))

    # ---- Final verification + fallback ----
    result = sorted(best_overall)
    if not result:
        result = list(range(1, N + 1, 2))

    s = set(result)
    valid = True
    for a in s:
        for b in s:
            if a != b and b in conflict[a]:
                valid = False
                break
        if not valid:
            break

    if not valid:
        result = list(range(1, N + 1, 2))

    return result
# EVOLVE-BLOCK-END
