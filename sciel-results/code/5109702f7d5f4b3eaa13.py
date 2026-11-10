"""Constraint-based backtracking solver for the no-3-sum-divisor task."""
import time
import random

_N = 192
_MAX = 48


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under 90s kill

    # universe = values 1..192
    universe = list(range(1, _N + 1))
    n = len(universe)

    # Precompute all triples of DISTINCT values (indices) and their sums
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    T = len(triples)

    # sum of each triple
    tsum = [universe[i] + universe[j] + universe[k] for (i, j, k) in triples]

    # For each triple, which universe values divide its sum?
    # divisors_of[t] = list of values v such that tsum[t] % v == 0
    divisors_of = []
    for s in tsum:
        ds = [v for v in universe if s % v == 0]
        divisors_of.append(ds)

    # For each universe value v, which triples does v divide?
    # div_index[v] = list of triple indices t where tsum[t] % v == 0
    div_index = [[] for _ in range(n)]
    for t, ds in enumerate(divisors_of):
        for v in ds:
            div_index[universe.index(v)].append(t)

    # For each universe value v, the triples that contain v (by index)
    contains = [[] for _ in range(n)]
    for t, (i, j, k) in enumerate(triples):
        contains[i].append(t)
        contains[j].append(t)
        contains[k].append(t)

    # A triple is "active" if all three of its values are selected.
    # A conflict exists iff there is an active triple t with some selected v
    # (v in the triple) that divides tsum[t].
    #
    # We maintain:
    #   sel: set of selected indices
    #   active_count[v] for each v: number of active triples (all 3 in sel)
    #     that v divides. If >0, v is in conflict.
    # Actually simpler: a triple t is active when its 3 members are all selected.
    # Conflict iff exists active t and exists v in t with tsum[t] % v == 0.
    #
    # Maintain per-triple active flag via counting selected members.
    member_count = [0] * T  # how many of the triple's 3 are selected
    active = [False] * T
    conflict = [False] * n  # per value, whether it currently divides an active triple

    # But recomputing conflict per node is expensive. Instead:
    # For a candidate addition of index a:
    #   new active triples = triples t in contains[a] where the other two are already selected.
    #   For each such t, if a divides tsum[t] -> conflict (a is bad).
    #   Also, for each such t, for each of the other two members b in t,
    #     if b divides tsum[t] -> b becomes conflicted (but b is already selected;
    #     that means the set was ALREADY invalid before adding a? No: t was not active
    #     before because a was missing, so t only becomes active now. So b's conflict
    #     is NEW and caused by a.)
    # So adding a is valid iff:
    #   for every t in contains[a] with the other two already selected:
    #       no member of t divides tsum[t].
    # i.e. a is valid iff for all such newly-active t, none of its 3 members divides tsum[t].
    #
    # This is a clean local check. Precompute for each triple t the list of its
    # members that divide its sum: bad_members[t] = [v in t : tsum[t] % v == 0]
    # Then a newly-active triple t is "bad" iff bad_members[t] is non-empty.
    bad_members = []
    for t, (i, j, k) in enumerate(triples):
        s = tsum[t]
        bm = []
        if s % universe[i] == 0:
            bm.append(i)
        if s % universe[j] == 0:
            bm.append(j)
        if s % universe[k] == 0:
            bm.append(k)
        bad_members.append(bm)

    selected = [False] * n
    best = []  # best set (list of values) found
    best_size = 0

    def is_bad_triple(t):
        return len(bad_members[t]) > 0

    def can_add(a, sel_count, depth):
        # check triples containing a whose other two are selected
        for t in contains[a]:
            i, j, k = triples[t]
            # the other two besides a
            others = [x for x in (i, j, k) if x != a]
            b, c = others
            if selected[b] and selected[c]:
                if is_bad_triple(t):
                    return False
        return True

    def try_add(a):
        selected[a] = True
        # record which triples became active for rollback
        activated = []
        for t in contains[a]:
            i, j, k = triples[t]
            others = [x for x in (i, j, k) if x != a]
            b, c = others
            if selected[b] and selected[c]:
                activated.append(t)
        return activated

    def rollback(a, activated):
        selected[a] = False

    def backtrack(start, cur_size, cur_list):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if cur_size > best_size:
            best = list(cur_list)
            best_size = cur_size
        if best_size >= _MAX:
            return
        # bound: number of values from start..n-1 still available
        remaining = n - start
        if cur_size + remaining <= best_size:
            return
        # iterate candidates from large to small (prefer big values)
        for idx in range(start, n):
            a = n - 1 - idx  # descending value order
            if a < start:
                break
            if not selected[a]:
                if can_add(a, cur_size, idx):
                    activated = try_add(a)
                    cur_list.append(universe[a])
                    backtrack(idx + 1, cur_size + 1, cur_list)
                    cur_list.pop()
                    rollback(a, activated)
                    if time.time() > deadline:
                        return

    # initial: try to build greedily large-to-small as a quick seed
    cur_list = []
    backtrack(0, 0, cur_list)

    result = best if best else []
    # ensure distinct positive ints <=192, sorted
    result = sorted(set(result))
    return result
