import math
import time
import random

_cache = None


def _compute(deadline):
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    excluded = set()
    for a in range(1, n + 1):
        if 2 * a in squares:
            excluded.add(a)
    candidates = [a for a in range(1, n + 1) if a not in excluded]
    cand_set = set(candidates)

    # Conflict graph among candidates (edge iff a+b is a square).
    adj = {a: set() for a in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a + b in squares:
                adj[a].add(b)
                adj[b].add(a)

    # Connected components of the conflict graph.
    seen = set()
    comps = []
    for a in candidates:
        if a in seen:
            continue
        comp = set()
        stack = [a]
        seen.add(a)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)

    def color_bound(subset):
        """Greedy coloring count = upper bound on independent-set size."""
        if not subset:
            return 0
        order = sorted(subset, key=lambda u: -len(adj[u] & subset))
        color_of = {}
        maxc = 0
        for u in order:
            used = set()
            for w in adj[u]:
                if w in color_of:
                    used.add(color_of[w])
            c = 0
            while c in used:
                c += 1
            color_of[u] = c
            if c > maxc:
                maxc = c
        return maxc + 1

    def greedy(comp, mode='min'):
        rem = set(comp)
        chosen = set()
        while rem:
            best = None
            bd = None
            for u in rem:
                d = len(adj[u] & rem)
                if mode == 'min':
                    better = (bd is None or d < bd)
                else:
                    better = (bd is None or d > bd)
                if better:
                    bd, best = d, u
                    if mode == 'min' and d == 0:
                        break
            chosen.add(best)
            rem.discard(best)
            rem -= adj[best]
        return chosen

    def solve_component(comp, deadline):
        comp = set(comp)
        best = set()
        for mode in ('min', 'max'):
            g = greedy(comp, mode)
            if len(g) > len(best):
                best = g
        blen = len(best)

        def time_up():
            return time.time() > deadline

        def expand(sub, chosen):
            nonlocal best, blen
            if not sub:
                if len(chosen) > blen:
                    blen, best = len(chosen), set(chosen)
                return
            if time_up():
                return
            cb = color_bound(sub)
            if len(chosen) + cb <= blen:
                return
            v = None
            hv = -1
            for u in sub:
                d = len(adj[u] & sub)
                if d > hv:
                    hv, v = d, u
            # Branch 1: include v
            sub_inc = (sub - {v}) - adj[v]
            expand(sub_inc, chosen | {v})
            # Branch 2: exclude v
            sub_exc = sub - {v}
            expand(sub_exc, chosen)

        expand(comp, set())
        return best

    union = set()
    for c in comps:
        union |= solve_component(c, deadline)

    # Local augmentation: add any non-conflicting candidate.
    changed = True
    while changed:
        changed = False
        for a in candidates:
            if a not in union and not (adj[a] & union):
                union.add(a)
                changed = True

    # Local search: 1-removal + refill to escape plateaus.
    def local_search(s, deadline):
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur = len(s)
            # Try adding any vertex that has no conflicts
            for v in candidates:
                if v not in s and not (adj[v] & s):
                    s.add(v)
                    cur = len(s)
                    improved = True
                    break
            if not improved:
                # Try 1-for-2 swap
                swap_done = False
                for u in list(s):
                    candidates_swap = []
                    for v in candidates:
                        if v in s:
                            continue
                        if u not in adj[v]:
                            continue
                        common = adj[v] & s
                        if len(common) == 1 and u in common:
                            candidates_swap.append(v)
                    if len(candidates_swap) >= 2:
                        found_swap = False
                        for i in range(len(candidates_swap)):
                            if found_swap:
                                break
                            for j in range(i + 1, len(candidates_swap)):
                                if candidates_swap[i] not in adj[candidates_swap[j]]:
                                    s.discard(u)
                                    s.add(candidates_swap[i])
                                    s.add(candidates_swap[j])
                                    cur = len(s)
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    break
                        if swap_done:
                            break
                if not swap_done:
                    # Perturbation: Remove 1-3 random vertices and refill
                    if time.time() > deadline - 2:
                        break
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(s), min(k_remove, len(s)))
                    for v in to_remove:
                        s.discard(v)
                    # Refill greedily
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in candidates:
                            if v not in s and not (adj[v] & s):
                                s.add(v)
                                refill_improved = True
                                break
                    if len(s) > cur:
                        improved = True
                        cur = len(s)
        return s

    union = local_search(union, deadline)

    # Safety verification.
    lst = sorted(union)
    ok = True
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    if not ok:
        rem = set(candidates)
        fb = set()
        while rem:
            best = None
            bd = None
            for u in rem:
                d = len(adj[u] & rem)
                if bd is None or d < bd:
                    bd, best = d, u
            fb.add(best)
            rem.discard(best)
            rem -= adj[best]
        union = fb
    return sorted(union)


def construct():
    global _cache
    if _cache is not None:
        return _cache
    start = time.time()
    deadline = start + 80.0  # safety margin before the 90s wall-clock kill
    random.seed(42)
    _cache = _compute(deadline)
    return _cache
