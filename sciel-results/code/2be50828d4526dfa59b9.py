"""Rewritten program: SEARCH approach for unique-triple-sum sets."""
import itertools
import random
import time

from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0   # safety margin under the 90s kill

    N = 4096

    # ---------- helper: validate a set ----------
    def is_valid(vals):
        # O(k^3) sum check with early exit
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # ---------- helper: greedy extend from a base set ----------
    def greedy_extend(base, candidates, limit):
        chosen = list(base)
        # build triple sums for current chosen
        def rebuild_sums():
            st = set()
            for x in itertools.combinations_with_replacement(chosen, 3):
                st.add(sum(x))
            return st
        sums = rebuild_sums()
        chosen_set = set(chosen)
        # precompute pair sums for fast incremental test
        def try_add(v):
            # new triple sums involving v: (v,v,v), (v,v,u), (v,u,u), (v,u,w)
            new_sums = set()
            # (v,v,v)
            new_sums.add(3 * v)
            for u in chosen:
                new_sums.add(2 * v + u)      # (v,v,u)
                new_sums.add(v + 2 * u)      # (v,u,u)
            # (v,u,w) with u,w in chosen (any order, allow u=w)
            # iterate pairs (u,w) with u<=w
            cl = list(chosen)
            for ui in range(len(cl)):
                u = cl[ui]
                for wi in range(ui, len(cl)):
                    w = cl[wi]
                    new_sums.add(v + u + w)
            return new_sums
        for v in candidates:
            if time.time() > limit:
                break
            if v in chosen_set:
                continue
            ns = try_add(v)
            if ns & sums:
                continue
            sums |= ns
            chosen.append(v)
            chosen_set.add(v)
        return chosen

    best = []

    # ---------- 1) power-of-2 core (provable) ----------
    pow2 = [2 ** i for i in range(0, 12) if 2 ** i <= 2048]
    if is_valid(pow2):
        best = pow2
    # extend the power-2 core greedily with ascending order
    try:
        ext = greedy_extend(pow2, list(range(1, N + 1)), DEADLINE)
        if is_valid(ext) and len(ext) > len(best):
            best = ext
    except Exception:
        pass
    # Also try extending with descending order
    try:
        ext_desc = greedy_extend(pow2, list(range(N, 0, -1)), DEADLINE)
        if is_valid(ext_desc) and len(ext_desc) > len(best):
            best = ext_desc
    except Exception:
        pass

    # ---------- 2) randomized greedy constructions ----------
    # Strategy: try multiple base spacings, both ascending and shuffled orders
    random.seed(12345)
    best_size = len(best)
    for it in range(80):
        if time.time() > DEADLINE:
            break
        # Vary the base spacing
        base = random.randint(2, 8)
        # Create candidate pool with some density
        density = random.choice([0.1, 0.2, 0.3, 0.5])
        pool = []
        for x in range(1, N + 1):
            if (x - 1) % base == 0 or random.random() < density:
                pool.append(x)
        
        # Alternate between ascending and shuffled order
        if it % 2 == 0:
            # Ascending order (deterministic greedy)
            candidates = sorted(pool)
        else:
            # Shuffled order
            candidates = pool[:]
            random.shuffle(candidates)
        
        # Greedy construction
        chosen = []
        chosen_set = set()
        sums = set()
        for v in candidates:
            if time.time() > DEADLINE:
                break
            if v in chosen_set:
                continue
            ns = set()
            ns.add(3 * v)
            for u in chosen:
                ns.add(2 * v + u)
                ns.add(v + 2 * u)
            cl = list(chosen)
            for ui in range(len(cl)):
                u = cl[ui]
                for wi in range(ui, len(cl)):
                    w = cl[wi]
                    ns.add(v + u + w)
            if ns & sums:
                continue
            sums |= ns
            chosen.append(v)
            chosen_set.add(v)
        
        if len(chosen) > best_size:
            if is_valid(chosen):
                best = chosen
                best_size = len(chosen)

    # ---------- 3) local improvement: multiple passes to add missing numbers ----------
    if time.time() < DEADLINE - 5:
        chosen = list(best)
        chosen_set = set(chosen)
        # Multiple passes with different orderings
        for pass_num in range(3):
            if time.time() > DEADLINE - 3:
                break
            sums = set()
            for x in itertools.combinations_with_replacement(chosen, 3):
                sums.add(sum(x))
            # Different candidate orderings per pass
            if pass_num == 0:
                candidates = list(range(1, N + 1))
            elif pass_num == 1:
                candidates = list(range(N, 0, -1))
            else:
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
            
            added_any = False
            for v in candidates:
                if time.time() > DEADLINE - 3:
                    break
                if v in chosen_set:
                    continue
                ns = set()
                ns.add(3 * v)
                for u in chosen:
                    ns.add(2 * v + u)
                    ns.add(v + 2 * u)
                cl = list(chosen)
                for ui in range(len(cl)):
                    u = cl[ui]
                    for wi in range(ui, len(cl)):
                        w = cl[wi]
                        ns.add(v + u + w)
                if not (ns & sums):
                    sums |= ns
                    chosen.append(v)
                    chosen_set.add(v)
                    added_any = True
            if not added_any:
                break  # No progress, stop
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # ---------- 4) sanity: ensure within range, distinct, positive ----------
    best = [v for v in best if 1 <= v <= N]
    best = sorted(set(best))

    return best
