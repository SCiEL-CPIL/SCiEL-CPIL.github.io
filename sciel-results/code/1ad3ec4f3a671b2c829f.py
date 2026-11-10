"""Search for a max-cardinality subset of [1,100] with no subset summing to 200."""
import time

TARGET = 200

def reachable_sums(selected):
    reach = {0}
    for x in selected:
        new = {s + x for s in reach if s + x <= TARGET}
        reach |= new
        if TARGET in reach:
            break
    return reach

def is_valid(selected):
    return TARGET not in reachable_sums(selected)

def find_forbidden_subset(selected):
    """Return a subset (list) of selected summing to TARGET, or None."""
    parent = {0: (None, None)}
    reach = {0}
    for e in selected:
        for s in list(reach):
            ns = s + e
            if ns == TARGET:
                subset = []
                cur = ns
                while cur != 0:
                    el, prev = parent[cur]
                    subset.append(el)
                    cur = prev
                return subset
            if ns < TARGET and ns not in parent:
                parent[ns] = (e, s)
                reach.add(ns)
    return None

def construct():
    deadline = time.time() + 80.0

    # Valid 35-element seed: [66,100]. 2-sums <=199, 3-sums >=201.
    best = list(range(66, 101))
    cur = set(best)

    def attempt(adds, max_removals):
        """Try adding `adds` (set) to cur, removing <= max_removals to stay valid.
        Return new set if strictly larger and valid, else None."""
        adds = set(adds)
        if adds & cur:
            return None
        work = set(cur)
        for _ in range(max_removals + 1):
            cand = work | adds
            if is_valid(cand):
                if len(cand) > len(cur):
                    return cand
                return None
            sub = find_forbidden_subset(cand)
            if sub is None:
                return None
            # remove a culprit from `work` (not from adds)
            victim = None
            for v in sorted(sub, reverse=True):
                if v in work:
                    victim = v
                    break
            if victim is None:
                return None
            work.discard(victim)
        return None

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # single-add improvements
        for x in range(1, 66):
            if time.time() > deadline:
                break
            if x in cur:
                continue
            res = attempt({x}, max_removals=2)
            if res is not None:
                cur = res
                improved = True
        # double-add / single-remove improvements (net +1)
        if time.time() < deadline:
            for x1 in range(1, 66):
                if time.time() > deadline:
                    break
                if x1 in cur:
                    continue
                for x2 in range(x1 + 1, 66):
                    if time.time() > deadline:
                        break
                    if x2 in cur:
                        continue
                    res = attempt({x1, x2}, max_removals=1)
                    if res is not None:
                        cur = res
                        improved = True

    best = list(cur)
    if not best:
        best = list(range(66, 101))
    return best
