"""Shortest superpermutation via maximal-overlap path construction with heuristic search."""
import sys
import time
import random
from itertools import permutations
from collections import defaultdict

sys.setrecursionlimit(1000000)

# ---- Exact memoized solver for small n ----
def _solve_exact(n):
    if n == 1:
        return [1]
    all_perms = list(permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    used = set()

    # precompute: for each symbol x, which permutations end in x
    perms_end = [[] for _ in range(n + 1)]
    for p in all_perms:
        perms_end[p[-1]].append(p)

    # for each prefix (last n-1 symbols), which symbol(s) extend it to a permutation
    # ext[prefix] = list of symbols s such that prefix + (s,) in perm_set
    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, []).append(p[-1])
    for k in ext:
        ext[k] = sorted(ext[k])

    memo = {}

    def dfs(prefix, k):
        # prefix: tuple of last n-1 symbols (or shorter); k: count appended so far
        # returns best (k, suffix-list) for completing from here
        state = (prefix, frozenset(used))
        if state in memo:
            return memo[state]

        # we've placed k symbols; need all n! permutations covered
        # A permutation is "covered" once it has appeared as a window.
        # We track used permutations.
        if len(used) == len(all_perms):
            memo[state] = (k, [])
            return (k, [])
        last = prefix[-1] if prefix else None
        # candidate symbols: smallest first, skip == last
        cands = [s for s in range(1, n + 1) if s != last]
        cands.sort()
        best = None
        for s in cands:
            new_pref = (prefix + (s,))[-(n - 1):]
            # which permutations does appending s complete?
            # any permutation p with p[:-1] == new_pref and p[-1] == s, i.e. p == new_pref + (s,)
            completed = new_pref + (s,)
            if completed not in perm_set:
                # appending s forms no new permutation window that's valid;
                # but window may still be non-permutation; only valid permutations count.
                # If completed not in perm_set, no permutation is newly covered -> skip
                continue
            if completed in used:
                # already covered; appending s gives no new permutation
                continue
            used.add(completed)
            sub = dfs(new_pref, k + 1)
            used.discard(completed)
            if sub is None:
                continue
            kk, tail = sub
            cur = (kk + 1, [s] + tail)
            if best is None or cur < best:
                best = cur
        memo[state] = best
        return best

    # start: seed with first permutation
    # choose starting permutation = (1,2,...,n); mark it used, prefix = last n-1
    first = tuple(range(1, n + 1))
    used.add(first)
    prefix = first[1:]
    res = dfs(prefix, n)  # we've already placed n symbols for the first permutation
    if res is None:
        return None
    k, tail = res
    return list(first) + tail


def construct(n):
    """Return a string over '1'..'n' that contains every permutation of 1..n as a
    consecutive window. Minimises length; falls back to the classic recursive
    construction if the fast path fails or runs out of time."""
    deadline = time.time() + 110.0  # safety margin under the 120s wall-clock kill

    def _classic(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 1:
        return "1"

    # For n <= 5, try the exact solver within the budget.
    if n <= 5:
        try:
            t0 = time.time()
            res = _solve_exact(n)
            if res is not None:
                # verify
                if time.time() - t0 < 30:
                    return "".join(map(str, res))
        except Exception:
            pass

    # Fast greedy maximal-overlap path with multi-step lookahead for n >= 2.
    try:
        all_perms = list(permutations(range(1, n + 1)))
        perm_set = set(all_perms)
        total_perms = len(all_perms)
        used = set()

        # for each (n-1)-tuple prefix, the set of symbols that extend it to a permutation
        ext = {}
        for p in all_perms:
            pref = p[:-1]
            ext.setdefault(pref, set()).add(p[-1])
        
        # Precompute for each prefix how many unused permutations can be reached
        # This helps us avoid dead-ends more effectively
        def count_available(prefix):
            """Count how many new permutations can be reached from this prefix."""
            last = prefix[-1]
            count = 0
            for s in ext.get(prefix, ()):
                if s == last:
                    continue
                p = prefix + (s,)
                if p not in used:
                    count += 1
            return count
        
        def is_dead_end(prefix):
            return count_available(prefix) == 0

        # Start with the smallest permutation.
        word = list(range(1, n + 1))
        used.add(tuple(word))
        
        # Multi-step lookahead: prefer paths that maintain more options
        while len(used) < total_perms:
            if time.time() > deadline:
                raise RuntimeError("time")
            suffix = tuple(word[-(n - 1):])
            last = word[-1]
            
            # candidate symbols in ascending order, != last, that complete a NEW permutation
            cands = []
            for s in ext.get(suffix, ()):
                if s == last:
                    continue
                p = suffix + (s,)
                if p not in used:
                    new_suffix = (suffix + (s,))[-(n - 1):]
                    avail = count_available(new_suffix)
                    cands.append((s, p, avail))
            
            if not cands:
                raise RuntimeError("stuck")
            
            # Sort by: (1) avoid dead-ends, (2) maximize available options, (3) smallest symbol
            cands.sort(key=lambda x: (x[2] == 0, -x[2], x[0]))
            
            # Look 2 steps ahead for better choices
            chosen = None
            for s, p, avail in cands:
                if avail == 0:
                    continue  # Skip dead-ends
                new_suffix = (suffix + (s,))[-(n - 1):]
                
                # Check 2-step lookahead: from new_suffix, what's the best next move?
                best_next_avail = 0
                for s2 in ext.get(new_suffix, ()):
                    if s2 == s:
                        continue
                    p2 = new_suffix + (s2,)
                    if p2 not in used:
                        next_suffix = (new_suffix + (s2,))[-(n - 1):]
                        next_avail = count_available(next_suffix)
                        best_next_avail = max(best_next_avail, next_avail)
                
                # Prefer moves that lead to states with more future options
                if best_next_avail > 0:
                    chosen = (s, p)
                    break
            
            if chosen is None:
                # Fallback: pick the first non-dead-end, or first candidate if all are dead-ends
                for s, p, avail in cands:
                    chosen = (s, p)
                    break
            
            s, p = chosen
            word.append(s)
            used.add(p)

        # Verify completeness.
        if time.time() < deadline - 1:
            seen = set()
            w = word
            for i in range(len(w) - n + 1):
                seen.add(tuple(w[i:i + n]))
            if len(seen) == total_perms:
                return "".join(map(str, word))
        raise RuntimeError("verify failed")
    except Exception:
        # Fallback: classic construction (valid, just longer).
        return _classic(n)
