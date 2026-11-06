"""Max independent set on square-sum conflict graph over [1,256]."""
import math
import time
import random

def construct():
    start = time.time()
    deadline = 80.0

    n = 256
    squares = set()
    i = 1
    while i * i <= 2 * n:
        squares.add(i * i)
        i += 1

    # Self-loop: 2a is square -> a forbidden
    self_loop = [False] * (n + 1)
    for a in range(1, n + 1):
        if (2 * a) in squares:
            self_loop[a] = True

    candidates = [a for a in range(1, n + 1) if not self_loop[a]]
    m = len(candidates)

    # Precompute conflict pairs as a set of frozensets for O(1) check
    # conflict[a] = set of b in candidates (b > a) where a+b is square
    conflict = {a: set() for a in candidates}
    for i_idx, a in enumerate(candidates):
        for b in candidates[i_idx + 1:]:
            if (a + b) in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Greedy initial solution: pick vertices in order of increasing degree
    def greedy_initial():
        S = set()
        rem = list(candidates)
        # sort by degree ascending
        rem.sort(key=lambda x: len(conflict[x]))
        for a in rem:
            if a in S:
                continue
            # check conflicts with current S
            ok = True
            for b in S:
                if b in conflict[a]:
                    ok = False
                    break
            if ok:
                S.add(a)
        return S

    S = greedy_initial()

    def is_valid(S):
        for a in S:
            if self_loop[a]:
                return False
        for a in S:
            for b in S:
                if a < b and b in conflict[a]:
                    return False
        return True

    # Local search: add, swap1-1, swap1-2
    def try_add(S):
        for a in candidates:
            if a in S:
                continue
            ok = True
            for b in S:
                if b in conflict[a]:
                    ok = False
                    break
            if ok:
                S.add(a)
                return True
        return False

    def try_swap11(S):
        for a in list(S):
            S.remove(a)
            for x in candidates:
                if x == a or x in S:
                    continue
                ok = True
                for b in S:
                    if b in conflict[x]:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    return True
            S.add(a)
        return False

    def try_swap12(S):
        for a in list(S):
            S.remove(a)
            added = 0
            for x in candidates:
                if x == a or x in S:
                    continue
                ok = True
                for b in S:
                    if b in conflict[x]:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    added += 1
                    if added == 2:
                        break
            if added >= 1:
                return True
            else:
                S.add(a)
        return False

    # Iterate local search
    improved = True
    while improved and time.time() - start < deadline - 5:
        improved = False
        if try_add(S):
            improved = True
        elif try_swap11(S):
            improved = True
        elif try_swap12(S):
            improved = True

    # Ensure validity
    if not is_valid(S):
        S = set()
        for a in sorted(candidates, key=lambda x: len(conflict[x])):
            ok = True
            for b in S:
                if b in conflict[a]:
                    ok = False
                    break
            if ok:
                S.add(a)

    if not S:
        S = {1}

    return list(S)
