"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def is_square(x):
        if x < 0:
            return False
        r = math.isqrt(x)
        return r * r == x

    squares = set(k * k for k in range(1, 23))

    # Diagonal conflicts: 2a is a perfect square
    diag_conflict = set(a for a in range(1, 257) if is_square(2 * a))

    candidates = [a for a in range(1, 257) if a not in diag_conflict]

    # Build conflict sets
    conf = {a: set() for a in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if a + b in squares:
                conf[a].add(b)
                conf[b].add(a)

    # Greedy: sort by degree, add if no conflict
    sorted_cands = sorted(candidates, key=lambda x: len(conf[x]))
    current = set()
    for a in sorted_cands:
        if not (conf[a] & current):
            current.add(a)

    best = set(current)
    best_size = len(current)

    # Local search: try to add excluded elements
    excluded = [a for a in candidates if a not in current]

    for _ in range(100):
        if time.time() > deadline:
            break
        progress = False
        for a in list(excluded):
            if a in current:
                continue
            conflicts = conf[a] & current
            if not conflicts:
                current.add(a)
                excluded.remove(a)
                progress = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            elif len(conflicts) == 1:
                b = next(iter(conflicts))
                current.remove(b)
                current.add(a)
                excluded.remove(a)
                excluded.append(b)
                progress = True
        if not progress:
            break

    # Verify
    def is_valid(s):
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                if is_square(sl[i] + sl[j]):
                    return False
        return True

    if is_valid(best):
        return sorted(best)
    return list(range(1, 257, 3))
# EVOLVE-BLOCK-END
