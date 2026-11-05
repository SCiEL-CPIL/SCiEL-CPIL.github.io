import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    """
    N = 128

    # Precompute gcd table (1-indexed)
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    def can_add(x, sel):
        """True if x can be added to sel without creating a bad triple."""
        for a in sel:
            gax = gt[x][a]
            for b in sel:
                if b > a and gax == gt[x][b] == gt[a][b]:
                    return False
        return True

    def greedy(order):
        sel = set()
        for x in order:
            if can_add(x, sel):
                sel.add(x)
        return sel

    def polish(sel):
        """Repeatedly add any excluded element that fits (greedy completion)."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    changed = True
        return sel

    def local_search(sel, max_iters=200):
        """
        Local search with add, 1-1 swap, and 2-1 exchange moves.
        The 2-1 exchange (remove one, add two) is the key move for escaping
        local optima that only-adding cannot fix.
        """
        sel = set(sel)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # 1) Add any excluded element that fits
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    improved = True
                    break

            # 2) 2-1 exchange: remove one, add two
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in sel]
                random.shuffle(excluded)
                found = False
                for r in list(sel):
                    base = sel - {r}
                    added = []
                    for y in excluded:
                        if len(added) >= 2:
                            break
                        if can_add(y, base | set(added)):
                            added.append(y)
                    if len(added) >= 2:
                        sel = base | set(added)
                        improved = True
                        found = True
                        break
                if found:
                    continue

            # 3) 1-1 swap: remove one, add one (repositioning)
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in sel]
                random.shuffle(excluded)
                found = False
                for r in list(sel):
                    base = sel - {r}
                    for y in excluded:
                        if can_add(y, base):
                            sel = base | {y}
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

            # 4) Random perturbation to escape plateaus
            if not improved and random.random() < 0.3 and len(sel) > 1:
                r = random.choice(list(sel))
                sel.discard(r)
                improved = True

        return polish(sel)

    deadline = time.time() + 86.0
    best = set()
    best_size = 0

    def consider(s):
        nonlocal best, best_size
        if len(s) > best_size:
            best_size = len(s)
            best = set(s)

    random.seed(12345)

    # --- Diverse greedy seeds ---
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(reversed(sorted(range(1, N + 1), key=lambda x: x.bit_count()))),
        list(sorted(range(1, N + 1), key=lambda x: x.bit_count())),
        list(sorted(range(1, N + 1), key=lambda x: (x % 2, x))),
    ]
    for _ in range(6):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        s = polish(greedy(o))
        consider(s)
        if time.time() > deadline:
            break

    # --- Local search on each seed and on the running best ---
    for o in orders:
        if time.time() > deadline:
            break
        s = local_search(greedy(o))
        consider(s)

    # --- Repeated local search on the best set (refinement) ---
    for _ in range(15):
        if time.time() > deadline:
            break
        s = local_search(best)
        consider(s)

    # --- Final polish pass (high-precision completion) ---
    consider(polish(best))

    if not best:
        best = {1}
    return sorted(best)
# EVOLVE-BLOCK-END
