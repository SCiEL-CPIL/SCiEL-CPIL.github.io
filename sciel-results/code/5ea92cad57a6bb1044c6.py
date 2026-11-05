"""Minimum maximal no-isosceles set on [0,N)^2 via CSP backtracking."""
import time
import random
import itertools

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
M = len(ALL_POINTS)

# Precompute isosceles triangle triples
def _is_iso(a, b, c):
    ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
    ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
    bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
    return ab2 == ac2 or ab2 == bc2 or ac2 == bc2

TRIPLES = []
for i in range(M):
    for j in range(i+1, M):
        for k in range(j+1, M):
            if _is_iso(ALL_POINTS[i], ALL_POINTS[j], ALL_POINTS[k]):
                TRIPLES.append((i, j, k))

# For each point, the triples it participates in
PT_TRIPLES = [[] for _ in range(M)]
for (i, j, k) in TRIPLES:
    PT_TRIPLES[i].append((i, j, k))
    PT_TRIPLES[j].append((i, j, k))
    PT_TRIPLES[k].append((i, j, k))

def _is_maximal(chosen):
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if _is_iso(p, chosen[i], chosen[j]):
                    break
            else:
                return False
    return True

def _greedy_random():
    order = list(range(M))
    random.shuffle(order)
    chosen = []
    for idx in order:
        p = ALL_POINTS[idx]
        ok = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if _is_iso(p, ALL_POINTS[chosen[i]], ALL_POINTS[chosen[j]]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(idx)
    return chosen

def _try_remove(current, deadline):
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            test = current[:i] + current[i+1:]
            if _is_maximal(test):
                current = test
                improved = True
                break
    return current

def _csp_construct(deadline, best_size):
    """Backtracking CSP: assign each point IN or OUT (blocked).
    A point can be OUT only if it forms an iso triangle with 2 IN points.
    At the end, verify maximality (every OUT point is blocked by some pair).
    """
    assignment = [None] * M  # None, True(IN), False(OUT)
    in_count = 0
    best = [best_size]
    best_set = [None]
    rng = random.Random()

    # Precompute for each point the list of triples (for quick blocked check)
    # blocked(p) = exists (a,b,c) in TRIPLES containing p with other two IN

    def blocked(p):
        for (a, b, c) in PT_TRIPLES[p]:
            others = [x for x in (a, b, c) if x != p]
            if assignment[others[0]] is True and assignment[others[1]] is True:
                return True
        return False

    def backtrack(idx):
        nonlocal in_count, best, best_set
        if time.time() > deadline:
            return
        if in_count >= best[0]:
            return
        if idx == M:
            # Check maximality: every OUT point must be blocked
            for p in range(M):
                if assignment[p] is False:
                    if not blocked(p):
                        return
            # Valid maximal no-isosceles set
            chosen = [p for p in range(M) if assignment[p] is True]
            if len(chosen) < best[0]:
                best[0] = len(chosen)
                best_set[0] = chosen[:]
            return

        # Branching order: try OUT first (smaller set) but OUT requires blocked
        # We need to decide: can we put idx OUT? Only if blocked(idx) is currently true
        # But blocked depends on future assignments too. Standard CSP: assign then check consistency.
        # For OUT: we require that at the END it's blocked. To prune early, we can allow OUT
        # only if it's blocked NOW or could be blocked later. For simplicity, allow OUT if
        # blocked now OR if there exists a triple where the other two are not yet OUT.
        
        # Try IN first? No, we want small sets, so try OUT first if possible.
        # But OUT must eventually be blocked. If we assign OUT now, we must ensure it gets blocked.
        
        # Heuristic: try OUT first if it can be blocked (has a triple with both others not OUT)
        can_be_out = False
        for (a, b, c) in PT_TRIPLES[idx]:
            others = [x for x in (a, b, c) if x != idx]
            o1, o2 = others[0], others[1]
            # Both must not be OUT (they could be IN or unassigned)
            if assignment[o1] is not False and assignment[o2] is not False:
                can_be_out = True
                break

        if can_be_out:
            assignment[idx] = False
            if backtrack(idx + 1):
                return True
            assignment[idx] = None

        assignment[idx] = True
        in_count += 1
        if backtrack(idx + 1):
            return True
        in_count -= 1
        assignment[idx] = None
        return False

    # Use a randomized variable ordering to escape local minima
    # Actually, fixed order with IN/OUT branching is deterministic. 
    # To make it stochastic, we can shuffle the point order.
    order = list(range(M))
    rng.shuffle(order)
    
    # Remap: we need to backtrack over the shuffled order
    # Let's rewrite backtrack to use a position in the order
    def backtrack2(pos):
        nonlocal in_count, best, best_set
        if time.time() > deadline:
            return
        if in_count >= best[0]:
            return
        if pos == M:
            for p in range(M):
                if assignment[p] is False:
                    if not blocked(p):
                        return
            chosen = [p for p in range(M) if assignment[p] is True]
            if len(chosen) < best[0]:
                best[0] = len(chosen)
                best_set[0] = chosen[:]
            return

        idx = order[pos]
        can_be_out = False
        for (a, b, c) in PT_TRIPLES[idx]:
            others = [x for x in (a, b, c) if x != idx]
            o1, o2 = others[0], others[1]
            if assignment[o1] is not False and assignment[o2] is not False:
                can_be_out = True
                break

        if can_be_out:
            assignment[idx] = False
            if backtrack2(pos + 1):
                return True
            assignment[idx] = None

        assignment[idx] = True
        in_count += 1
        if backtrack2(pos + 1):
            return True
        in_count -= 1
        assignment[idx] = None
        return False

    backtrack2(0)
    return best_set[0]

def construct():
    start_time = time.time()
    deadline = start_time + 280.0

    best = None
    best_size = float('inf')

    # First, get a good upper bound via greedy
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        cand = _greedy_random()
        cand = _try_remove(cand, deadline - 30.0)
        if len(cand) < best_size:
            best_size = len(cand)
            best = cand

    if best is None:
        best = _greedy_random()
        best_size = len(best)

    # Now use CSP to try to beat the greedy bound
    # CSP with pruning by best_size
    while time.time() < deadline - 10.0:
        result = _csp_construct(deadline - 5.0, best_size)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best = result
            # Try to improve further
            best = _try_remove(best, deadline - 5.0)
            if len(best) < best_size:
                best_size = len(best)

    return best
