"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # seconds (90s wall-clock limit, keep margin)


def _pair_sums_ok(lst):
    """Check if all pair sums (including a+a) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            b = lst[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a != b."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - a)
    return d


def _can_add(lst, x, sums_used, diffs_used):
    """
    Check if x can be added to lst maintaining Sidon property
    (pair sums unique) and that new differences don't conflict with diffs_used.
    sums_used: set of all pair sums currently in lst
    diffs_used: set of all nonzero differences currently in lst
    Returns (new_sums, new_diffs) if valid, else (None, None).
    """
    # Check new pair sums
    new_sums = set(sums_used)
    for a in lst:
        s = a + x
        if s in new_sums:
            return None, None
        new_sums.add(s)
    s = x + x
    if s in new_sums:
        return None, None
    new_sums.add(s)

    # Check new differences
    new_diffs = set(diffs_used)
    for a in lst:
        d1 = x - a
        d2 = a - x
        if d1 in new_diffs or d2 in new_diffs:
            return None, None
        new_diffs.add(d1)
        new_diffs.add(d2)

    return new_sums, new_diffs


def _greedy_build(t0, rng, order_mode):
    """
    Build two Sidon sets A and B greedily.
    order_mode: 'up' = 1..256, 'down' = 256..1, 'rand' = random order
    Returns (A, B).
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    if order_mode == 'up':
        candidates = list(range(1, N + 1))
    elif order_mode == 'down':
        candidates = list(range(N, 0, -1))
    else:
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)

    for x in candidates:
        if time.time() - t0 > DEADLINE:
            break
        # Try adding to A
        ns, nd = _can_add(A, x, sumsA, diffsA)
        if ns is not None:
            # Check difference disjointness with B
            if not (nd & diffsB):
                A.append(x)
                sumsA = ns
                diffsA = nd
                continue
        # Try adding to B
        ns, nd = _can_add(B, x, sumsB, diffsB)
        if ns is not None:
            # Check difference disjointness with A
            if not (nd & diffsA):
                B.append(x)
                sumsB = ns
                diffsB = nd

    return A, B


def _greedy_build_priority(t0, rng, priority):
    """
    Build two Sidon sets with priority on which set to try first.
    priority: 0 = prefer A, 1 = prefer B
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    for x in candidates:
        if time.time() - t0 > DEADLINE:
            break
        sides = (0, 1) if priority == 0 else (1, 0)
        added = False
        for side in sides:
            if side == 0:
                ns, nd = _can_add(A, x, sumsA, diffsA)
                if ns is not None and not (nd & diffsB):
                    A.append(x)
                    sumsA = ns
                    diffsA = nd
                    added = True
                    break
            else:
                ns, nd = _can_add(B, x, sumsB, diffsB)
                if ns is not None and not (nd & diffsA):
                    B.append(x)
                    sumsB = ns
                    diffsB = nd
                    added = True
                    break
        # If neither side accepts, skip x

    return A, B


def _try_local_improvement(A, B, t0, rng, max_iters=50):
    """
    Try to improve by swapping elements or adding new ones.
    """
    A = list(A)
    B = list(B)
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    # Recompute sums and diffs
    for i in range(len(A)):
        for j in range(i, len(A)):
            sumsA.add(A[i] + A[j])
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    for i in range(len(B)):
        for j in range(i, len(B)):
            sumsB.add(B[i] + B[j])
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])

    best_size = len(A) + len(B)
    best_A, best_B = list(A), list(B)

    for _ in range(max_iters):
        if time.time() - t0 > DEADLINE - 2:
            break

        # Try adding a random unused element
        used = set(A) | set(B)
        available = [x for x in range(1, N + 1) if x not in used]
        if available:
            x = rng.choice(available)
            # Try A
            ns, nd = _can_add(A, x, sumsA, diffsA)
            if ns is not None and not (nd & diffsB):
                A.append(x)
                sumsA = ns
                diffsA = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue
            # Try B
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                B.append(x)
                sumsB = ns
                diffsB = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue

        # Try swapping: remove random element from A or B, recompute, try to add back or add new
        if A:
            idx = rng.randrange(len(A))
            x = A.pop(idx)
            # Recompute sumsA and diffsA
            sumsA = set()
            diffsA = set()
            for i in range(len(A)):
                for j in range(i, len(A)):
                    sumsA.add(A[i] + A[j])
            for i in range(len(A)):
                for j in range(len(A)):
                    if i != j:
                        diffsA.add(A[j] - A[i])
            # Try to add x back to B
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                B.append(x)
                sumsB = ns
                diffsB = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue
            # Try adding x back to A
            ns, nd = _can_add(A, x, sumsA, diffsA)
            if ns is not None and not (nd & diffsB):
                A.append(x)
                sumsA = ns
                diffsA = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue
            # Try adding a different element to A
            used = set(A) | set(B)
            available = [y for y in range(1, N + 1) if y not in used]
            if available:
                y = rng.choice(available)
                ns, nd = _can_add(A, y, sumsA, diffsA)
                if ns is not None and not (nd & diffsB):
                    A.append(y)
                    sumsA = ns
                    diffsA = nd
                    if len(A) + len(B) > best_size:
                        best_size = len(A) + len(B)
                        best_A, best_B = list(A), list(B)
                    continue
        elif B:
            idx = rng.randrange(len(B))
            x = B.pop(idx)
            sumsB = set()
            diffsB = set()
            for i in range(len(B)):
                for j in range(i, len(B)):
                    sumsB.add(B[i] + B[j])
            for i in range(len(B)):
                for j in range(len(B)):
                    if i != j:
                        diffsB.add(B[j] - B[i])
            ns, nd = _can_add(A, x, sumsA, diffsA)
            if ns is not None and not (nd & diffsB):
                A.append(x)
                sumsA = ns
                diffsA = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                B.append(x)
                sumsB = ns
                diffsB = nd
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A, best_B = list(A), list(B)
                continue

    return best_A, best_B


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])
    if diffsA & diffsB:
        return False
    # Check all elements in [1, N]
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
    return True


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # Multiple strategies with different random seeds and orderings
    strategies = [
        ('up', 0), ('down', 0), ('rand', 0), ('rand', 1),
        ('up', 1), ('down', 1),
    ]

    for attempt in range(20):
        if time.time() - t0 > DEADLINE - 3:
            break

        mode = strategies[attempt % len(strategies)][0]
        priority = strategies[attempt % len(strategies)][1]

        if mode == 'rand':
            A, B = _greedy_build_priority(t0, rng, priority)
        elif mode == 'up':
            A, B = _greedy_build(t0, rng, 'up')
            if priority == 1:
                A, B = B, A
        elif mode == 'down':
            A, B = _greedy_build(t0, rng, 'down')
            if priority == 1:
                A, B = B, A

        if not A or not B:
            continue

        # Try local improvement
        A2, B2 = _try_local_improvement(A, B, t0, rng, max_iters=30)
        if len(A2) + len(B2) > len(A) + len(B):
            A, B = A2, B2

        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)

    # If we still have time, do more random restarts
    while time.time() - t0 < DEADLINE - 2:
        A, B = _greedy_build_priority(t0, rng, rng.choice([0, 1]))
        if not A or not B:
            continue
        A2, B2 = _try_local_improvement(A, B, t0, rng, max_iters=20)
        if len(A2) + len(B2) > len(A) + len(B):
            A, B = A2, B2
        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    # Final validation
    if not _validate(best_A, best_B):
        # Fallback: simple deterministic
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
