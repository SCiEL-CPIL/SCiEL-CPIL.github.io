"""Fresh design: greedy + local search for max valid subset of [1,256]."""
import time
import random
import itertools

# ---- Precompute forbidden triples (a<b<=c, a | (b+c)) ----
N = 256
# forbidden[a] = list of (b, c) such that a < b <= c and a | (b+c)
forbidden = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    # b+c must be a multiple of a, and a < b <= c <= N
    # b+c ranges from a+1+ (a+1)=2a+2? No: b>=a+1, c>=b so b+c>=2(a+1)
    # b+c <= 2N
    # multiples of a in [2(a+1), 2N]
    lo = 2 * (a + 1)
    hi = 2 * N
    first_k = (lo + a - 1) // a
    for k in range(first_k, hi // a + 1):
        s = k * a
        # b in [a+1, s//2], c = s - b, need c >= b and c <= N
        b_min = a + 1
        b_max = s // 2
        if b_max > N:
            b_max = N
        if b_min > b_max:
            continue
        for b in range(b_min, b_max + 1):
            c = s - b
            if c < b:
                continue
            if c > N:
                continue
            forbidden[a].append((b, c))

# Precompute for fast violation checks:
# For a given set S, a violation exists if there is a in S and (b,c) in forbidden[a] with b in S and c in S.
# We'll maintain counts.

def count_violations(S, S_set):
    """Count number of violated triples (a,b,c) with a<b<=c, a|b+c, all in S."""
    v = 0
    for a in S:
        for (b, c) in forbidden[a]:
            if b in S_set and c in S_set:
                v += 1
    return v

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    random.seed(42)

    # ---- Initial greedy: sort by "difficulty" and add if it doesn't create violations ----
    # A good heuristic: numbers with fewer forbidden outgoing edges are "safer" to include.
    # But we want max cardinality. Let's try a few greedy orders and local search.

    best_S = None
    best_size = 0

    # Greedy 1: ascending
    S = set()
    for x in range(1, N + 1):
        S.add(x)
        # check if adding x creates violations
        # x could be a, b, or c
        viol = False
        # x as a: need b,c in S already with a<b<=c
        for (b, c) in forbidden[x]:
            if b in S and c in S:
                viol = True
                break
        if not viol:
            # x as b: need a in S (a<x) and c in S (c>=x, a|b+c)
            # and x as c: need a in S and b in S (a<b<=x)
            # This is complex; let's just recompute violations
            pass
        # Recompute: if violations > 0, remove x
        if count_violations(list(S), S) > 0:
            S.discard(x)

    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    # Greedy 2: descending
    S = set()
    for x in range(N, 0, -1):
        S.add(x)
        if count_violations(list(S), S) > 0:
            S.discard(x)

    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    # Greedy 3: random order
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            S.add(x)
            if count_violations(list(S), S) > 0:
                S.discard(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # ---- Local search: add, remove, swap moves to maximize size ----
    if best_S is None:
        return list(range(192, 257))

    S = best_S
    S_set = set(S)

    # Precompute violation count
    v = count_violations(list(S), S_set)

    def try_add(x):
        """Return new violation count if x added, or None if x already in S."""
        if x in S_set:
            return None
        # Count new violations involving x
        dv = 0
        # x as a
        for (b, c) in forbidden[x]:
            if b in S_set and c in S_set:
                dv += 1
        # x as b: a < x <= c, a | (x+c), a in S, c in S
        for a in S_set:
            if a >= x:
                continue
            # need c >= x, c in S, a | (x+c)
            for c in S_set:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    dv += 1
        # x as c: a < b <= x, a | (b+x), a in S, b in S
        for a in S_set:
            if a >= x:
                continue
            for b in S_set:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    dv += 1
        return v + dv

    def try_remove(x):
        """Return new violation count if x removed."""
        if x not in S_set:
            return None
        # Count violations that involve x
        dv = 0
        # x as a
        for (b, c) in forbidden[x]:
            if b in S_set and c in S_set:
                dv += 1
        # x as b
        for a in S_set:
            if a >= x:
                continue
            for c in S_set:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    dv += 1
        # x as c
        for a in S_set:
            if a >= x:
                continue
            for b in S_set:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    dv += 1
        return v - dv

    # Simulated annealing style local search
    T = 5.0
    cooling = 0.999
    steps = 0
    while time.time() < deadline - 5.0:
        steps += 1
        # Pick a random move type
        move_type = random.random()
        if move_type < 0.4:
            # Add a random element not in S
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_v = try_add(x)
            if new_v is None:
                continue
            if new_v <= v or random.random() < 0.5:
                S_set.add(x)
                v = new_v
                if len(S_set) > best_size:
                    best_size = len(S_set)
                    best_S = set(S_set)
        elif move_type < 0.7:
            # Remove a random element from S
            if not S_set:
                continue
            x = random.choice(list(S_set))
            new_v = try_remove(x)
            if new_v is None:
                continue
            if new_v <= v or random.random() < 0.3:
                S_set.discard(x)
                v = new_v
        else:
            # Swap: remove one, add another
            if not S_set:
                continue
            x = random.choice(list(S_set))
            candidates = [y for y in range(1, N + 1) if y not in S_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Compute new violation count
            # Remove x, add y
            # We need to recompute from scratch for swap (simpler)
            S_set.discard(x)
            S_set.add(y)
            new_v = count_violations(list(S_set), S_set)
            if new_v <= v or random.random() < 0.4:
                v = new_v
                if len(S_set) > best_size:
                    best_size = len(S_set)
                    best_S = set(S_set)
            else:
                # Revert
                S_set.discard(y)
                S_set.add(x)
                v = count_violations(list(S_set), S_set)
            T *= cooling

    # Final polish: try to add any element that doesn't create violations
    for x in range(1, N + 1):
        if x not in best_S:
            S_test = set(best_S)
            S_test.add(x)
            if count_violations(list(S_test), S_test) == 0:
                best_S = S_test

    return list(best_S)
