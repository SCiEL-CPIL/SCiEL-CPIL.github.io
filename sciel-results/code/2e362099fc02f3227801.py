import time
import random

N = 256

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

LARGE_PRIMES = [p for p in range(129, N + 1) if is_prime(p)]
CANDS = [x for x in range(2, N + 1) if x not in LARGE_PRIMES]
M = len(CANDS)
IDX = {v: i for i, v in enumerate(CANDS)}
BIT = [1 << i for i in range(M)]

def _precompute():
    # For each candidate x, precompute conflict_mask[x] = bitmask of all y
    # such that {x, y} would cause a violation with SOME third element.
    # More precisely: conflict_mask[x] = union over all pairs (a,b) in CANDS
    # of {c : violation among {a,b,c} where x is one of a,b,c}
    # 
    # Actually, let's think differently. For a set S, x can be added iff
    # for all a,b in S: a ∤ x*b AND x ∤ a*b AND b ∤ a*x
    #
    # Define for each x and each a: 
    #   bad1[x][a] = bitmask of b such that a | x*b  (i.e., b is "bad" with a,x)
    #   bad2[x][a] = bitmask of b such that x | a*b
    #   bad3[x][a] = bitmask of b such that b | a*x
    #
    # Then x can be added to S iff for all a in S: (bad1[x][a] | bad2[x][a] | bad3[x][a]) & S == 0
    # But that's still O(|S|) per check.
    #
    # Better: precompute for each x a single bitmask conflict[x] such that
    # x can be added to S iff conflict[x] & S == 0.
    # This works if: x conflicts with S iff there exist a,b in S with violation.
    # But the violation involves THREE elements, so it's not pairwise.
    #
    # Alternative: For each x, define pair_conflict[x][a] = bitmask of b such that
    # {x, a, b} has a violation. Then x can be added to S iff for all a in S,
    # pair_conflict[x][a] & S == 0. Still O(|S|).
    #
    # Let's just precompute the triple masks efficiently and use a different approach:
    # Precompute for each pair (i,j): mask of k such that {CANDS[i], CANDS[j], CANDS[k]} violates.
    # Then for adding x to S: check if exists a in S such that triple_mask[x][a] & S is non-empty.
    
    # triple_mask[i][j] = bitmask of k such that {i,j,k} has a violation
    triple_mask = [[0] * M for _ in range(M)]
    
    for i in range(M):
        vi = CANDS[i]
        for j in range(i+1, M):
            vj = CANDS[j]
            prod = vi * vj
            mask = 0
            for k in range(M):
                if k == i or k == j:
                    continue
                vk = CANDS[k]
                # Check: i | j*k, j | i*k, k | i*j
                if (vj * vk) % vi == 0 or (vi * vk) % vj == 0 or prod % vk == 0:
                    mask |= BIT[k]
            triple_mask[i][j] = mask
            triple_mask[j][i] = mask
    
    return triple_mask

_TRIPLE = _precompute()

def _can_add(S, x):
    if S & BIT[x]:
        return False
    # For each a in S, check if triple_mask[x][a] & S is non-empty
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
        if _TRIPLE[x][a] & S:
            return False
    return True

def _popcount(S):
    return bin(S).count('1')

def _members(S):
    return [CANDS[i] for i in range(M) if S & BIT[i]]

def _local_add(S):
    changed = True
    while changed:
        changed = False
        for x in range(M):
            if not (S & BIT[x]) and _can_add(S, x):
                S |= BIT[x]
                changed = True
                break
    return S

def _greedy(order):
    S = 0
    for x in order:
        if _can_add(S, x):
            S |= BIT[x]
    return S

def _swap_improve(S, rng, rounds=10):
    for _ in range(rounds):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        best_add = None
        for x in range(M):
            if x == y or (S1 & BIT[x]):
                continue
            if _can_add(S1, x):
                best_add = x
                break
        if best_add is not None:
            S2 = S1 | BIT[best_add]
            S2 = _local_add(S2)
            if _popcount(S2) >= _popcount(S):
                S = S2
    return S

def _two_exchange(S, rng, trials=80):
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if len(mems) < 2:
            break
        y1, y2 = rng.sample(mems, 2)
        S1 = S & ~BIT[y1] & ~BIT[y2]
        cands = []
        for x in range(M):
            if (S1 & BIT[x]):
                continue
            if _can_add(S1, x):
                cands.append(x)
        if len(cands) >= 2:
            x1, x2 = rng.sample(cands, 2)
            S2 = S1 | BIT[x1] | BIT[x2]
            S2 = _local_add(S2)
            if _popcount(S2) > best_sz:
                best = S2
                best_sz = _popcount(S2)
    return best

def _one_two_exchange(S, rng, trials=80):
    """Remove one, add two."""
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        cands = []
        for x in range(M):
            if (S1 & BIT[x]):
                continue
            if _can_add(S1, x):
                cands.append(x)
        if len(cands) >= 2:
            x1, x2 = rng.sample(cands, 2)
            S2 = S1 | BIT[x1] | BIT[x2]
            S2 = _local_add(S2)
            if _popcount(S2) > best_sz:
                best = S2
                best_sz = _popcount(S2)
    return best

def construct():
    deadline = time.time() + 75.0
    best_S = 0
    best = 0

    # Baseline: Greedy on sorted candidates
    order = list(range(M))
    S = _greedy(order)
    S = _local_add(S)
    best_S = S
    best = _popcount(S)

    # Try multiple random orders
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        order = list(range(M))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        S = _swap_improve(S, rng)
        S = _two_exchange(S, rng)
        S = _one_two_exchange(S, rng)
        S = _local_add(S)
        sz = _popcount(S)
        if sz > best:
            best = sz
            best_S = S

    # Combine with large primes
    small_members = _members(best_S)
    final_set = small_members + LARGE_PRIMES
    
    final_set = list(set(final_set))
    
    return final_set
