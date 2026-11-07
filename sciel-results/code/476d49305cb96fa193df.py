import time
import random

N = 256
# Identify large primes > 128
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
# Candidates for small set: 2..256 excluding large primes
CANDS = [x for x in range(2, N + 1) if x not in LARGE_PRIMES]
M = len(CANDS)
IDX = {v: i for i, v in enumerate(CANDS)}
BIT = [1 << i for i in range(M)]

def _precompute():
    # _DIV_BY[a][b] = mask of c such that CANDS[a] | CANDS[b] * CANDS[c]
    div_by = [[0] * M for _ in range(M)]
    # _DIVS_OF_PROD[a][b] = mask of c such that CANDS[c] | CANDS[a] * CANDS[b]
    divs_of_prod = [[0] * M for _ in range(M)]
    
    # Antichain masks
    div_mask = [0] * M # div_mask[x] = mask of y such that CANDS[x] | CANDS[y]
    mult_mask = [0] * M # mult_mask[x] = mask of y such that CANDS[y] | CANDS[x]
    
    for a in range(M):
        va = CANDS[a]
        row = div_by[a]
        row_prod = divs_of_prod[a]
        
        # Compute antichain masks for a
        for y in range(M):
            if CANDS[y] % va == 0:
                div_mask[a] |= BIT[y]
            if va % CANDS[y] == 0:
                mult_mask[a] |= BIT[y]
        
        for b in range(M):
            vb = CANDS[b]
            prod_ab = va * vb
            mask_div = 0
            mask_divs = 0
            for c in range(M):
                vc = CANDS[c]
                # Check a | b * c
                if (vb * vc) % va == 0:
                    mask_div |= BIT[c]
                # Check c | a * b
                if prod_ab % vc == 0:
                    mask_divs |= BIT[c]
            row[b] = mask_div
            row_prod[b] = mask_divs
            
    return div_by, divs_of_prod, div_mask, mult_mask

_DIV_BY, _DIVS_OF_PROD, _DIV_MASK, _MULT_MASK = _precompute()
_ALL = (1 << M) - 1

def _can_add(S, x):
    if S & BIT[x]:
        return False
    
    # Antichain check
    if (_DIV_MASK[x] & S) or (_MULT_MASK[x] & S):
        return False

    dx = _DIV_BY[x]
    dprod_x = _DIVS_OF_PROD[x] # Not directly used in loop, but _DIVS_OF_PROD[a][x] is
    
    for a in range(M):
        ba = BIT[a]
        if not (S & ba):
            continue
            
        # 1. a | x * c
        if (_DIV_BY[a][x] & S & ~ba & ~BIT[x]):
            return False
        
        # 2. x | a * c
        if (dx[a] & S & ~ba & ~BIT[x]):
            return False
        
        # 3. c | a * x
        # _DIVS_OF_PROD[a][x] is mask of c such that c | a*x
        if (_DIVS_OF_PROD[a][x] & S & ~ba & ~BIT[x]):
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

def _swap_improve(S, rng, rounds=5):
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

def _two_exchange(S, rng, trials=50):
    best = S
    best_sz = _popcount(S)
    for _ in range(trials):
        mems = [i for i in range(M) if S & BIT[i]]
        if not mems:
            break
        y = rng.choice(mems)
        S1 = S & ~BIT[y]
        cands = []
        for x1 in range(M):
            if x1 == y or (S1 & BIT[x1]):
                continue
            if _can_add(S1, x1):
                cands.append(x1)
        rng.shuffle(cands)
        for x1 in cands[:3]:
            S2 = S1 | BIT[x1]
            for x2 in range(M):
                if x2 == y or x2 == x1 or (S2 & BIT[x2]):
                    continue
                if _can_add(S2, x2):
                    S3 = S2 | BIT[x2]
                    S3 = _local_add(S3)
                    if _popcount(S3) > best_sz:
                        best = S3
                        best_sz = _popcount(S3)
                    break
    return best

def construct():
    deadline = time.time() + 50.0
    best_S = 0
    best = 0

    # Baseline: Greedy on sorted candidates
    order = list(range(M))
    S = _greedy(order)
    S = _local_add(S)
    best_S = S
    best = _popcount(S)

    for seed in range(100):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        order = list(range(M))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        S = _swap_improve(S, rng)
        S = _two_exchange(S, rng)
        S = _local_add(S)
        sz = _popcount(S)
        if sz > best:
            best = sz
            best_S = S

    # Combine with large primes
    small_members = _members(best_S)
    final_set = small_members + LARGE_PRIMES
    
    # Ensure distinct and sorted (optional, but good)
    final_set = list(set(final_set))
    
    return final_set
