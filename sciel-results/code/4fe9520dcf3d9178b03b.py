import random
import time

# EVOLVE-BLOCK-START
N = 128

def _build_bitset(S):
    mask = 0
    for x in S:
        mask |= 1 << x
    return mask

def _check(S, mask):
    """Return True if S is valid (no triple a<b<c with all pair sums in S)."""
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i + 1, n):
            b = L[j]
            ab = a + b
            if ab > N or not (mask >> ab) & 1:
                continue
            for k in range(j + 1, n):
                c = L[k]
                if (mask >> (a + c)) & 1 and (mask >> (b + c)) & 1:
                    return False
    return True

def _greedy_start():
    # Prefer large numbers: smaller pair-sums are less likely to land in the set,
    # and large sums (>128) are automatically safe.
    S = set(range(65, 129))
    mask = _build_bitset(S)
    if not _check(S, mask):
        # Fallback: random greedy
        S = set()
        for x in random.sample(range(1, N + 1), N):
            cand = S | {x}
            m2 = mask | (1 << x)
            if _check(cand, m2):
                S = cand
                mask = m2
    return S, _build_bitset(S)

def _try_add(S, mask, x):
    if x in S or x < 1 or x > N:
        return False
    cand = S | {x}
    m2 = mask | (1 << x)
    if _check(cand, m2):
        return True
    return False

def _try_remove(S, mask, x):
    cand = S - {x}
    return cand, _build_bitset(cand)

def _random_walk(S, mask, steps, rng):
    best = (len(S), S)
    cur = set(S)
    cur_mask = mask
    for _ in range(steps):
        r = rng.random()
        if r < 0.45:
            # add a random element
            x = rng.randint(1, N)
            if x not in cur and _try_add(cur, cur_mask, x):
                cur.add(x)
                cur_mask |= (1 << x)
                if len(cur) > best[0]:
                    best = (len(cur), set(cur))
        elif r < 0.75:
            # remove a random element
            if cur:
                x = rng.choice(list(cur))
                cur.discard(x)
                cur_mask &= ~(1 << x)
        else:
            # swap: remove one, add another
            if cur:
                x = rng.choice(list(cur))
                y = rng.randint(1, N)
                if y not in cur:
                    trial = cur - {x}
                    trial.add(y)
                    tm = (cur_mask & ~(1 << x)) | (1 << y)
                    if _check(trial, tm):
                        cur = trial
                        cur_mask = tm
                        if len(cur) > best[0]:
                            best = (len(cur), set(cur))
    return best[1]

def _hill_climb_add(S, mask, rng, iters):
    improved = True
    while improved and iters > 0:
        improved = False
        iters -= 1
        for x in range(1, N + 1):
            if x not in S and _try_add(S, mask, x):
                S.add(x)
                mask |= (1 << x)
                improved = True
    return S, mask

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    rng = random.Random(20240517)

    S, mask = _greedy_start()
    best = (len(S), set(S))

    # First: pure add-hill-climb to saturate
    S, mask = _hill_climb_add(S, mask, rng, 200)
    if len(S) > best[0]:
        best = (len(S), set(S))

    # Randomized local search with add/remove/swap
    while time.time() < deadline:
        # Perturb the best solution
        S = set(best[1])
        mask = _build_bitset(S)
        # Random removals to escape local optimum
        for _ in range(rng.randint(0, 4)):
            if S:
                x = rng.choice(list(S))
                S.discard(x)
                mask &= ~(1 << x)
        # Re-saturate with adds
        S, mask = _hill_climb_add(S, mask, rng, 100)
        if len(S) > best[0]:
            best = (len(S), set(S))
        # Random walk from current
        S2 = _random_walk(S, mask, rng.randint(20, 60), rng)
        S2, m2 = _hill_climb_add(S2, _build_bitset(S2), rng, 100)
        if len(S2) > best[0]:
            best = (len(S2), set(S2))

    result = sorted(best[1])
    # Final safety: verify validity
    if not _check(set(result), _build_bitset(result)):
        # Should not happen, but fall back to seed
        result = list(range(65, 129))
    # Ensure Python int, list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
