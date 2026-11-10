"""Rewritten: max-weight independent set on coprime-conflict graph, 1..256."""
# EVOLVE-BLOCK-START
import time

_N = 256
_DEADLINE = None
_START = None

def _setup():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0  # safety margin under 90s kill

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factor_mask(x, primes):
    m = 0
    for i, p in enumerate(primes):
        if p > x:
            break
        if x % p == 0:
            m |= (1 << i)
    return m

def construct():
    _setup()
    primes = _primes_upto(_N)
    np_ = len(primes)
    # mask for each number 1..256 (index 0 unused)
    mask = [0] * (_N + 1)
    for x in range(2, _N + 1):
        mask[x] = _factor_mask(x, primes)
    # number 1 has empty mask

    # Upper bound per prime: largest prime power <= N
    prime_best = {}
    for p in primes:
        best = 1
        v = p
        while v <= _N:
            best = v
            v *= p
        prime_best[p] = best
    # Sum of best prime powers + 1 (for the number one) is a global upper bound
    global_ub = 1 + sum(prime_best.values())

    # Conflict: two numbers conflict if their masks intersect
    # We'll do branch & bound over numbers, ordered by weight descending.
    # Candidate numbers: 1..256. Weight = the number itself.
    # We need pairwise coprime => masks pairwise disjoint.

    # Precompute conflict: for speed, store mask per number.
    # Branching: pick a number, include or exclude.
    # Use a recursive search with a "used mask" of primes already taken.
    # Upper bound at a node: current_sum + sum over free primes of best prime power
    #   that is still available (not conflicting with used mask) ... 
    #   but a prime power p^k might itself conflict with used primes? No: p^k only
    #   uses prime p. So if p is free, p^k is available. Good.
    #   So UB = current + sum_{p free} prime_best[p]. This is valid and tight-ish.

    # Order primes by best value descending for tighter bound ordering (not needed for sum).
    # Map prime index -> best value
    pidx_best = [0] * np_
    for i, p in enumerate(primes):
        pidx_best[i] = prime_best[p]

    best_sum = [0]
    best_set = [set()]

    # Numbers sorted by value descending; we consider including each.
    # To bound, at each step we track used_mask (primes taken).
    # Free primes sum = sum of pidx_best[i] for bits not in used_mask.
    # Precompute for speed: we can compute free sum by iterating bits; np_ is ~54, fine.

    def free_sum(used_mask):
        s = 0
        um = used_mask
        for i in range(np_):
            if not (um >> i) & 1:
                s += pidx_best[i]
        return s

    # We include number 1 optionally; it has mask 0, value 1.
    # Strategy: branch over numbers in descending value order.
    # Include number x if mask[x] & used_mask == 0.
    # Also try excluding x (move to next).

    # To make it fast, use a greedy initial solution.
    # Greedy: take 1, then for each prime take its best prime power.
    greedy = [1]
    for p in primes:
        greedy.append(prime_best[p])
    # verify pairwise coprime (they are by construction: distinct primes, and 1)
    best_sum[0] = sum(greedy)
    best_set[0] = set(greedy)

    # Now branch and bound.
    # State: index into sorted numbers, used_mask, current_sum, current_set.
    # We sort numbers by value descending to find good solutions early.
    numbers = list(range(2, _N + 1))  # exclude 1, handle 1 separately (always include)
    numbers.sort(key=lambda x: -x)

    # Since 1 is always beneficial (value 1, no conflict), always include it.
    # So start current_sum = 1, used_mask = 0, set = {1}.

    def dfs(idx, used_mask, cur_sum, cur_set):
        if time.time() > _DEADLINE:
            return
        # Upper bound
        ub = cur_sum + free_sum(used_mask)
        if ub <= best_sum[0]:
            return
        if idx >= len(numbers):
            return
        x = numbers[idx]
        mx = mask[x]
        # Option 1: include x if possible
        if (mx & used_mask) == 0:
            new_sum = cur_sum + x
            if new_sum > best_sum[0]:
                best_sum[0] = new_sum
                best_set[0] = set(cur_set) | {x}
            dfs(idx + 1, used_mask | mx, new_sum, cur_set | {x})
        # Option 2: exclude x
        dfs(idx + 1, used_mask, cur_sum, cur_set)

    # Start with 1 included
    dfs(0, 0, 1, {1})

    result = list(best_set[0])
    # Ensure pure Python ints and a list
    result = [int(v) for v in result]
    return result
# EVOLVE-BLOCK-END
