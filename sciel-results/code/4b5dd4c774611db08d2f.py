"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _prime_factors(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    s = set()
    while n % 2 == 0:
        s.add(2); n //= 2
    while n % 3 == 0:
        s.add(3); n //= 3
    f = 5
    while f * f <= n:
        while n % f == 0:
            s.add(f); n //= f
        f += 2
    if n > 1:
        s.add(n)
    return s

def _is_smooth(n, allowed):
    """Check if all prime factors of n are in allowed set."""
    if n <= 1:
        return True
    for p in allowed:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    deadline = time.time() + 80
    N = 12
    M = 210  # 2*3*5*7
    max_x = 10000 // M  # 47
    
    allowed_base = {2, 3, 5, 7}
    
    # Try to find N distinct positive integers in [1, max_x] such that
    # all pairwise sums have prime factors only in allowed_base.
    # This would give total distinct primes = 4.
    
    # Greedy + backtracking search for smooth-sum set
    def find_smooth_set():
        candidates = list(range(1, max_x + 1))
        for start in range(1, max_x + 1 - N + 1):
            if time.time() > deadline - 10:
                return None
            current = [start]
            for c in range(start + 1, max_x + 1):
                if len(current) >= N:
                    break
                ok = True
                for d in current:
                    if not _is_smooth(c + d, allowed_base):
                        ok = False
                        break
                if ok:
                    current.append(c)
            if len(current) >= N:
                return current[:N]
        return None
    
    smooth_set = find_smooth_set()
    if smooth_set is not None:
        result = [M * x for x in smooth_set]
        assert len(set(result)) == N
        assert all(1 <= x <= 10000 for x in result)
        return result
    
    # If no perfect smooth set found, do local search to minimize additional primes
    def score(xs):
        pr = set(allowed_base)
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                pr |= _prime_factors(xs[i] + xs[j])
        return len(pr)
    
    # Start with a reasonable initial set
    best_x = list(range(1, N + 1))
    best_s = score(best_x)
    
    # Local search: try swapping elements
    import random
    random.seed(42)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        xs = best_x[:]
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old = xs[i]
            for _ in range(100):
                new_x = random.randint(1, max_x)
                if new_x == old or new_x in xs:
                    continue
                xs[i] = new_x
                s = score(xs)
                if s < best_s:
                    best_s = s
                    best_x = xs[:]
                    improved = True
                xs[i] = old
    
    result = [M * x for x in best_x]
    assert len(set(result)) == N
    assert all(1 <= x <= 10000 for x in result)
    return result
# EVOLVE-BLOCK-END
