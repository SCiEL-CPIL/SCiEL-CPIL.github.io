"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random

NMAX = 256

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = b"\x00" * len(sieve[p * p::p])
    return [i for i in range(2, n + 1) if sieve[i]]

def _conflicts_with(x, chosen):
    """Return set of chosen elements that conflict with x."""
    conflicts = set()
    for a in chosen:
        # Check if a divides x * b for some b in chosen, b != a, b != x
        # i.e., a | x*b where b in chosen, b != a, b != x
        # Equivalent: there exists b in chosen, b != a, b != x, with a | x*b
        for b in chosen:
            if b == a or b == x:
                continue
            if (x * b) % a == 0:
                conflicts.add(a)
                break
    return conflicts

def _is_valid_set(S):
    """Check if set S satisfies: no a in S divides b*c for distinct b,c in S, c != b."""
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if (b * L[k]) % a == 0:
                    return False
    return True

def _fast_valid_check(S):
    """Faster validity check using the fact that we only need to check a | b*c."""
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if (b * L[k]) % a == 0:
                    return False
    return True

def _greedy_with_conflicts(deadline, order=None, seed=42):
    """Greedy construction with conflict tracking."""
    rng = random.Random(seed)
    if order is None:
        # Start with large numbers (they can't divide small products)
        candidates = list(range(2, NMAX + 1))
        rng.shuffle(candidates)
        # Sort: prefer larger numbers first (they're less likely to divide products)
        candidates.sort(key=lambda x: (-x, rng.random()))
    
    chosen = []
    chosen_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        # Check if x conflicts with any pair in chosen
        # x conflicts with a in chosen if there exists b in chosen (b != a, b != x) such that a | x*b
        # OR x conflicts with a if there exists b in chosen (b != x, b != a) such that a | x*b
        # Actually: the constraint is for every triple (a,b,c) distinct: a ∤ b*c
        # So x can be added if for all a in chosen, for all b in chosen (b != a, b != x): a ∤ x*b
        # AND for all a in chosen, for all b in chosen (b != a, b != x): a ∤ b*x (same thing)
        # AND x ∤ a*b for all a,b in chosen, a != b
        conflict = False
        for a in chosen:
            # Check if x divides a*b for some b in chosen, b != a
            for b in chosen:
                if b == a:
                    continue
                if (a * b) % x == 0:
                    conflict = True
                    break
            if conflict:
                break
            # Check if a divides x*b for some b in chosen, b != a
            for b in chosen:
                if b == a:
                    continue
                if (x * b) % a == 0:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _local_search(S, deadline, max_iters=500):
    """Local search: try adding elements, removing elements, swapping."""
    S = list(S)
    S_set = set(S)
    best = S
    best_size = len(S)
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        if len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                # Check if x can be added
                can_add = True
                for a in S:
                    for b in S:
                        if b == a:
                            continue
                        if (a * b) % x == 0:
                            can_add = False
                            break
                        if (x * b) % a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        
        # Try swapping: remove one, add another
        if S:
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                # Try a few random candidates
                rng.shuffle(candidates)
                for x in candidates[:20]:
                    can_add = True
                    for a in S_temp:
                        for b in S_temp:
                            if b == a:
                                continue
                            if (a * b) % x == 0:
                                can_add = False
                                break
                            if (x * b) % a == 0:
                                can_add = False
                                break
                        if not can_add:
                            break
                    if can_add:
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break
                    else:
                        S = S_temp + [removed]
                        S_set = set(S)
                        break
            else:
                S = S_temp + [removed]
                S_set = set(S)
    
    return best

def _greedy_largest_first(deadline):
    """Greedy: try numbers from largest to smallest."""
    chosen = []
    chosen_set = set()
    
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        # Check if x can be added
        conflict = False
        for a in chosen:
            for b in chosen:
                if b == a:
                    continue
                if (a * b) % x == 0:
                    conflict = True
                    break
                if (x * b) % a == 0:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_with_pruning(deadline, seed=42):
    """Greedy with smarter ordering: numbers with few prime factors first."""
    rng = random.Random(seed)
    
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    candidates = list(range(2, NMAX + 1))
    # Sort by (number of prime factors, -value, random)
    # Numbers with fewer prime factors are "safer" to include
    candidates.sort(key=lambda x: (count_prime_factors(x), -x, rng.random()))
    
    chosen = []
    chosen_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        conflict = False
        for a in chosen:
            for b in chosen:
                if b == a:
                    continue
                if (a * b) % x == 0:
                    conflict = True
                    break
                if (x * b) % a == 0:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best = None
    best_size = 0
    
    all_primes = _primes_upto(NMAX)
    
    # Candidate 1: all primes (baseline)
    if _fast_valid_check(all_primes):
        best = all_primes
        best_size = len(all_primes)
    
    # Candidate 2: greedy largest first
    if time.time() < deadline:
        c = _greedy_largest_first(deadline - 5)
        if c and len(c) > best_size:
            if _fast_valid_check(c):
                best = c
                best_size = len(c)
    
    # Candidate 3: greedy with pruning
    if time.time() < deadline - 5:
        c = _greedy_with_pruning(deadline - 5, seed=42)
        if c and len(c) > best_size:
            if _fast_valid_check(c):
                best = c
                best_size = len(c)
    
    # Candidate 4: randomized greedy with different seeds
    for seed in [1, 7, 13, 99]:
        if time.time() > deadline - 10:
            break
        c = _greedy_with_conflicts(deadline - 10, seed=seed)
        if c and len(c) > best_size:
            if _fast_valid_check(c):
                best = c
                best_size = len(c)
    
    # Local search to improve
    if best and time.time() < deadline - 5:
        c = _local_search(best, deadline - 5, max_iters=200)
        if c and len(c) > best_size:
            if _fast_valid_check(c):
                best = c
                best_size = len(c)
    
    # Final: try more aggressive local search from multiple starts
    for seed in [2, 5, 11, 23, 42, 100]:
        if time.time() > deadline - 3:
            break
        c = _greedy_with_pruning(deadline - 3, seed=seed)
        if c:
            c2 = _local_search(c, deadline - 3, max_iters=100)
            if c2 and len(c2) > best_size:
                if _fast_valid_check(c2):
                    best = c2
                    best_size = len(c2)
    
    if best is None:
        best = all_primes
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
