import random
import time

N = 256

def _is_sidon(S):
    S = sorted(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal_fast(S, sums):
    """Check maximality: every missing element must be blocked."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for x in range(1, N+1):
        if x in S_set:
            continue
        if 2*x in sums:
            continue
        # Check if any s in S has x+s in sums
        blocked = False
        for s in S_list:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal():
    selected = []
    sums = set()
    for x in range(1, N+1):
        added = [x + a for a in selected] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce(S, deadline=None):
    """Try to remove elements while maintaining maximality."""
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        if deadline and time.time() > deadline:
            break
        for i in range(len(S)):
            if deadline and time.time() > deadline:
                break
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def _local_search_swap(S, sums, deadline):
    """Try swap moves: remove one element, add another, check if still maximal and smaller or same size."""
    S = sorted(S)
    S_set = set(S)
    improved = False
    for i in range(len(S)):
        if time.time() > deadline:
            break
        # Try removing S[i]
        S_without = S[:i] + S[i+1:]
        if not S_without:
            continue
        if not _is_sidon(S_without):
            continue
        sums_without = _get_sums(S_without)
        S_without_set = set(S_without)
        
        # Now try adding each missing element
        for x in range(1, N+1):
            if x in S_without_set:
                continue
            # Check if x can be added to S_without
            added = [x + a for a in S_without] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums_without.intersection(added):
                continue
            # S_without + [x] is a valid Sidon set of same size
            # Check if it's maximal
            new_S = S_without + [x]
            new_S.sort()
            new_sums = sums_without | set(added)
            if _is_maximal_fast(new_S, new_sums):
                return new_S
    return None

def _random_sidon_set(seed_offset=0):
    """Build a random Sidon set."""
    S = set()
    sums = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in S] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            S.add(x)
            sums.update(added)
    return sorted(S), sums

def _random_sidon_set_weighted(weights, seed_offset=0):
    """Build a Sidon set with weighted random selection."""
    S = set()
    sums = set()
    candidates = list(range(1, N+1))
    # Weighted shuffle
    for i in range(len(candidates) - 1, 0, -1):
        j = random.randint(0, i)
        candidates[i], candidates[j] = candidates[j], candidates[i]
    for x in candidates:
        added = [x + a for a in S] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            S.add(x)
            sums.update(added)
    return sorted(S), sums

def construct():
    deadline = time.time() + 25  # safety margin
    
    best = _greedy_maximal()
    best_size = len(best)
    
    # Try to reduce the greedy result
    reduced = _reduce(best, deadline)
    if len(reduced) < best_size:
        best = reduced
        best_size = len(best)
    
    # Try random approaches with multiple strategies
    random.seed(42)
    strategies = [
        'uniform',      # uniform random order
        'low_first',    # bias toward low numbers
        'high_first',   # bias toward high numbers
        'mid_first',    # bias toward middle
        'prime_bias',   # bias toward primes
    ]
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        strategy = strategies[attempt % len(strategies)]
        
        if strategy == 'uniform':
            S, sums = _random_sidon_set()
        elif strategy == 'low_first':
            # Weight lower numbers more
            S = set()
            sums = set()
            candidates = list(range(1, N+1))
            # Custom shuffle favoring low numbers
            for i in range(len(candidates) - 1, 0, -1):
                # Biased random
                j = random.randint(0, min(i, i * 2 // 3))
                candidates[i], candidates[j] = candidates[j], candidates[i]
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        elif strategy == 'high_first':
            S = set()
            sums = set()
            candidates = list(range(1, N+1))
            for i in range(len(candidates) - 1, 0, -1):
                j = random.randint(max(0, i - i * 2 // 3), i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        elif strategy == 'mid_first':
            S = set()
            sums = set()
            # Start with middle, then expand outward
            mid = N // 2
            candidates = [mid]
            for d in range(1, N):
                candidates.append(mid + d)
                if mid - d >= 1:
                    candidates.append(mid - d)
            random.shuffle(candidates[1:])  # Keep mid first, shuffle rest
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        else:  # prime_bias
            S = set()
            sums = set()
            # Generate primes up to N
            is_prime = [True] * (N+1)
            is_prime[0] = is_prime[1] = False
            for p in range(2, int(N**0.5)+1):
                if is_prime[p]:
                    for m in range(p*p, N+1, p):
                        is_prime[m] = False
            primes = [x for x in range(1, N+1) if is_prime[x]]
            non_primes = [x for x in range(1, N+1) if not is_prime[x]]
            random.shuffle(primes)
            random.shuffle(non_primes)
            candidates = primes + non_primes
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        
        if len(S) >= best_size:
            continue
        
        # Check maximality
        if _is_maximal_fast(S, sums):
            best = S
            best_size = len(best)
            # Try to reduce further
            reduced = _reduce(best, deadline)
            if len(reduced) < best_size:
                best = reduced
                best_size = len(best)
            # Try swap local search
            if time.time() < deadline:
                swapped = _local_search_swap(best, _get_sums(best), deadline)
                if swapped and len(swapped) < best_size:
                    best = swapped
                    best_size = len(best)
                    reduced = _reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(best)
    
    return best
