"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random
from itertools import combinations

def _sieve_primes(limit):
    """Return list of primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 20000 (max sum is 10000+9999=19999)
_MAX_SUM = 20000
_primes_list = _sieve_primes(_MAX_SUM)
_prime_set = set(_primes_list)

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed primes."""
    factors = set()
    for p in _primes_list:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return factors

# Cache for factorization
_factor_cache = {}

def _prime_factors_cached(n):
    """Cached version of prime factorization."""
    if n not in _factor_cache:
        _factor_cache[n] = _prime_factors_fast(n)
    return _factor_cache[n]

def _build_state(arr):
    """Build incremental state: prime counts across all pairwise sums."""
    prime_counts = {}
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            for p in _prime_factors_cached(s):
                prime_counts[p] = prime_counts.get(p, 0) + 1
    return prime_counts

def _objective_from_state(prime_counts):
    """Count primes with nonzero count."""
    return sum(1 for c in prime_counts.values() if c > 0)

def _distinct_primes_in_sums(arr):
    """Count distinct primes dividing any sum of two distinct elements."""
    state = _build_state(arr)
    return set(p for p, c in state.items() if c > 0)

def _objective(arr):
    return len(_distinct_primes_in_sums(arr))

def _set_cover_heuristic(arr):
    """
    Graph coloring / set-cover approach:
    1. Compute all 66 pairwise sums
    2. For each sum, compute its prime factors
    3. Build a set cover: primes -> which sums they cover
    4. Greedily select primes that cover the most uncovered sums
    5. Return the minimum cover size found
    """
    n = len(arr)
    sums = []
    for i in range(n):
        for j in range(i+1, n):
            sums.append(arr[i] + arr[j])
    
    # Get all distinct sums and their prime factors
    sum_factors = {}
    for s in sums:
        if s not in sum_factors:
            sum_factors[s] = _prime_factors_cached(s)
    
    # Build set cover: prime -> set of sum indices it covers
    prime_to_sums = {}
    for idx, s in enumerate(sums):
        for p in sum_factors[s]:
            if p not in prime_to_sums:
                prime_to_sums[p] = set()
            prime_to_sums[p].add(idx)
    
    # Greedy set cover
    uncovered = set(range(len(sums)))
    cover_size = 0
    while uncovered:
        # Find prime that covers most uncovered sums
        best_prime = None
        best_count = 0
        for p, covered in prime_to_sums.items():
            count = len(covered & uncovered)
            if count > best_count:
                best_count = count
                best_prime = p
        if best_prime is None or best_count == 0:
            break
        uncovered -= prime_to_sums[best_prime]
        cover_size += 1
    
    return cover_size

def _random_candidates(arr, count):
    """Generate random perturbation candidates."""
    cands = []
    for _ in range(count):
        c = list(arr)
        for _ in range(random.randint(1, 4)):
            idx = random.randint(0, len(c)-1)
            c[idx] = random.randint(1, 10000)
        c = list(dict.fromkeys(c))  # dedupe preserving order
        if len(c) == 12 and all(1 <= x <= 10000 for x in c):
            cands.append(c)
    return cands

def _incremental_evaluate(arr, i, old_val, new_val, prime_counts):
    """
    Evaluate candidate arr with element i changed from old_val to new_val.
    Uses incremental update on prime_counts. Returns (new_score, new_prime_counts) or None if invalid.
    """
    n = len(arr)
    # Remove contributions of old_val at index i
    for j in range(n):
        if j == i:
            continue
        s = old_val + arr[j]
        for p in _prime_factors_cached(s):
            prime_counts[p] -= 1
            if prime_counts[p] == 0:
                del prime_counts[p]
    
    # Add contributions of new_val at index i
    for j in range(n):
        if j == i:
            continue
        s = new_val + arr[j]
        for p in _prime_factors_cached(s):
            prime_counts[p] = prime_counts.get(p, 0) + 1
    
    new_score = sum(1 for c in prime_counts.values() if c > 0)
    return new_score, prime_counts

def _local_search_incremental(arr, deadline, max_iters=5000):
    """
    Fast local search using incremental state updates.
    Each single-element swap is evaluated in O(n) instead of O(n^2).
    """
    current = list(arr)
    prime_counts = _build_state(current)
    current_score = _objective_from_state(prime_counts)
    best_score = current_score
    best_arr = list(current)
    
    iters = 0
    no_improve = 0
    while iters < max_iters and time.time() < deadline:
        iters += 1
        improved = False
        # Try each position
        for i in range(12):
            old_val = current[i]
            # Try several candidate values
            for _ in range(20):
                new_val = random.randint(1, 10000)
                if new_val == old_val:
                    continue
                # Check distinctness
                if new_val in current and new_val != old_val:
                    continue
                
                # Save old prime_counts
                old_pc = dict(prime_counts)
                
                # Incremental update
                result = _incremental_evaluate(current, i, old_val, new_val, prime_counts)
                if result is None:
                    prime_counts = old_pc
                    continue
                new_score, prime_counts = result
                
                if new_score < best_score:
                    # Accept
                    current[i] = new_val
                    best_score = new_score
                    best_arr = list(current)
                    improved = True
                    no_improve = 0
                    break
                else:
                    # Revert
                    prime_counts = old_pc
                    new_val = None  # mark as not accepted
            
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                break
    
    return best_arr, best_score

def _local_search_with_set_cover(arr, deadline):
    """Local search using set-cover heuristic as secondary objective."""
    current = list(arr)
    current_score = _objective(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single-element changes
        for i in range(12):
            for _ in range(15):
                new_val = random.randint(1, 10000)
                if new_val in current and new_val != current[i]:
                    continue
                cand = list(current)
                cand[i] = new_val
                if len(set(cand)) != 12 or max(cand) > 10000:
                    continue
                score = _objective(cand)
                if score < current_score:
                    current = cand
                    current_score = score
                    improved = True
                    break
                # Also check if set-cover heuristic suggests potential
                elif score == current_score and score > 4:
                    sc = _set_cover_heuristic(cand)
                    if sc < _set_cover_heuristic(current):
                        current = cand
                        current_score = score
                        improved = True
                        break
            if improved:
                break
    return current, current_score

def _targeted_prime_elimination(arr, prime_counts, deadline, max_iters=2000):
    """
    Targeted search: identify primes that appear in few sums and try to eliminate them.
    For each rare prime, try to find a value change that removes it from all sums.
    """
    current = list(arr)
    best_score = _objective_from_state(prime_counts)
    best_arr = list(current)
    best_pc = dict(prime_counts)
    
    iters = 0
    while iters < max_iters and time.time() < deadline:
        iters += 1
        
        # Find primes with smallest count (hardest to eliminate = most impactful)
        sorted_primes = sorted(prime_counts.items(), key=lambda x: x[1])
        
        found = False
        for target_prime, count in sorted_primes[:5]:
            if count <= 1:
                continue
            
            # For each element, try to find a new value that doesn't produce target_prime in any sum
            for i in range(12):
                old_val = current[i]
                
                # Check if old_val contributes to target_prime
                contributes = False
                for j in range(12):
                    if j == i:
                        continue
                    s = old_val + current[j]
                    if target_prime in _prime_factors_cached(s):
                        contributes = True
                        break
                
                if not contributes:
                    continue
                
                # Try random new values
                for _ in range(30):
                    new_val = random.randint(1, 10000)
                    if new_val == old_val:
                        continue
                    if new_val in current and new_val != old_val:
                        continue
                    
                    # Check if new_val still produces target_prime
                    still_produces = False
                    for j in range(12):
                        if j == i:
                            continue
                        s = new_val + current[j]
                        if target_prime in _prime_factors_cached(s):
                            still_produces = True
                            break
                    
                    if still_produces:
                        continue
                    
                    # Full evaluation
                    old_pc = dict(prime_counts)
                    result = _incremental_evaluate(current, i, old_val, new_val, prime_counts)
                    if result is None:
                        prime_counts = old_pc
                        continue
                    new_score, prime_counts = result
                    
                    if new_score < best_score:
                        current[i] = new_val
                        best_score = new_score
                        best_arr = list(current)
                        best_pc = dict(prime_counts)
                        found = True
                        break
                    else:
                        prime_counts = old_pc
                
                if found:
                    break
            
            if found:
                break
        
        if not found:
            break
    
    return best_arr, best_score, best_pc

def construct():
    deadline = time.time() + 80  # safety margin below 90s
    
    # --- Phase 1: Structured constructions ---
    best_arr = None
    best_score = float('inf')
    
    # Generate many structured candidates
    structured = []
    
    # Multiples of small numbers
    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
        # Take 12 multiples: base*1, base*2, ..., base*12
        arr = [base * (i+1) for i in range(12)]
        if max(arr) <= 10000 and len(set(arr)) == 12:
            structured.append(arr)
        # Take 12 random multiples
        for _ in range(5):
            mults = random.sample(range(1, 10000//base + 1), 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12 and max(arr) <= 10000:
                structured.append(arr)
    
    # All same parity (all even or all odd) with small values
    for _ in range(30):
        # Random 12 even numbers
        arr = random.sample(range(2, 10001, 2), 12)
        structured.append(arr)
    for _ in range(30):
        # Random 12 odd numbers
        arr = random.sample(range(1, 10001, 2), 12)
        structured.append(arr)
    
    # Small consecutive ranges
    for start in range(1, 60):
        arr = list(range(start, start+12))
        structured.append(arr)
    
    # Arithmetic progressions
    for a in range(1, 40):
        for d in range(1, 40):
            arr = [a + i*d for i in range(12)]
            if max(arr) <= 10000:
                structured.append(arr)
    
    # Evaluate structured candidates
    for arr in structured:
        if time.time() > deadline:
            break
        arr = list(dict.fromkeys(arr))
        if len(arr) != 12:
            continue
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            if best_score <= 3:  # very good, likely near-optimal
                break
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 2: Fast incremental local search ---
    if time.time() < deadline - 10:
        current, current_score = _local_search_incremental(best_arr, deadline - 10, max_iters=10000)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(current)
    
    # --- Phase 3: Targeted prime elimination ---
    if time.time() < deadline - 5:
        pc = _build_state(best_arr)
        current, current_score, pc = _targeted_prime_elimination(best_arr, pc, deadline - 5, max_iters=3000)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(current)
    
    # --- Phase 4: Random restarts with evaluation ---
    if time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline:
                break
            # Random 12 distinct numbers
            arr = random.sample(range(1, 10001), 12)
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            # Also try biased: small numbers, same parity
            if random.random() < 0.5:
                arr = random.sample(range(1, 200), 12)
            else:
                arr = random.sample(range(2, 400, 2), 12)
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
    
    # --- Phase 5: Final fast local search polish ---
    if time.time() < deadline - 2:
        current, current_score = _local_search_incremental(best_arr, deadline - 2, max_iters=5000)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(current)
    
    # --- Phase 6: Final targeted elimination ---
    if time.time() < deadline - 1:
        pc = _build_state(best_arr)
        current, current_score, pc = _targeted_prime_elimination(best_arr, pc, deadline - 1, max_iters=1000)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(current)
    
    return best_arr

# EVOLVE-BLOCK-END
