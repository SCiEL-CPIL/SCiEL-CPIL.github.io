import time
import random
import heapq
from collections import Counter

MAX_VAL = 10000
N = 12

def _precompute():
    limit = 2 * MAX_VAL
    factors = [set() for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        if not factors[i]:
            for j in range(i, limit + 1, i):
                factors[j].add(i)
    return factors

FACTORS = _precompute()

def _get_cost(arr):
    u = set()
    n = len(arr)
    f = FACTORS
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            u.update(f[ai + arr[j]])
    return len(u)

def _get_prime_frequencies(arr):
    """Count how many pairwise sums each prime divides."""
    freq = Counter()
    n = len(arr)
    f = FACTORS
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            for p in f[ai + arr[j]]:
                freq[p] += 1
    return freq

def _get_essential_primes(arr, min_freq=1):
    """Primes that appear in at least min_freq pairwise sums."""
    freq = _get_prime_frequencies(arr)
    return {p for p, c in freq.items() if c >= min_freq}

def _get_optional_primes(arr):
    """Primes that appear in exactly 1 pairwise sum (easiest to eliminate)."""
    freq = _get_prime_frequencies(arr)
    return {p for p, c in freq.items() if c == 1}

def _set_cover_heuristic(arr):
    """
    Greedy set-cover: find minimum set of primes covering all pairwise sums.
    Returns the set of primes in the cover.
    """
    n = len(arr)
    f = FACTORS
    
    # Build: for each sum, which primes cover it
    sum_to_primes = {}
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            sum_to_primes[s] = f[s]
    
    # Count how many sums each prime covers
    prime_coverage = Counter()
    for s, primes in sum_to_primes.items():
        for p in primes:
            prime_coverage[p] += 1
    
    uncovered = set(sum_to_primes.keys())
    cover = set()
    
    while uncovered:
        # Find prime covering most uncovered sums
        best_p = None
        best_count = 0
        for p in prime_coverage:
            count = sum(1 for s in uncovered if p in sum_to_primes[s])
            if count > best_count:
                best_count = count
                best_p = p
        
        if best_p is None:
            break
            
        cover.add(best_p)
        uncovered = {s for s in uncovered if best_p not in sum_to_primes[s]}
    
    return cover

def _astar_construct(deadline):
    """
    A* search to build a set of N distinct integers minimizing new prime factors.
    State: (tuple of chosen integers, set of primes seen so far)
    Heuristic: lower bound on additional primes needed
    """
    start_time = time.time()
    f = FACTORS
    
    # Start with a good initial element
    # Try multiple starting points
    start_candidates = [2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 42, 60, 84, 120]
    
    best_solution = None
    best_cost = float('inf')
    
    for start_val in start_candidates:
        if time.time() > deadline:
            break
            
        # State: (f_score, g_score, chosen_tuple, primes_set)
        # g_score = current number of distinct primes
        initial_primes = set()
        initial_state = (0, 0, (start_val,), frozenset(initial_primes))
        
        # Priority queue: (f_score, counter, chosen_tuple, primes_set)
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, (start_val,), frozenset(initial_primes)))
        
        visited = set()
        max_expansions = 50000  # Limit to avoid timeout
        
        while open_set and max_expansions > 0:
            if time.time() > deadline:
                break
                
            max_expansions -= 1
            f_score, _, chosen_tuple, primes_set = heapq.heappop(open_set)
            
            state_key = (chosen_tuple, primes_set)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            if len(chosen_tuple) == N:
                cost = len(primes_set)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = list(chosen_tuple)
                break  # A* guarantees optimality for this path
            
            # Heuristic: minimum possible additional primes
            # Each new element adds at least 1 prime to each of its sums with existing elements
            # But many primes may overlap, so use a lower bound
            remaining = N - len(chosen_tuple)
            h = 0  # Conservative: assume no new primes (admissible)
            
            # Generate neighbors: add a new distinct integer
            existing = set(chosen_tuple)
            
            # Candidate values: focus on values that create sums with few new primes
            # Try values in ranges that share factors with existing sums
            candidate_vals = set()
            
            # Add values that are multiples of common primes
            for p in list(primes_set)[:20]:
                # Values v such that v + x is divisible by p for some existing x
                for x in chosen_tuple:
                    # v ≡ -x (mod p)
                    base = (-x) % p
                    # Add a few candidates in range [1, MAX_VAL]
                    for k in range(1, 5):
                        v = base + k * p
                        if 1 <= v <= MAX_VAL and v not in existing:
                            candidate_vals.add(v)
            
            # Also try some random values
            for _ in range(20):
                v = random.randint(1, MAX_VAL)
                if v not in existing:
                    candidate_vals.add(v)
            
            # Limit candidates to keep search tractable
            if len(candidate_vals) > 100:
                # Score candidates by how many new primes they'd introduce
                scored = []
                for v in candidate_vals:
                    new_primes = set()
                    for x in chosen_tuple:
                        s = v + x
                        for p in f[s]:
                            if p not in primes_set:
                                new_primes.add(p)
                    scored.append((len(new_primes), v))
                scored.sort()
                candidate_vals = {v for _, v in scored[:50]}
            
            for new_val in candidate_vals:
                if new_val in existing:
                    continue
                    
                new_chosen = chosen_tuple + (new_val,)
                
                # Compute new primes introduced
                new_primes = set(primes_set)
                for x in chosen_tuple:
                    s = new_val + x
                    new_primes.update(f[s])
                
                g_score = len(new_primes)
                f_score_new = g_score + h
                
                counter += 1
                heapq.heappush(open_set, (f_score_new, counter, new_chosen, frozenset(new_primes)))
        
        if best_solution is not None:
            break
    
    return best_solution, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_arr = None
    best_cost = float('inf')
    
    # 1. A* Guided Construction (primary strategy)
    astar_arr, astar_cost = _astar_construct(deadline - 10.0)
    if astar_arr is not None:
        best_cost = astar_cost
        best_arr = astar_arr[:]
    
    # 2. Structured Initializations (backup / diversity)
    if time.time() < deadline:
        # A: All even
        arr_a = list(range(2, 2*N + 1, 2))
        c_a = _get_cost(arr_a)
        if c_a < best_cost:
            best_cost = c_a
            best_arr = arr_a[:]
            
        # B: All multiples of 3
        arr_b = list(range(3, 3*N + 1, 3))
        c_b = _get_cost(arr_b)
        if c_b < best_cost:
            best_cost = c_b
            best_arr = arr_b[:]
            
        # C: All multiples of 5
        arr_c5 = list(range(5, 5*N + 1, 5))
        c_c5 = _get_cost(arr_c5)
        if c_c5 < best_cost:
            best_cost = c_c5
            best_arr = arr_c5[:]
            
        # D: Random
        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            arr_d = random.sample(range(1, MAX_VAL + 1), N)
            c_d = _get_cost(arr_d)
            if c_d < best_cost:
                best_cost = c_d
                best_arr = arr_d[:]
    
    # 3. Local Search
    if best_arr is None:
        best_arr = list(range(1, N + 1))
        best_cost = _get_cost(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    T = 5.0
    cooling = 0.99995
    step_count = 0
    
    while time.time() < deadline - 5.0:
        step_count += 1
        
        # Every 500 steps, recompute set cover to guide search
        if step_count % 500 == 0:
            cover = _set_cover_heuristic(current_arr)
            # Identify primes in cover that appear in few sums (easy to eliminate)
            freq = _get_prime_frequencies(current_arr)
            easy_to_eliminate = {p for p in cover if freq.get(p, 0) <= 2}
            
        i = random.randint(0, N - 1)
        old_val = current_arr[i]
        
        # Generate neighbor with set-cover awareness
        if random.random() < 0.6:
            # Local move
            step = random.randint(-50, 50)
            new_val = old_val + step
            if new_val < 1: new_val = 1
            if new_val > MAX_VAL: new_val = MAX_VAL
        elif random.random() < 0.3:
            # Medium jump
            new_val = random.randint(1, MAX_VAL // 2)
        else:
            # Global jump
            new_val = random.randint(1, MAX_VAL)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        distinct = True
        for k in range(N):
            if k != i and current_arr[k] == new_val:
                distinct = False
                break
        if not distinct:
            continue
            
        current_arr[i] = new_val
        new_cost = _get_cost(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if delta > 0:
                if T > 1e-6:
                    p = 2.718281828 ** (-delta / T)
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        current_arr[i] = old_val
                else:
                    current_arr[i] = old_val
                    
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # 4. Final polish: try to eliminate optional primes
    if time.time() < deadline - 2.0:
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
                
            optional = _get_optional_primes(best_arr)
            if not optional:
                break
                
            # Try to eliminate one optional prime
            target_p = random.choice(list(optional))
            
            # Find which pair(s) sum to a multiple of target_p
            n = len(best_arr)
            f = FACTORS
            pairs_to_fix = []
            for i in range(n):
                for j in range(i + 1, n):
                    if target_p in f[best_arr[i] + best_arr[j]]:
                        pairs_to_fix.append((i, j))
                        
            if not pairs_to_fix:
                continue
                
            # Try to fix one pair by changing one element
            i, j = random.choice(pairs_to_fix)
            old_i = best_arr[i]
            old_j = best_arr[j]
            
            # Try changing element i
            for attempt in range(10):
                new_i = old_i + random.randint(-20, 20)
                if new_i < 1 or new_i > MAX_VAL:
                    continue
                if any(new_i == best_arr[k] for k in range(n) if k != i):
                    continue
                    
                test_arr = best_arr[:]
                test_arr[i] = new_i
                test_cost = _get_cost(test_arr)
                
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_arr = test_arr
                    break
            else:
                # Try changing element j
                for attempt in range(10):
                    new_j = old_j + random.randint(-20, 20)
                    if new_j < 1 or new_j > MAX_VAL:
                        continue
                    if any(new_j == best_arr[k] for k in range(n) if k != j):
                        continue
                        
                    test_arr = best_arr[:]
                    test_arr[j] = new_j
                    test_cost = _get_cost(test_arr)
                    
                    if test_cost < best_cost:
                        best_cost = test_cost
                        best_arr = test_arr
                        break

    return best_arr
