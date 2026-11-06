"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random

# Precompute smallest-prime-factor table up to 20000 (max sum = 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

# Precompute prime factor sets for all numbers up to 20000
_pf_cache = [None] * (_MAX_SUM + 1)
_pf_cache[1] = frozenset()
for _n in range(2, _MAX_SUM + 1):
    _factors = set()
    _x = _n
    while _x > 1:
        _p = _spf[_x]
        _factors.add(_p)
        while _x % _p == 0:
            _x //= _p
    _pf_cache[_n] = frozenset(_factors)

def _distinct_primes_in_sums(arr):
    """Return the set of distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            primes_set.update(_pf_cache[ai + arr[j]])
    return primes_set

def _objective(arr):
    return len(_distinct_primes_in_sums(arr))

def _prime_frequency(arr):
    """Count how many pair sums each prime divides."""
    freq = {}
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for p in _pf_cache[arr[i] + arr[j]]:
                freq[p] = freq.get(p, 0) + 1
    return freq

def _element_prime_contribution(arr, idx):
    """For element at idx, count how many pair sums it participates in and which primes those sums have."""
    contributions = {}
    n = len(arr)
    for j in range(n):
        if j == idx:
            continue
        s = arr[idx] + arr[j]
        for p in _pf_cache[s]:
            contributions[p] = contributions.get(p, 0) + 1
    return contributions

def _generate_structured_candidates():
    """Generate diverse structured candidate arrays."""
    structured = []
    
    # 1. Small consecutive numbers
    for start in range(1, 30):
        arr = list(range(start, start + 12))
        if max(arr) <= 10000:
            structured.append(arr)
    
    # 2. Arithmetic progressions with small parameters
    for d in range(1, 50):
        for a in range(1, 50):
            arr = [a + i * d for i in range(12)]
            if max(arr) <= 10000 and len(set(arr)) == 12:
                structured.append(arr)
    
    # 3. Multiples of small bases
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        max_mult = 10000 // base
        if max_mult < 12:
            continue
        # Regular consecutive multiples
        for start in range(1, min(50, max_mult - 11)):
            arr = [base * (start + i) for i in range(12)]
            if len(set(arr)) == 12 and max(arr) <= 10000:
                structured.append(arr)
        # Random multiples
        for _ in range(30):
            mults = random.sample(range(1, min(max_mult, 500) + 1), 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12 and max(arr) <= 10000:
                structured.append(arr)
    
    # 4. All even small numbers
    for _ in range(100):
        arr = random.sample(range(2, 300, 2), 12)
        structured.append(arr)
    
    # 5. All multiples of 3
    for _ in range(100):
        arr = random.sample(range(3, 300, 3), 12)
        structured.append(arr)
    
    # 6. All multiples of 6
    for _ in range(100):
        arr = random.sample(range(6, 500, 6), 12)
        structured.append(arr)
    
    # 7. Mix: numbers that are multiples of 2 or 3
    for _ in range(100):
        candidates = list(set(list(range(2, 200, 2)) + list(range(3, 200, 3))))
        arr = random.sample(candidates, 12)
        structured.append(arr)
    
    # 8. Very small numbers
    for _ in range(200):
        arr = random.sample(range(1, 100), 12)
        structured.append(arr)
    
    return structured

def _targeted_local_search(current, current_score, deadline, best_score, best_arr):
    """Local search using prime frequency to guide element replacements."""
    improved = True
    no_improve = 0
    current_set = set(current)
    
    while improved and time.time() < deadline:
        improved = False
        no_improve += 1
        if no_improve > 20:
            break
        
        # Compute prime frequency
        freq = _prime_frequency(current)
        
        # Identify expensive primes (appear in few sums)
        rare_primes = {p for p, c in freq.items() if c <= 2}
        
        # Try replacing each element
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Generate diverse candidates for replacement
            candidates = []
            
            # Strategy 1: Small numbers
            candidates.extend(random.sample(range(1, 100), 15))
            
            # Strategy 2: Even numbers
            candidates.extend(2 * m for m in random.sample(range(1, 100), 15))
            
            # Strategy 3: Multiples of 3
            candidates.extend(3 * m for m in random.sample(range(1, 100), 15))
            
            # Strategy 4: Multiples of 6
            candidates.extend(6 * m for m in random.sample(range(1, 50), 10))
            
            # Strategy 5: Targeted - avoid rare primes
            for _ in range(10):
                cand_val = random.randint(1, 300)
                # Quick check: does this create new rare primes?
                new_primes = set()
                for j in range(12):
                    if j == i:
                        continue
                    s = cand_val + current[j]
                    for p in _pf_cache[s]:
                        if p in rare_primes:
                            new_primes.add(p)
                            break
                if len(new_primes) == 0:
                    candidates.append(cand_val)
            
            # Try candidates
            for new_val in candidates:
                if time.time() > deadline:
                    break
                if new_val in current_set and new_val != current[i]:
                    continue
                if new_val < 1 or new_val > 10000:
                    continue
                
                cand = list(current)
                cand[i] = new_val
                score = _objective(cand)
                
                if score < current_score:
                    current = cand
                    current_score = score
                    current_set = set(cand)
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(current)
                    break
                elif score == current_score and random.random() < 0.1:
                    current = cand
                    current_set = set(cand)
                    improved = True
    
    return best_score, best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_arr = None
    best_score = float('inf')
    
    # --- Phase 1: Structured Constructions ---
    structured = _generate_structured_candidates()
    random.shuffle(structured)
    
    for arr in structured:
        if time.time() > deadline - 20:
            break
        arr = list(dict.fromkeys(arr))
        if len(arr) != 12:
            continue
        if any(x < 1 or x > 10000 for x in arr):
            continue
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            if best_score <= 4:
                break
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 2: Targeted Local Search from Best ---
    if time.time() < deadline - 15:
        current = list(best_arr)
        current_score = best_score
        local_deadline = min(deadline - 10, start_time + 40)
        best_score, best_arr = _targeted_local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts with Local Search ---
    if time.time() < deadline - 5:
        restarts = 0
        while time.time() < deadline - 3 and restarts < 200:
            restarts += 1
            
            r = random.random()
            if r < 0.15:
                arr = random.sample(range(1, 100), 12)
            elif r < 0.30:
                arr = random.sample(range(2, 300, 2), 12)
            elif r < 0.45:
                arr = random.sample(range(3, 300, 3), 12)
            elif r < 0.60:
                arr = random.sample(range(6, 500, 6), 12)
            elif r < 0.75:
                base = random.choice([2, 3, 4, 5, 6, 10, 12, 15, 20])
                max_m = min(10000 // base, 400)
                if max_m < 12:
                    arr = random.sample(range(1, 100), 12)
                else:
                    mults = random.sample(range(1, max_m + 1), 12)
                    arr = [base * m for m in mults]
            else:
                # Mix of small evens and multiples of 3
                candidates = list(set(list(range(2, 200, 2)) + list(range(3, 200, 3))))
                arr = random.sample(candidates, 12)
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Short local search from this candidate
            if time.time() < deadline - 3:
                cur = list(arr)
                cur_score = score
                search_deadline = min(deadline - 3, time.time() + 0.8)
                best_score, best_arr = _targeted_local_search(cur, cur_score, search_deadline, best_score, best_arr)
    
    return best_arr
