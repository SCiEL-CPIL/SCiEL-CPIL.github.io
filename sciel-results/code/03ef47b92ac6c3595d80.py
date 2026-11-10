import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val=10000):
    """Generate smooth numbers with given prime bases."""
    result = set()
    def rec(idx, current):
        if current > max_val:
            return
        result.add(current)
        if idx < len(primes):
            val = current
            while val * primes[idx] <= max_val:
                val *= primes[idx]
                rec(idx + 1, val)
    rec(0, 1)
    return sorted(result)

def local_search(A, time_limit, rng, candidate_pool=None):
    """
    Perform local search to minimize the score.
    Uses simulated annealing style moves for better exploration.
    """
    A = sorted(list(A))
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    if candidate_pool is None:
        # Build a diverse candidate pool
        # 1. Small integers
        pool = list(range(1, 201))
        # 2. Smooth numbers (2,3,5)
        smooth_235 = _generate_smooth([2, 3, 5])
        pool.extend(smooth_235)
        # 3. Powers of 2 and 3
        for i in range(14):
            if 2**i <= 10000: pool.append(2**i)
            if 3**i <= 10000: pool.append(3**i)
        # 4. Some random numbers
        for _ in range(200):
            pool.append(rng.randint(1, 10000))
        
        pool = list(set(x for x in pool if 1 <= x <= 10000))
        rng.shuffle(pool)
        candidate_pool = pool

    # Simulated Annealing parameters
    T = 50.0
    cooling_rate = 0.999
    min_T = 0.5
    
    while (time.time() - start_time) < time_limit:
        # Choose a random element to replace
        i = rng.randint(0, len(A) - 1)
        old_val = A[i]
        
        # Choose a new value from the pool
        # Bias towards values close to old_val or small values
        new_val = candidate_pool[rng.randint(0, len(candidate_pool) - 1)]
        
        if new_val == old_val:
            continue
        if new_val in A:
            continue
        
        # Perform the move
        new_A = A.copy()
        new_A[i] = new_val
        new_A.sort()
        
        new_score = calc_score(new_A)
        delta = new_score - current_score
        
        # Acceptance probability
        if delta < 0:
            # Always accept improvements
            A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
        else:
            # Accept worse solutions with probability exp(-delta/T)
            if T > min_T:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    A = new_A
                    current_score = new_score
        
        T *= cooling_rate
        
        # Check time
        if (time.time() - start_time) >= time_limit:
            break

    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    rng = random.Random(42)
    
    candidates = []
    
    # Candidate 1: Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))
    
    # Candidate 2: 2,3-smooth numbers
    smooth_23 = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth_23.append(val)
    smooth_23 = sorted(set(smooth_23))
    candidates.append(smooth_23[:20])
    
    # Candidate 3: 2,3,5-smooth numbers
    smooth_235 = _generate_smooth([2, 3, 5])
    candidates.append(smooth_235[:20])
    
    # Candidate 4: 2,3,5,7-smooth numbers
    smooth_2357 = _generate_smooth([2, 3, 5, 7])
    candidates.append(smooth_2357[:20])
    
    # Candidate 5: Geometric progression ratio 1.2
    geo_12 = []
    val = 1
    for i in range(30):
        val = int(round(val * 1.2))
        if val > 10000:
            break
        if val not in geo_12:
            geo_12.append(val)
    if len(geo_12) >= 20:
        candidates.append(geo_12[:20])
    
    # Candidate 6: Geometric progression ratio 1.3
    geo_13 = []
    val = 1
    for i in range(30):
        val = int(round(val * 1.3))
        if val > 10000:
            break
        if val not in geo_13:
            geo_13.append(val)
    if len(geo_13) >= 20:
        candidates.append(geo_13[:20])
    
    # Candidate 7: Random small integers
    rand_small = random.sample(range(1, 100), 20)
    candidates.append(rand_small)
    
    # Candidate 8: Mixed - small AP + smooth
    mixed = list(range(1, 11)) + [x for x in smooth_235 if x > 10][:10]
    mixed = sorted(set(mixed))
    if len(mixed) >= 20:
        candidates.append(mixed[:20])
    
    best_A = None
    best_score = float('inf')
    
    # Allocate time for each candidate
    # We have 8 candidates. Let's give each a fair share of the time.
    # But some candidates might converge faster.
    # Let's run local search on each candidate for a fixed amount of time,
    # but ensure we finish before the deadline.
    
    time_per_candidate = (deadline - start_time) / len(candidates)
    
    for A in candidates:
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
            
        # Ensure validity
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
            
        # Run local search
        # Use a portion of the remaining time
        alloc_time = min(time_per_candidate, deadline - time.time() - 1)
        
        improved_A, score = local_search(A, alloc_time, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
        # If we've found a very good score, we might want to spend more time polishing it
        # But for now, let's just move to the next candidate to explore more.
        
    if best_A is None:
        best_A = list(range(1, 21))
        
    # Final polish: Run local search on the best found so far with remaining time
    if time.time() < deadline - 1:
        final_time = deadline - time.time() - 1
        if final_time > 1:
            best_A, best_score = local_search(best_A, final_time, rng)
            
    return best_A

