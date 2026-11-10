import time
import random
from itertools import combinations

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set()
    prodset = set()
    for i in range(len(A)):
        for j in range(len(A)):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes, max_count=200000):
    """Generate smooth numbers up to limit using given primes."""
    smooth = [1]
    i = 0
    while i < len(smooth) and len(smooth) < max_count:
        for p in primes:
            val = smooth[i] * p
            if val <= limit:
                smooth.append(val)
            else:
                break
        i += 1
    return sorted(set(smooth))

def local_search_fast(A, time_limit, rng, candidate_pool):
    """
    Fast local search with precomputed candidate pool.
    Uses incremental score updates where possible.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    n = len(A)
    
    # Precompute current sumset and prodset
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    
    improved = True
    no_improve_count = 0
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        
        # Shuffle candidates for diversity
        shuffled_pool = candidate_pool.copy()
        rng.shuffle(shuffled_pool)
        
        # Single-element swap moves
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in shuffled_pool:
                if new_val == old_val or new_val in A:
                    continue
                
                # Check if new_val is in range
                if new_val < 1 or new_val > 10000:
                    continue
                
                # Compute the delta in sumset and prodset sizes
                new_sumset_add = set()
                new_prodset_add = set()
                old_sumset_remove = set()
                old_prodset_remove = set()
                
                for other in A:
                    if other is not A[i]:
                        new_sumset_add.add(new_val + other)
                        new_sumset_add.add(other + new_val)
                        new_prodset_add.add(new_val * other)
                        new_prodset_add.add(other * new_val)
                        old_sumset_remove.add(old_val + other)
                        old_sumset_remove.add(other + old_val)
                        old_prodset_remove.add(old_val * other)
                        old_prodset_remove.add(other * old_val)
                
                new_sumset_add.add(2 * new_val)
                new_prodset_add.add(new_val * new_val)
                old_sumset_remove.add(2 * old_val)
                old_prodset_remove.add(old_val * old_val)
                
                # Estimate new sizes
                est_sumset_size = len(sumset) + len(new_sumset_add - sumset) - len(old_sumset_remove & (sumset - new_sumset_add))
                est_prodset_size = len(prodset) + len(new_prodset_add - prodset) - len(old_prodset_remove & (prodset - new_prodset_add))
                
                est_score = max(est_sumset_size, est_prodset_size)
                
                # Only compute full score if estimate suggests improvement
                if est_score < current_score:
                    new_A = A.copy()
                    new_A[i] = new_val
                    score = calc_score(new_A)
                    
                    if score < current_score:
                        current_score = score
                        A = new_A
                        improved = True
                        no_improve_count = 0
                        
                        # Update sumset and prodset
                        sumset = set(a + b for a in A for b in A)
                        prodset = set(a * b for a in A for b in A)
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        
                        break
        
        # Pair-swap moves: simultaneously replace two elements
        if (time.time() - start_time) < time_limit and no_improve_count < 3:
            for i in range(n - 1):
                if (time.time() - start_time) > time_limit:
                    return best_A, best_score
                for j in range(i + 1, n):
                    if (time.time() - start_time) > time_limit:
                        return best_A, best_score
                    
                    old_i = A[i]
                    old_j = A[j]
                    
                    # Try a few random pairs of replacements
                    for _ in range(3):
                        new_i = rng.choice(shuffled_pool)
                        new_j = rng.choice(shuffled_pool)
                        if new_i == old_i or new_j == old_j or new_i == new_j:
                            continue
                        if new_i in A or new_j in A:
                            continue
                        if new_i < 1 or new_i > 10000 or new_j < 1 or new_j > 10000:
                            continue
                        
                        new_A = A.copy()
                        new_A[i] = new_i
                        new_A[j] = new_j
                        
                        score = calc_score(new_A)
                        if score < current_score:
                            current_score = score
                            A = new_A
                            improved = True
                            no_improve_count = 0
                            sumset = set(a + b for a in A for b in A)
                            prodset = set(a * b for a in A for b in A)
                            if score < best_score:
                                best_score = score
                                best_A = list(A)
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate a rich pool of smooth numbers with more primes
    primes = [2, 3, 5, 7, 11]
    smooth_numbers = generate_smooth_numbers(10000, primes, max_count=200000)
    
    # Also add small integers for diversity
    candidate_pool = list(range(1, 51)) + smooth_numbers
    candidate_pool = sorted(set(candidate_pool))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initial candidates with different structures
    
    # Candidate 1: Smallest smooth numbers
    cand1 = smooth_numbers[:20]
    
    # Candidate 2: Powers of 2 and 3 mixed
    cand2 = []
    for e2 in range(15):
        for e3 in range(10):
            val = (2 ** e2) * (3 ** e3)
            if val <= 10000 and val not in cand2:
                cand2.append(val)
            if len(cand2) >= 20:
                break
        if len(cand2) >= 20:
            break
    cand2 = sorted(cand2[:20])
    
    # Candidate 3: Include 1 and some smooth numbers
    cand3 = [1] + smooth_numbers[1:19]
    
    # Candidate 4: Random selection from smooth numbers
    cand4 = sorted(rng.sample(smooth_numbers[:100], 20))
    
    # Candidate 5: Dense small numbers
    cand5 = list(range(1, 21))
    
    # Candidate 6: Geometric-like with smooth numbers
    cand6 = []
    val = 1
    while val <= 10000 and len(cand6) < 20:
        cand6.append(val)
        # Multiply by next small prime to get next value
        if val % 2 == 0:
            val = val * 3
        elif val % 3 == 0:
            val = val * 2
        else:
            val = val * 2
    
    # Candidate 7: Optimized - use numbers that are highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    
    # Candidate 8: Log-spaced smooth numbers for balance
    cand8 = []
    for target_log in [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7]:
        target = int(round(2 ** target_log))
        # Find nearest smooth number
        best_smooth = min(smooth_numbers, key=lambda x: abs(x - target))
        if best_smooth not in cand8:
            cand8.append(best_smooth)
    cand8 = sorted(cand8[:20])
    
    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]
    
    # Remove invalid candidates
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 20 and all(1 <= x <= 10000 for x in c):
            valid_candidates.append(c)
    
    # Evaluate initial scores
    initial_scores = []
    for c in valid_candidates:
        s = calc_score(c)
        initial_scores.append((s, c))
    
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search on the best candidates
    time_per_candidate = min(12.0, (deadline - time.time()) * 0.9 / max(1, len(valid_candidates)))
    
    for score, cand in initial_scores:
        if time.time() > deadline - 5.0:
            break
        
        if score < best_score:
            best_score = score
            best_A = list(cand)
        
        # Only do local search if we have enough time
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        alloc_time = min(time_per_candidate, remaining * 0.7)
        
        improved_A, ls_score = local_search_fast(cand, alloc_time, rng, candidate_pool)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = improved_A
    
    # Final check: try a few random restarts if time permits
    if best_A is None or time.time() < deadline - 20.0:
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Start from current best and perturb
            if best_A is not None:
                start_A = best_A.copy()
                # Perturb 2-3 elements
                for _ in range(rng.randint(2, 3)):
                    idx = rng.randint(0, 19)
                    new_val = rng.choice(candidate_pool)
                    while new_val in start_A:
                        new_val = rng.choice(candidate_pool)
                    start_A[idx] = new_val
            else:
                start_A = list(range(1, 21))
            
            if len(set(start_A)) != 20:
                continue
            
            s = calc_score(start_A)
            if s < best_score:
                best_score = s
                best_A = start_A
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            improved_A, ls_score = local_search_fast(start_A, min(15.0, remaining * 0.7), rng, candidate_pool)
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
