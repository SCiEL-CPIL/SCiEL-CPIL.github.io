import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    
    def score(A):
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    # Structural change: Use a structured candidate pool and greedy modularity-like selection
    # instead of random swaps. We build a pool of "smooth" numbers (many small prime factors)
    # which tend to have more product collisions. Then we select 20 from this pool
    # using a greedy algorithm that maximizes internal collision density.
    
    # Generate candidate pool: numbers with many divisors (smooth numbers)
    candidates = set()
    # Include small integers
    for i in range(1, 200):
        candidates.add(i)
    # Include numbers with small prime factors
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    for p1 in primes:
        for p2 in primes:
            for p3 in primes:
                val = p1 * p2 * p3
                if val <= max_val:
                    candidates.add(val)
                if p1 * p2 <= max_val:
                    candidates.add(p1 * p2)
                if p1 <= max_val:
                    candidates.add(p1)
    # Add some structured sets
    for i in range(1, 100):
        candidates.add(2 * i)
        candidates.add(3 * i)
        candidates.add(6 * i)
    
    candidate_list = list(candidates)
    candidate_list = [x for x in candidate_list if 1 <= x <= max_val]
    
    # Greedy selection: start with a good seed, then add elements that maximize
    # collision density with existing set
    best_A = None
    best_score = float('inf')
    
    # Try multiple starting points
    start_points = [1, 2, 3, 6, 12, 24, 48]
    
    for start_val in start_points:
        current = [start_val]
        current_set = {start_val}
        
        while len(current) < n:
            best_candidate = None
            best_collisions = -1
            
            for cand in candidate_list:
                if cand in current_set:
                    continue
                
                # Count how many new sums and products this candidate creates
                # that already exist in the current set's sum/product sets
                current_sums = set()
                current_prods = set()
                for a in current:
                    for b in current:
                        current_sums.add(a + b)
                        current_prods.add(a * b)
                
                collisions = 0
                for a in current:
                    if (a + cand) in current_sums:
                        collisions += 1
                    if (a * cand) in current_prods:
                        collisions += 1
                
                if collisions > best_collisions:
                    best_collisions = collisions
                    best_candidate = cand
            
            if best_candidate is not None:
                current.append(best_candidate)
                current_set.add(best_candidate)
        
        sc = score(current)
        if sc < best_score:
            best_score = sc
            best_A = current[:]
    
    # Now refine with local search using the structured candidate pool
    start_time = time.time()
    deadline = start_time + 55
    
    current_A = best_A[:]
    current_score = best_score
    current_set = set(current_A)
    
    # Local search: swap elements with candidates that improve collision density
    while time.time() < deadline:
        improved = False
        
        for i in range(n):
            old_val = current_A[i]
            best_new_val = None
            best_new_score = current_score
            
            for cand in candidate_list:
                if cand in current_set:
                    continue
                
                # Try swapping
                temp_A = current_A[:]
                temp_A[i] = cand
                temp_score = score(temp_A)
                
                if temp_score < best_new_score:
                    best_new_score = temp_score
                    best_new_val = cand
            
            if best_new_val is not None and best_new_val != old_val:
                current_set.remove(old_val)
                current_A[i] = best_new_val
                current_set.add(best_new_val)
                current_score = best_new_score
                improved = True
        
        if not improved:
            break
    
    if current_score < best_score:
        best_score = current_score
        best_A = current_A[:]
    
    return best_A
