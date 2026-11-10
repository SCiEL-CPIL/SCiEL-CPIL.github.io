import time
import random
from itertools import combinations

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all 2^k subset products are distinct.
    
    Strategy:
    1. Precompute prime factorization vectors for all numbers in [2, 40].
    2. Use a fast vector-sum collision check (equivalent to product distinctness due to unique factorization).
    3. Perform multiple random restarts with greedy construction.
    4. Use a time budget to allow many restarts and potentially find a larger set.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for safety
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute prime factorization vectors for all numbers in 2..40
    vectors = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = tuple(v)
        
    candidates = list(range(2, 41))
    
    best_set = []
    best_len = 0
    
    # Helper function to check if adding a candidate vector to the current sum set causes collision
    def can_add(v_new, current_sums):
        # We need to check if any s + v_new is already in current_sums
        # This is equivalent to checking if v_new is in {s1 - s2 | s1, s2 in current_sums}
        # But simpler: just iterate.
        # Optimization: If we maintain current_sums as a set of tuples, we can check membership.
        # The cost is O(|current_sums|).
        for s in current_sums:
            new_s = tuple(s[i] + v_new[i] for i in range(num_primes))
            if new_s in current_sums:
                return False
        return True

    def update_sums(current_sums, v_new):
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + v_new[i] for i in range(num_primes))
            new_sums.add(new_s)
        current_sums.update(new_sums)
        return current_sums

    # We run multiple restarts.
    # In each restart, we shuffle the candidates and greedily add them if possible.
    # To make it faster, we can sort candidates by a heuristic before shuffling?
    # Or just shuffle. Shuffling is good for escaping local optima.
    
    # We can also try a deterministic pass first to get a good baseline.
    # Let's do a few deterministic passes with different orderings first.
    
    # Ordering 1: Primes first, then composites
    # Ordering 2: Primes first, then composites sorted by value
    # Ordering 3: Random
    
    def run_greedy(order):
        current_set = []
        current_sums = {tuple([0]*num_primes)}
        
        for n in order:
            if time.time() > deadline:
                break
            v = vectors[n]
            if can_add(v, current_sums):
                current_set.append(n)
                update_sums(current_sums, v)
        return current_set

    # Try a few deterministic orders
    # 1. Primes first, then rest
    primes_list = [p for p in primes if p <= 40]
    composites_list = [n for n in range(2, 41) if n not in primes_list]
    
    # Order: Primes, then composites
    order1 = primes_list + composites_list
    set1 = run_greedy(order1)
    if len(set1) > best_len:
        best_len = len(set1)
        best_set = set1[:]
        
    # Order: Primes, then composites sorted by "independence" (fewer factors?)
    def complexity(n):
        v = vectors[n]
        return sum(v)
    composites_sorted = sorted(composites_list, key=complexity)
    order2 = primes_list + composites_sorted
    set2 = run_greedy(order2)
    if len(set2) > best_len:
        best_len = len(set2)
        best_set = set2[:]
        
    # Now random restarts
    # We want to maximize the number of restarts.
    # Each restart takes O(N * 2^k * D) roughly, where D is number of primes.
    # For k=10, 2^10=1024. 39 * 1024 * 12 ~ 480k ops. Very fast.
    # For k=12, 2^12=4096. 39 * 4096 * 12 ~ 1.9M ops. Still fast.
    # So we can do many restarts.
    
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_sums = {tuple([0]*num_primes)}
        
        for n in shuffled:
            if time.time() > deadline:
                break
            v = vectors[n]
            if can_add(v, current_sums):
                current_set.append(n)
                update_sums(current_sums, v)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            # If we found 18, we can stop early? 
            # The task says 1..18 distinct integers. 
            # Is 18 the max possible? 
            # With 12 primes, the vector space is 12D. 
            # We are looking for a linearly independent set over integers? 
            # No, it's about subset sums being distinct. 
            # This is equivalent to the vectors being "sum-free" in a certain sense? 
            # Actually, if we have a set of vectors V, the subset sums are distinct iff 
            # no non-empty subset sums to zero? No. 
            # Distinct subset sums means that for any two distinct subsets A, B, sum(A) != sum(B).
            # This is equivalent to sum(A \ B) != sum(B \ A) for disjoint sets.
            # This is a strong condition.
            
            # Let's just keep searching if we have time.
            # But if we hit 18, it's likely very hard to beat.
            # However, the prompt says "1..18", implying 18 is a target or max.
            # Let's assume 18 is the max possible or a good target.
            # If we find 18, we can break out of restarts to save time for safety?
            # No, the evaluator measures raw value. Higher is better.
            # If 18 is the max, finding it is optimal.
            # Let's check if we can find > 18. 
            # The range is 2..40. 
            # If we find 18, we can probably stop, as it's very dense.
            # But to be safe, let's just let the loop finish or timeout.
            
    # Fallback if no set found (shouldn't happen)
    if not best_set:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Ensure return type is list of ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
