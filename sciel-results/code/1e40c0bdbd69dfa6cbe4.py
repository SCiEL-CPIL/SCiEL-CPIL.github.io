import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    best_set = []
    
    # Helper: check if adding 'candidate' to 'current_set' keeps all triple sums unique
    # We maintain a set of all existing triple sums for O(1) lookup
    def try_add(candidate, current_set, sum_set):
        # Generate all new triple sums involving 'candidate'
        # Types:
        # 1. (candidate, candidate, candidate) -> 3*candidate
        # 2. (candidate, candidate, x) where x in current_set -> 2*candidate + x
        # 3. (candidate, x, y) where x, y in current_set, x <= y -> candidate + x + y
        
        # 1.
        s1 = 3 * candidate
        if s1 in sum_set:
            return False
        sum_set.add(s1)
        
        # 2.
        # If current_set is empty, no x
        if current_set:
            # Vectorized check for 2*candidate + x
            # We need to check if any 2*candidate + x is in sum_set
            # Since sum_set is a Python set, we can iterate or use numpy for speed if large
            # For k up to 80, k is small. Iteration is fine.
            for x in current_set:
                s2 = 2 * candidate + x
                if s2 in sum_set:
                    return False
                sum_set.add(s2)
        
        # 3.
        # Pairs (x, y) from current_set with replacement, x <= y
        if current_set:
            # Generate all pairs
            # For small k, combinations_with_replacement is fast enough
            from itertools import combinations_with_replacement
            for x, y in combinations_with_replacement(current_set, 2):
                s3 = candidate + x + y
                if s3 in sum_set:
                    return False
                sum_set.add(s3)
                
        return True

    def greedy_with_order(order, time_limit):
        """
        Greedily add numbers from 'order' to the set.
        Returns the largest valid set found.
        """
        current_set = []
        sum_set = set()
        
        for candidate in order:
            if time.time() > time_limit:
                break
            if try_add(candidate, current_set, sum_set):
                current_set.append(candidate)
                # Optional: stop if we hit max cardinality 80
                if len(current_set) >= 80:
                    break
        
        return current_set

    # Strategies for ordering candidates in [1, 4096]
    # 1. Standard ascending (baseline, but we know it's weak)
    # 2. Descending (large numbers first, might allow smaller gaps later? Or just different local optimum)
    # 3. Random shuffles (multiple restarts)
    # 4. "Sparse" start: pick a few random large numbers, then fill in? 
    #    Actually, the greedy nature means the first picks are critical.
    #    Let's try different random seeds and maybe different ranges.
    
    # To maximize cardinality, we want to pack as many numbers as possible.
    # The constraint is on triple sums. 
    # If we pick numbers that are far apart, we avoid collisions easily.
    # If we pick numbers close together, we risk collisions but might fit more in range?
    # Actually, to maximize count in a fixed range [1, N], we usually want to be as dense as possible,
    # but the "Sidon set" like properties for triples (B_3) force sparsity.
    # The maximum size of a B_3 set in [1, N] is roughly N^(1/3) * constant? 
    # Wait, the problem says "Maximize cardinality... up to 80".
    # For N=4096, 4096^(1/3) = 16. So 80 is much larger than N^(1/3).
    # This implies the "unique triple sum" constraint for *all* unordered triples with replacement
    # is actually much weaker than I thought, or my bound is wrong.
    # Let's re-read: "Every unordered triple, allowing repetitions, must have a different sum."
    # This is exactly the definition of a B_3 set (or Sidon set for k=3).
    # The size of a B_k set in [1, N] is O(N^(1/k)).
    # For k=3, N=4096, N^(1/3) = 16.
    # So a B_3 set of size 80 in [1, 4096] is IMPOSSIBLE if the definition is strict.
    # Let's check the seed program. It returns a list. The score is 1.0.
    # The seed program finds a set. Let's estimate its size.
    # The seed iterates 1..512. It likely finds a small set.
    # If the max size is ~16, then the "Maximize cardinality" target of 80 in the prompt might be a red herring or I am misinterpreting "unique triple sums".
    # "Every unordered triple... must have a different sum."
    # Triple (a,b,c) with a<=b<=c. Sum S = a+b+c.
    # If (a,b,c) != (a',b',c'), then S != S'.
    # This is the definition of a B_3 set.
    # Erdos-Turan: max size of B_k set in [1,N] is ~ c * N^(1/k).
    # For N=4096, k=3, size ~ 16.
    # Why does the prompt say "1..80"? Maybe it means the *output* can have up to 80, but we maximize what we find?
    # Or maybe "unique triple sums" allows for some collisions if the triples are not distinct? No, "Every unordered triple... must have a different sum" implies injectivity of the map from triples to sums.
    
    # If the max size is indeed ~16, then any valid set is small.
    # The seed program likely finds a set of size ~15-16.
    # The inspiration 1 (score 0.22) was worse.
    # The "improved island 0" strategy mentioned in the text: "Replaces O(N^3)... incremental O(N^2)... Measured: 1 -> 1.778".
    # This suggests the seed score was 1.0 (size ~15?) and the improved one got 1.778 (size ~26-27?).
    # If the score is proportional to size, and seed is 1.0, maybe seed size is 10?
    # If improved is 1.778, size is ~18?
    # Let's assume the goal is to find the largest possible B_3 set.
    
    # Strategy:
    # 1. Use the incremental check.
    # 2. Try multiple random orderings.
    # 3. Try "greedy with backtracking"? Too slow.
    # 4. Try "random restarts" with different initial seeds.
    
    # Let's run several random shuffles of the range [1, 4096].
    # Also try ranges [1, 2000], [1, 1000], etc., to see if smaller ranges allow denser packing?
    # No, larger range allows larger numbers, which don't help density, but we are constrained by [1, 4096].
    # Actually, using smaller numbers might lead to more collisions. Using larger numbers might be safer?
    # But we want to maximize count.
    # Let's just do random restarts over the full range.
    
    max_iter = 0
    while time.time() < deadline - 2.0:
        random.seed()
        # Shuffle the candidate range
        # To save time, maybe don't shuffle the whole 4096, but pick a subset?
        # No, greedy needs to consider all candidates in some order.
        # Shuffling 4096 items is fast.
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        # Run greedy
        result = greedy_with_order(candidates, deadline - 1.0)
        
        if len(result) > len(best_set):
            best_set = result
            # If we hit a high number, maybe stop?
            # But let's keep trying to see if we can beat it.
        
        # Also try a "descending" order occasionally
        if max_iter % 5 == 0:
            candidates_desc = list(range(4096, 0, -1))
            result_desc = greedy_with_order(candidates_desc, deadline - 1.0)
            if len(result_desc) > len(best_set):
                best_set = result_desc
        
        # Try a "sparse" random start: pick 5 random large numbers, then greedy fill?
        # The greedy_with_order handles this if the large numbers are early in the shuffle.
        
        max_iter += 1
        
        # If we've found a set of size > 20, we might be close to optimal.
        # But let's use the time.
        
    return best_set
