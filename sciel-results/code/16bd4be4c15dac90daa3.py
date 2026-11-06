import time
import random
import sys
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    def is_valid_addition(x, selected, sums_set):
        """
        Check if x can be added to selected without creating duplicate triple sums.
        Returns True if valid, False otherwise.
        Also returns the new sums if valid? No, just boolean for simplicity, 
        we will recompute if we want to add, but that's expensive. 
        Better: return the set of new sums if valid, else None.
        """
        # Pre-compute new sums
        # 1. x + x + x
        new_sums = []
        
        # 1. 3x
        val = 3 * x
        if val in sums_set:
            return None
        new_sums.append(val)
        
        # 2. 2x + s for s in selected
        for s in selected:
            val = 2 * x + s
            if val in sums_set:
                return None
            new_sums.append(val)
            
        # 3. x + s1 + s2 for s1, s2 in selected (s1 <= s2)
        # We must also check for collisions within new_sums
        # To make this efficient, we can check against sums_set AND new_sums (as a set)
        # But creating a set for new_sums at every step is O(k^2).
        # Let's just check against sums_set first, then check internal collisions.
        
        # Optimization: 
        # The number of new sums is 1 + k + k(k+1)/2 = (k+1)(k+2)/2.
        # For k=30, this is ~465. 
        # Checking 465 items against a set of ~4000 is fast.
        # Checking internal collisions is O(M^2) if naive, or O(M log M) with sort.
        # M is small enough that a set is fine.
        
        # Let's collect all new sums and check for dups
        temp_new_sums = set()
        
        # Re-do the collection cleanly to avoid the "continue" logic mess
        # 1. 3x
        v = 3 * x
        if v in sums_set or v in temp_new_sums:
            return None
        temp_new_sums.add(v)
        
        # 2. 2x + s
        for s in selected:
            v = 2 * x + s
            if v in sums_set or v in temp_new_sums:
                return None
            temp_new_sums.add(v)
            
        # 3. x + s1 + s2
        # selected is sorted
        for i in range(len(selected)):
            s1 = selected[i]
            for j in range(i, len(selected)):
                s2 = selected[j]
                v = x + s1 + s2
                if v in sums_set or v in temp_new_sums:
                    return None
                temp_new_sums.add(v)
                
        return temp_new_sums

    def run_greedy(strategy="linear", random_seed=0):
        random.seed(random_seed)
        selected = []
        sums_set = set()
        
        if strategy == "linear":
            candidates = range(1, 4097)
        elif strategy == "random_shuffled":
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        elif strategy == "random_step":
            # Randomly skip some numbers? 
            # Or just shuffle. Shuffling is better for diversity.
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        else:
            candidates = range(1, 4097)
            
        for x in candidates:
            if time.time() > deadline:
                break
                
            new_sums = is_valid_addition(x, selected, sums_set)
            if new_sums is not None:
                selected.append(x)
                sums_set.update(new_sums)
                
        return selected

    # 1. Run standard linear greedy
    res = run_greedy("linear", 0)
    if len(res) > best_len:
        best_set = res
        best_len = len(res)
        
    # 2. Run randomized greedy multiple times
    # The linear greedy is often suboptimal for B_h sets because it greedily takes the smallest,
    # which might block better combinations later.
    # Randomized restarts can find larger sets.
    
    # How many restarts? 
    # Each run takes ~O(N * K^2). 
    # If K ~ 30, K^2 ~ 900. N=4096. 4096*900 ~ 3.7M ops.
    # In Python, 3.7M simple ops is ~0.1-0.2s.
    # We have 55s. We can afford ~100-200 restarts.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave 5s buffer
            break
        res = run_greedy("random_shuffled", i + 100)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            
    # 3. Try a "backtracking" or "swap" improvement on the best set found?
    # This is complex. A simpler heuristic: 
    # If we have a set S, can we replace a small element with a larger one to allow more elements?
    # Or just try to add more elements to the best set found?
    # The greedy already tries to add all possible elements in the chosen order.
    # If we run greedy on the *remaining* candidates sorted by some heuristic?
    
    # Let's try one more strategy: 
    # Start with the best set found, and try to add any remaining number that fits?
    # The greedy already did this for its specific order. 
    # But if we shuffle, we might miss a number that would have fit if we had skipped an earlier one.
    # The multiple restarts handle this by exploring different orders.
    
    # One specific improvement: 
    # Sometimes, removing a "bad" element allows adding two "good" ones.
    # This is a local search. 
    # Given time constraints, randomized restarts are usually sufficient for B_h sets 
    # to get close to the theoretical bound.
    
    return best_set
