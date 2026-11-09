import time
import random
from fractions import Fraction

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    best_indices = [0]  # Index 0 corresponds to denom 2
    best_sum = recips[0]
    
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin for 60s limit
    
    def is_valid(indices):
        return len(indices) > 0 and sum(recips[i] for i in indices) < 1

    # Helper to greedily fill a set
    def greedy_fill(start_set):
        current_set = set(start_set)
        current_sum = sum(recips[i] for i in current_set)
        if current_sum >= 1:
            # Repair: remove random items until valid
            while current_sum >= 1 and current_set:
                idx_to_remove = random.choice(list(current_set))
                current_sum -= recips[idx_to_remove]
                current_set.remove(idx_to_remove)
            if not current_set:
                return None, None
        
        # Try to add items
        # Sort indices by reciprocal descending? No, just iterate all and try to add if fits.
        # To make it stochastic, shuffle the order of potential additions
        potential_adds = [i for i in range(n) if i not in current_set]
        random.shuffle(potential_adds)
        
        for i in potential_adds:
            if current_sum + recips[i] < 1:
                current_sum += recips[i]
                current_set.add(i)
        
        return list(current_set), current_sum

    # 1. Deterministic Greedy (Largest Reciprocals First)
    # This is equivalent to trying to add 1/2, 1/3, 1/4... in that order
    det_indices, det_sum = greedy_fill([]) 
    # Note: greedy_fill shuffles, so it's not strictly deterministic greedy by size.
    # Let's do a specific greedy by size for the first baseline.
    sorted_indices = sorted(range(n), key=lambda i: recips[i], reverse=True)
    curr_set = set()
    curr_sum = Fraction(0)
    for i in sorted_indices:
        if curr_sum + recips[i] < 1:
            curr_sum += recips[i]
            curr_set.add(i)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_indices = list(curr_set)
        
    # 2. Random Restart Greedy
    while time.time() - start_time < time_limit:
        # Start with a small random set to allow flexibility
        # Or start empty
        start_set = set()
        # Sometimes start with a random large element to constrain space?
        # Let's just start empty and let greedy fill do its thing.
        # To diversify, we can randomly skip some large reciprocals initially?
        # The shuffle in greedy_fill handles the order.
        
        cand_indices, cand_sum = greedy_fill(start_set)
        if cand_indices is not None and cand_sum > best_sum:
            best_sum = cand_sum
            best_indices = cand_indices
            
        # 3. Local Improvement on Best
        # Try to improve best_indices
        curr_set = set(best_indices)
        curr_sum = best_sum
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try adding
            potential_adds = [i for i in range(n) if i not in curr_set]
            random.shuffle(potential_adds)
            for i in potential_adds:
                if curr_sum + recips[i] < 1:
                    curr_sum += recips[i]
                    curr_set.add(i)
                    improved = True
                    # Update best immediately if better
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_indices = list(curr_set)
            # Try swapping
            # This is more complex. Pick i in set, j not in set.
            # If curr_sum - recips[i] + recips[j] < 1 and > curr_sum, swap.
            # We can do a few random swap trials.
            for _ in range(50):
                if time.time() - start_time > time_limit: break
                if len(curr_set) == 0: break
                i = random.choice(list(curr_set))
                j = random.choice([k for k in range(n) if k not in curr_set])
                new_sum = curr_sum - recips[i] + recips[j]
                if new_sum < 1 and new_sum > curr_sum:
                    curr_sum = new_sum
                    curr_set.remove(i)
                    curr_set.add(j)
                    improved = True
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_indices = list(curr_set)

    return best_indices
