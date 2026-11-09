import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added to current set
    def can_add(x, current_set, sum_counts):
        # Check sums x+y for y in current_set
        # and 2*x
        # If any sum has count 2, return False
        # We don't update sum_counts, just check
        for y in current_set:
            s = x + y
            if sum_counts.get(s, 0) == 2:
                return False
        s = 2 * x
        if sum_counts.get(s, 0) == 2:
            return False
        return True

    def update_counts(x, current_set, sum_counts):
        # Update sum_counts with new sums
        for y in current_set:
            s = x + y
            sum_counts[s] = sum_counts.get(s, 0) + 1
        s = 2 * x
        sum_counts[s] = sum_counts.get(s, 0) + 1

    # 1. Deterministic Greedy
    chosen = []
    counts = {}
    for x in range(1, 1001):
        if can_add(x, chosen, counts):
            chosen.append(x)
            update_counts(x, chosen[:-1], counts) # Note: update_counts uses current_set BEFORE adding x? 
            # My update_counts above takes current_set. 
            # If I call update_counts(x, chosen, counts) AFTER appending, it uses the new set.
            # Let's fix the helper.
            
    # Let's rewrite the helpers to be cleaner.
    
    def try_add(x, A, counts):
        # A is the current set (list)
        # counts is the dict of sum counts for A
        # Returns True if x can be added, and updates counts if so? 
        # No, better to check first.
        
        # Check
        for y in A:
            s = x + y
            if counts.get(s, 0) == 2:
                return False
        s = 2 * x
        if counts.get(s, 0) == 2:
            return False
        
        # Update
        for y in A:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        A.append(x)
        return True

    # Run Deterministic Greedy
    A_det = []
    counts_det = {}
    for x in range(1, 1001):
        try_add(x, A_det, counts_det)
        
    if len(A_det) > best_size:
        best_size = len(A_det)
        best_set = A_det[:]

    # Run Randomized Greedy
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        A_rand = []
        counts_rand = {}
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(x, A_rand, counts_rand):
                pass
        
        if len(A_rand) > best_size:
            best_size = len(A_rand)
            best_set = A_rand[:]
            
        # Optional: Try to improve by adding more elements in a second pass?
        # The random pass already tries all elements.
        
    return best_set
