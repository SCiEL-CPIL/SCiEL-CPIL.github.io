import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    powers = [3 ** p for p in range(n)]

    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls as numpy arrays for speed
    # Each ball has 13 words (the word itself + 6 positions * 2 alternative values)
    balls = np.empty((total_words, 13), dtype=np.int32)
    for i in range(total_words):
        d = [(i // powers[p]) % 3 for p in range(n)]
        balls[i, 0] = i
        k = 1
        for p in range(n):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    balls[i, k] = new_i
                    k += 1

    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_code = list(initial_code)
    best_size = len(best_code)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Phase 0: Greedy Set Cover to find a potentially smaller starting point
    # We use the numpy vectorized approach from inspiration 2
    uncovered = np.ones(total_words, dtype=np.int8)
    greedy_code = []
    
    # We have 115s. Let's allocate ~10s for greedy, rest for local search.
    greedy_deadline = start_time + 10.0
    
    while uncovered.sum() > 0:
        if time.time() > greedy_deadline:
            break
        
        # Find candidate that covers most uncovered words
        # counts[i] = number of uncovered words covered by candidate i
        counts = np.sum(uncovered[balls], axis=1)
        best_candidate = np.argmax(counts)
        
        if counts[best_candidate] == 0:
            break
            
        greedy_code.append(best_candidate)
        uncovered[balls[best_candidate]] = 0

    if len(greedy_code) < best_size:
        best_size = len(greedy_code)
        best_code = list(greedy_code)

    # If greedy didn't finish or didn't improve, we still have a valid covering if it finished.
    # But we need to ensure best_code is a valid covering.
    # The tetracode sum is always valid.
    # If greedy_code is not a valid covering (loop broke early due to time), we discard it.
    if uncovered.sum() > 0 and len(greedy_code) > 0:
        # Greedy failed to cover everything in time, revert to tetracode sum or keep best valid
        # Actually, if it broke due to time, it's not a valid covering.
        # We should only use greedy_code if it successfully covered everything.
        if len(greedy_code) < best_size and uncovered.sum() == 0:
            best_size = len(greedy_code)
            best_code = list(greedy_code)
        else:
            # Revert to tetracode sum if greedy failed
            if uncovered.sum() > 0:
                 best_code = list(initial_code)
                 best_size = len(best_code)

    # Now perform local search on the best valid code found so far
    # We will run multiple restarts of local search on the best_code
    
    # Helper functions for local search using numpy
    def compute_coverage_counts(code_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            counts[balls[c]] += 1
        return counts

    def can_remove(c, counts_arr):
        # A codeword c can be removed if all words it covers have count > 1
        return np.min(counts_arr[balls[c]]) > 1

    def try_remove_greedy(code_list, counts_arr):
        # Returns (new_code_list, new_counts_arr, removed_count)
        # We modify in place or return new lists. Let's return new lists to be safe with numpy views?
        # Actually, counts_arr is a numpy array. balls[c] is a view.
        # We need to be careful.
        removed = 0
        # We iterate through the code list. If we remove one, we update counts.
        # To avoid index issues, we iterate over a copy of the list indices?
        # Or we just iterate and remove.
        # Since we are checking min > 1, removing one reduces counts by 1.
        # If min was 2, it becomes 1. If min was 1, it was not removable.
        # So if removable, after removal, the min of its ball becomes 1 (or more).
        # We can just remove it.
        
        # To make it fast, we can try to remove in random order.
        indices = list(range(len(code_list)))
        random.shuffle(indices)
        
        new_code = []
        # We need to update counts as we go?
        # If we remove c1, counts change. This might make c2 removable which wasn't before?
        # No, removing a codeword only decreases counts. It can only make removal *harder* (min drops).
        # So if c2 was NOT removable (min <= 1), it remains not removable.
        # If c2 WAS removable (min > 1), it might become NOT removable if c1 and c2 shared a word with count 2.
        # So we must update counts.
        
        # However, a single pass is a standard heuristic.
        # Let's do a single pass.
        
        # We'll create a set of indices to remove? No, because removal of one affects others.
        # So we iterate.
        
        # Optimization: Instead of rebuilding the whole code list, we can mark for removal.
        # But we need to update counts.
        
        # Let's just do the simple loop.
        # We will create a new list.
        # But updating counts for each removal is O(13).
        
        # Let's try a different approach:
        # 1. Identify all candidates for removal based on current counts.
        # 2. Try to remove them one by one, updating counts.
        
        # Actually, the standard "greedy remove" is:
        # while True:
        #   found = False
        #   for c in code:
        #     if can_remove(c):
        #       remove(c)
        #       found = True
        #       break # restart or continue?
        #   if not found: break
        
        # Restarting is expensive. A single pass is faster but less effective.
        # Let's do a single pass with updates.
        
        # We iterate through the code.
        # If we remove c, we subtract its ball from counts.
        # We keep the code if it can't be removed.
        
        # But wait, if we keep c, and later remove d, c might become removable?
        # No, removing d decreases counts. If c was not removable (min <= 1), it stays not removable.
        # If c was removable (min > 1), it might become not removable.
        # So order matters.
        
        # Let's just do a while loop with a single pass check.
        
        changed = True
        while changed:
            changed = False
            # We need to check current state.
            # To avoid O(N^2) if we restart every time, we can just do one pass and see if any were removed.
            # If any were removed, do another pass.
            
            # This can be slow if many are removed.
            # Given N ~ 73, it's fine.
            
            for i in range(len(code_list)):
                c = code_list[i]
                if can_remove(c, counts_arr):
                    # Remove c
                    counts_arr[balls[c]] -= 1
                    code_list.pop(i)
                    changed = True
                    break # Restart the for loop / while loop to re-evaluate
        
        return code_list, counts_arr, changed

    # Main Local Search Loop
    current_code = list(best_code)
    current_counts = compute_coverage_counts(current_code)
    
    # We will try to improve best_code
    # We can do this by:
    # 1. Greedy removal on best_code
    # 2. Add-Remove local search
    
    # First, let's do a thorough greedy removal on the initial best_code
    temp_code = list(best_code)
    temp_counts = compute_coverage_counts(temp_code)
    try_remove_greedy(temp_code, temp_counts)
    
    if len(temp_code) < best_size:
        best_size = len(temp_code)
        best_code = list(temp_code)

    # Now, Add-Remove Local Search
    # We maintain current_code and current_counts.
    # We try to add a random word, then do greedy removal.
    # If we get smaller, update best.
    
    current_code = list(best_code)
    current_counts = compute_coverage_counts(current_code)
    current_set = set(current_code)
    
    # To speed up can_remove, we can precompute balls as lists? No, numpy is fine.
    # But accessing counts_arr[balls[c]] creates a temporary array.
    # np.min is fast.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Add a random word
        added = False
        for _ in range(10):
            rand_idx = random.randint(0, total_words - 1)
            if rand_idx not in current_set:
                current_code.append(rand_idx)
                current_counts[balls[rand_idx]] += 1
                current_set.add(rand_idx)
                added = True
                break
        
        if not added:
            # Try a different strategy: swap?
            # Remove a random word, add a random word?
            # Or just break if we can't add.
            # Actually, if we can't add, we are stuck in a local minimum for add-remove.
            # We can try to remove a word (if possible) to get a different state?
            # But greedy removal already did that.
            # So if we can't add, we should probably restart from best_code with a different random seed/order.
            # Let's just break and restart the loop with a fresh copy of best_code.
            break
        
        # Greedy Removal
        # We use the function defined above.
        # Note: try_remove_greedy modifies the list in place.
        # We pass current_code and current_counts.
        try_remove_greedy(current_code, current_counts)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
        # If we hit the lower bound (71) or known best (73), we can continue to try for 71.
        # But 71 is a lower bound, so 72 is possible?
        # The task says best known is 73, lower bound 71.
        # So 72 is a valid target.
        
        # To escape local minima, we can occasionally reset to best_code.
        if steps % 100 == 0:
            # Reset to best code to try a different path
            current_code = list(best_code)
            current_counts = compute_coverage_counts(current_code)
            current_set = set(current_code)

    return [to_word(idx) for idx in best_code]
