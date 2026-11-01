import time
import random

def construct(n):
    start_time = time.time()
    # 150s limit. Use 135s for construction + search to be safe.
    deadline = start_time + 135.0
    
    # 1. Standard recursive construction (Seed)
    # This gives a valid superpermutation of length sum_{k=1}^n k!
    # For n=7, this is 1+2+6+24+120+720+5040 = 5913.
    # The known optimum is 5905.
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For small n, standard is optimal
    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization Phase
    # Goal: Remove redundant symbols to shorten the word while keeping all permutations.
    # A symbol at index i is redundant if removing it does not destroy any unique permutation window.
    # A permutation window is "unique" if it appears exactly once in the word.
    # If a window appears multiple times, we can afford to destroy one instance.
    # If a window appears exactly once, we MUST preserve that specific window.
    
    # Efficient validity check strategy:
    # Maintain a count of occurrences for each permutation.
    # When we remove a symbol, we update the counts for affected windows.
    # However, removal shifts indices, making incremental updates complex.
    # Given L ~ 6000, a full recount is O(L * n) = 6000 * 7 = 42000 ops.
    # In Python, 42000 ops takes ~ 1-2 ms.
    # We have 135 seconds. We can afford ~ 100,000 - 1,000,000 checks.
    # Let's aim for a few hundred thousand removal attempts.
    
    current_word = word
    current_len = len(current_word)
    
    # Pre-compute the set of all required permutations?
    # No, we just need to ensure that the set of windows in current_word covers all n! permutations.
    # Actually, the problem statement says "contains every one of the 5040 permutations".
    # So we need the set of windows to be the full set of permutations.
    
    # Helper to check validity and return the set of windows
    def get_windows(w):
        windows = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n: # Only valid permutations
                windows.add(window)
        return windows
        
    # For n=7, there are 5040 permutations.
    # We can check if len(windows) == 5040.
    # But generating the set of all 5040 perms takes time.
    # We can just check if the number of unique windows is 5040.
    # Is it possible to have 5040 unique windows that are NOT all permutations?
    # No, because the windows are permutations of 1..n. There are exactly n! such permutations.
    # So if we have n! unique windows, we have all of them.
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
        
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == target_count

    # Optimization Loop
    # Strategy:
    # 1. Try to remove symbols from the middle or random positions.
    # 2. If valid, keep the removal.
    # 3. Repeat until no more removals are possible or time runs out.
    
    # To speed up is_valid, we can use a faster check.
    # Instead of building a set of tuples, we can use a bitset or a boolean array if we map perms to indices.
    # But mapping 5040 perms to indices requires a dictionary lookup per window.
    # Let's stick to the set of tuples for simplicity, but optimize the loop.
    
    # Actually, for n=7, 5040 tuples in a set is small.
    # The bottleneck is the loop over ~6000 windows.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through indices in a randomized order to avoid local minima
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing current_word[i]
            # Optimization: Skip if removing this symbol breaks a "critical" window?
            # Hard to know without checking.
            
            new_word = current_word[:i] + current_word[i+1:]
            
            if is_valid(new_word):
                current_word = new_word
                improved = True
                # Continue searching for more removals
                # Note: The indices list is now stale, but we just broke out of the inner loop
                # and the while loop will restart with a new shuffled list.
                # This is inefficient. We should continue scanning the rest of the indices in the current list?
                # But the indices are shifted.
                # Better to break and restart the while loop.
                break
                
    return "".join(map(str, current_word))
