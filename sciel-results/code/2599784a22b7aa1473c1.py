import collections

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
            # Find maximal overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Prune the word
    # Convert to string for easier slicing? Or keep list.
    # List of ints is fine.
    
    # If n is small, we might not need pruning, but it's safe.
    if n <= 6:
        return "".join(map(str, word))
        
    # For n >= 7, apply pruning
    # Initialize counts
    # Note: For n=7, 5040 permutations.
    # We only care about permutations of 1..n.
    # The recursive word contains all of them.
    
    # To speed up, we can just check if the window is a permutation of 1..n.
    # A window is a permutation if it has length n and all elements are distinct and in 1..n.
    
    target_len = n
    valid_perms = set()
    # Pre-compute all valid permutations? 
    # For n=7, 5040. 
    # Checking `len(set(w))==n and all(1<=x<=n for x in w)` is fast.
    
    # We maintain a Counter of all windows in the current word.
    # But we only need to track counts for windows that are permutations.
    # Actually, if a window is not a permutation, its count doesn't matter for validity.
    # But removing a char might turn a non-permutation window into a permutation? 
    # No, removing a char shrinks the word. It doesn't change the content of other windows.
    # It only destroys windows.
    # So we only need to track counts for windows that ARE permutations.
    
    # Initialize counts
    counts = collections.Counter()
    current_word = word[:]
    L = len(current_word)
    
    # Function to update counts for a specific range of start indices?
    # Or just rebuild?
    # Rebuilding counts for L=6000, n=7 is 6000 checks.
    # 6000 * 7 operations = 42k. Very fast.
    # We can rebuild counts after each removal?
    # If we remove 8 chars, we rebuild 8 times.
    # Total cost 8 * 42k = 336k operations.
    # Plus the check for each index.
    
    # Algorithm:
    # While improvements are made:
    #   changed = False
    #   counts = compute_counts(current_word)
    #   for i in range(len(current_word)):
    #      if is_removable(i, current_word, counts):
    #          remove i
    #          changed = True
    #          # Note: counts are now stale for indices > i.
    #          # But we are iterating left to right.
    #          # If we remove i, the indices shift.
    #          # It's safer to just do one pass, and repeat the whole pass.
    
    # Let's do a simple iterative pass.
    
    def compute_counts(w):
        c = collections.Counter()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check if permutation
            if len(set(window)) == n:
                c[window] += 1
        return c

    # We only care if count > 1 for removability.
    # Actually, if count == 1, we can't remove.
    # If count > 1, we can remove, provided no other window in the same removal set has count 1.
    
    # Optimization:
    # Instead of full rebuild, we can just check the specific windows.
    # But the "stale counts" issue is tricky.
    # Let's just do:
    # Pass 1:
    #   for i in range(L-1, -1, -1): # Right to left?
    #      Check windows containing i.
    #      If all counts > 1, remove i.
    #      (Counts for windows to the left are not affected by removal at i? 
    #       Windows to the left end before i. So they are not destroyed.
    #       Windows to the right start after i. They are not destroyed.
    #       Only windows overlapping i are destroyed.
    #       So if we go right-to-left, when we check i, the windows to the right (which overlap i) 
    #       are still present in the word.
    #       The counts we computed initially are valid for the current state of the word 
    #       IF we haven't removed anything to the right of i yet.
    #       Since we go right-to-left, we haven't removed anything to the right.
    #       So the initial counts are valid for the "right" part of the overlap?
    #       Wait. If we remove a char at $j > i$, it destroys windows overlapping $j$.
    #       Some of those windows might also overlap $i$?
    #       No, if $j > i + n - 1$, they don't overlap.
    #       If $j$ is close to $i$, they might.
    #       This makes incremental updating complex.
    
    # Given the small number of expected removals (maybe < 50), 
    # we can afford to recompute counts from scratch for each successful removal?
    # Or just do a full pass, remove all removable chars in that pass (based on initial counts of that pass), 
    # and then repeat the pass.
    
    # Let's do this:
    improved = True
    while improved:
        improved = False
        counts = compute_counts(current_word)
        L = len(current_word)
        # We need to be careful: if we remove index i, index i+1 becomes i.
        # So we should iterate from right to left to keep indices stable for the left part?
        # Or just mark indices to remove.
        
        to_remove = []
        for i in range(L - n + 1, -1, -1): # Right to left
            # Check windows containing i
            # s in [max(0, i-n+1), min(i, L-n)]
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            
            removable = True
            for s in range(s_min, s_max + 1):
                w = tuple(current_word[s:s+n])
                # Only check if it's a permutation
                # The counter only has permutations.
                # If w is not a permutation, it's not in counts, so count is 0?
                # But we only care about permutations.
                # If w is a permutation, and counts[w] <= 1, we cannot remove.
                # Wait, if w is NOT a permutation, removing it doesn't affect the "superpermutation" property
                # directly, but it might affect other windows? No.
                # The property is "contains every permutation".
                # Non-permutation windows are irrelevant to the validity check.
                # So we only check if `w` is a permutation.
                if len(set(w)) == n:
                    if counts[w] <= 1:
                        removable = False
                        break
            
            if removable:
                to_remove.append(i)
                improved = True
                
        if to_remove:
            # Remove in reverse order to keep indices valid
            for i in reversed(to_remove):
                del current_word[i]
                
    return "".join(map(str, current_word))
