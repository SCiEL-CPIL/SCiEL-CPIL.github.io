import time
import sys

def construct_standard(n):
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def is_perm_7(w):
    # w is a list or tuple of 7 ints
    if len(w) != 7:
        return False
    # Check distinct
    # Since inputs are 1..7, distinct check is sufficient
    return len(set(w)) == 7

def shorten_greedy(word):
    # word is a list of ints
    n = max(word)
    # Only apply if n is large enough for this to be useful/fast
    if n < 6:
        return word
        
    # Initialize counts
    counts = {}
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
            
    current_word = word[:]
    improved = True
    passes = 0
    max_passes = 50 # Safety limit
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        i = 0
        while i < len(current_word):
            # Check if removing current_word[i] is safe
            safe = True
            affected_indices = range(max(0, i - n + 1), min(i + 1, len(current_word) - n + 1))
            # We need to check windows that INCLUDE index i
            # A window starting at j includes i if j <= i < j+n
            # So j is in [i-n+1, i]
            # And j must be a valid start index: 0 <= j <= len-n
            
            for j in affected_indices:
                w = tuple(current_word[j:j+n])
                if len(set(w)) == n:
                    if counts.get(w, 0) <= 1:
                        safe = False
                        break
            
            if safe:
                # Remove current_word[i]
                del current_word[i]
                # Update counts
                # Re-evaluate affected windows
                # The indices shift.
                # It is safer to just recompute counts for the affected region or global?
                # Global recompute is O(L*n) = 6000*7 = 42000 ops.
                # If we do this for every removal, and we have ~100 removals, 4.2M ops. Fast.
                # But we can be smarter.
                
                # Let's just recompute counts for the whole word to be safe and simple.
                # Optimization: Only recompute if we actually removed something.
                new_counts = {}
                for k in range(len(current_word) - n + 1):
                    w = tuple(current_word[k:k+n])
                    if len(set(w)) == n:
                        new_counts[w] = new_counts.get(w, 0) + 1
                counts = new_counts
                improved = True
                # Do not increment i, as the next element is now at i
            else:
                i += 1
                
    return current_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Generate standard
    word = construct_standard(n)
    
    # Shorten
    word = shorten_greedy(word)
    
    # If we have time, try more passes or different heuristics?
    # The greedy pass is O(Passes * L * N * Cost_Update).
    # With L=6000, N=7, it's very fast.
    
    return "".join(map(str, word))
