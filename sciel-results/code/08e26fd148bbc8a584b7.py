import time
import itertools
import random

def construct(n):
    """
    Construct the shortest superpermutation on n symbols.
    Uses recursive construction followed by greedy deletion with count-based validity check.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to be safe before 150s kill
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        # Extract all distinct (m-1)-permutations from current word
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximal overlap
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations and map to integer codes for speed
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-(n+1) encoding for windows. Digits are 1..n, so base n+1 is safe.
    # Actually, base 8 is fine for n<=7. Let's use base 8 for simplicity and speed (bit shifts).
    # n is at most 7, so digits 1-7 fit in 3 bits.
    # Window length n.
    # Code = sum(digit * 8^i)
    
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
        
    # Map window code to permutation index
    code_to_pidx = {}
    for idx, p in enumerate(all_perms):
        c = 0
        for i in range(n):
            c += p[i] * pow8[n - 1 - i]
        code_to_pidx[c] = idx
        
    def window_code(w, start):
        c = 0
        for i in range(n):
            c += w[start + i] * pow8[n - 1 - i]
        return c
        
    # Initialize counts
    counts = [0] * num_perms
    
    # Count initial windows
    L = len(current_word)
    num_windows = L - n + 1
    # Pre-compute all window codes for the current state to avoid recomputation in inner loop
    # But we update counts on deletion, so we need to be careful.
    # For the initial setup:
    for i in range(num_windows):
        wc = window_code(current_word, i)
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1
            
    # Helper to check if position i can be deleted
    # We need to know the counts of permutations in windows affected by deleting i.
    # Affected windows are those starting at j where i-n+1 <= j <= i.
    
    # To make this fast, we maintain an array `wcodes` of current window codes.
    # When we delete a symbol, we shift the word, which changes all window codes.
    # This is O(L) per deletion, which is too slow for many deletions.
    # Alternative: Instead of updating the whole array, we can just re-compute the affected windows on the fly?
    # No, the window codes depend on the word content.
    # However, note that deleting a symbol at index i shifts all symbols after i to the left.
    # This changes the content of windows that overlap with position i.
    # Specifically, windows starting at j in [i-n+1, i] are affected.
    # Windows starting at j < i-n+1 are unaffected.
    # Windows starting at j > i are unaffected (their content is the same, just shifted index).
    # Wait, the *indices* of windows change, but the *content* of windows starting after i remains the same relative to the word?
    # No. If I delete word[i], then word[i+1] becomes word[i].
    # Window starting at j > i: its elements were word[j]...word[j+n-1]. Now they are word[j-1]...word[j+n-2].
    # So the content of *all* windows starting at j >= i changes?
    # Actually, window starting at j uses indices j..j+n-1.
    # If I delete index i:
    # - If j > i: the window now uses indices j-1..j+n-2. The content is different.
    # - If j < i-n+1: the window uses indices j..j+n-1. Since i >= j+n, the window is before the deletion. Content unchanged.
    # - If i-n+1 <= j <= i: the window overlaps the deletion. Content changes.
    
    # So effectively, deleting one symbol changes the content of O(n) windows directly, but shifts the indices of all subsequent windows.
    # This makes maintaining a global `wcodes` array very difficult to update efficiently without O(L) work.
    
    # Revised Strategy for Deletion:
    # Instead of maintaining a global array, we can just check validity on the fly using the current word.
    # Validity check for deleting index i:
    # For each window start j in [max(0, i-n+1), min(i, L-n)]:
    #   Compute window code from current_word
    #   Get pidx
    #   If counts[pidx] == 1, deletion is invalid.
    # This is O(n) work per check.
    # Total deletions ~ 100. Total checks ~ L * n.
    # L ~ 6000, n=7. 6000*7 = 42000 ops per pass.
    # Multiple passes (e.g., 20) -> 840,000 ops. This is very fast in Python.
    
    # Let's implement this simpler approach.
    
    # We need to be careful: `counts` must be accurate.
    # Initially, counts are accurate.
    # When we delete a symbol, we must update `counts` for the affected windows.
    # Affected windows: j in [i-n+1, i].
    # For each such j, the window content changes.
    # We need to decrement the count of the OLD window and increment the count of the NEW window?
    # No, the window at index j *disappears* or *changes*?
    # When we delete word[i], the list shrinks.
    # The window that was at index j (for j > i) is now at index j-1.
    # The window that was at index j (for j < i-n+1) is still at index j.
    # The windows at j in [i-n+1, i] are "mixed".
    
    # Actually, it's easier to just recompute the counts from scratch after each deletion?
    # O(L*n) to recompute counts.
    # If we do 100 deletions, 100 * 42000 = 4.2M ops. Still fast.
    # Let's try recomputing counts after each successful deletion.
    
    def recompute_counts(w):
        counts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            wc = window_code(w, i)
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1
        return counts
        
    # Initial counts
    counts = recompute_counts(current_word)
    
    # Deletion Loop
    # We will do multiple passes. In each pass, we try to delete as many as possible.
    # We alternate directions to avoid bias.
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
            
        L = len(current_word)
        if L <= n:
            break
            
        # Determine order of indices to try
        if pass_num % 2 == 0:
            indices = list(range(L))
        else:
            indices = list(range(L-1, -1, -1))
            
        # For randomization, we could shuffle, but sequential is often good enough for first few passes.
        # Let's add a random pass every 5th iteration.
        if pass_num % 5 == 4:
            random.shuffle(indices)
            
        made_change = False
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_word):
                continue
                
            # Check if we can delete current_word[i]
            # Affected windows are those that include index i.
            # Window start j must satisfy: j <= i <= j+n-1
            # => i-n+1 <= j <= i
            # Also j must be a valid window start: 0 <= j <= L-n
            
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            valid = True
            for j in range(start_j, end_j + 1):
                wc = window_code(current_word, j)
                pidx = code_to_pidx.get(wc, -1)
                if pidx >= 0:
                    if counts[pidx] == 1:
                        valid = False
                        break
                        
            if valid:
                # Perform deletion
                current_word.pop(i)
                # Recompute counts
                counts = recompute_counts(current_word)
                made_change = True
                # After deletion, indices shift. We can't just continue with the same list of indices.
                # We break the inner loop and restart the pass? Or adjust?
                # Breaking and restarting is safer and simpler.
                break
                
        if not made_change:
            # No more deletions possible in this order
            # Try a random pass?
            if pass_num > 0 and pass_num % 5 != 4:
                # If we haven't tried random recently, force a random pass
                indices = list(range(len(current_word)))
                random.shuffle(indices)
                made_change = False
                for i in indices:
                    if time.time() > deadline:
                        break
                    if i >= len(current_word):
                        continue
                    L = len(current_word)
                    start_j = max(0, i - n + 1)
                    end_j = min(i, L - n)
                    valid = True
                    for j in range(start_j, end_j + 1):
                        wc = window_code(current_word, j)
                        pidx = code_to_pidx.get(wc, -1)
                        if pidx >= 0:
                            if counts[pidx] == 1:
                                valid = False
                                break
                    if valid:
                        current_word.pop(i)
                        counts = recompute_counts(current_word)
                        made_change = True
                        break
                if not made_change:
                    break
            else:
                break
                
    return "".join(map(str, current_word))
