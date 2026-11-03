import math
import time
import random

def _classic_superpermutation(n):
    """Generates the classic superpermutation of length n! + (n-1)! + ... + 1!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superpermutation(word, n):
    """Checks if word contains all permutations of 1..n"""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i+n]))
    return len(seen) == int(math.factorial(n))

def _greedy_delete(word, n, deadline):
    """
    Attempts to shorten the word by removing characters if possible.
    Returns the shortened word.
    """
    word = list(word)
    # Precompute permutation counts
    # We need to know the count of each permutation to check validity quickly
    # But since we are modifying the word, we need to update counts.
    # A permutation is a tuple of length n.
    
    # To make it fast, we can just re-check validity of the whole word
    # if the word is short enough, but for n=7, len ~ 6000, re-checking is O(N * n)
    # which is 6000 * 7 = 42k ops.
    # Doing this for 6000 deletions is 252 million ops.
    # In Python, this might be too slow (10-20 seconds per pass?).
    # Let's optimize the check.
    
    # Actually, we can maintain a set of seen permutations.
    # But when we delete, we lose some windows and gain none (just shift).
    # The windows that disappear are those that included the deleted element.
    # There are n-1 such windows.
    # If any of those windows is a permutation that appears ONLY once in the whole word,
    # then we cannot delete.
    
    # So, we need a frequency map of permutations.
    # freq[perm] = count
    
    # Initial frequency map
    freq = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    # We need to be careful: if we delete, the windows shift.
    # The windows that are "lost" are:
    # (word[i-n+1] ... word[i]) -> ends at i
    # (word[i] ... word[i+n-1]) -> starts at i
    # ...
    # Actually, deleting index i removes the window starting at i-n+1 ... i-1?
    # No.
    # Original: ... A B C D ...
    # Delete B.
    # New: ... A C D ...
    # Windows in Original containing B:
    # ... A B ...
    # ... B C ...
    # Windows in New:
    # ... A C ...
    # The windows ... A B ... and ... B C ... are gone.
    # New windows ... A C ... appear.
    
    # So, we must:
    # 1. Decrement freq for all windows in Original that contain index i.
    # 2. Increment freq for all windows in New that contain the "gap" (i.e., windows that cross the deletion point).
    # 3. Check if any freq drops to 0.
    
    # Windows containing i in Original:
    # Start indices: max(0, i-n+1) to min(i, len(word)-n)
    
    # Windows crossing the gap in New:
    # The gap is between new index i-1 and i.
    # Windows containing this gap in New are those that start at i-1, i-2, ...
    
    # This is doable but complex to implement correctly without bugs.
    # Given the 120s limit, a simpler approach might be safer:
    # Just try to delete, and if the word is small, re-verify.
    # For n=7, len=5900.
    # Re-verification is O(N * n) = 41k ops.
    # 5900 deletions * 41k = 242 million ops.
    # Python does ~10-50 million simple ops per second.
    # So one pass takes ~5-25 seconds.
    # We can afford 1-2 passes within 120s.
    
    # Let's implement the "Re-verify" approach for simplicity and correctness.
    
    current_word = list(word)
    improved = True
    while improved:
        improved = False
        # Try deleting from right to left to avoid index shifting issues
        for i in range(len(current_word) - 1, -1, -1):
            if time.time() > deadline:
                return current_word
            
            # Try deleting i
            # Create a new word without i
            # To avoid O(N) copy, we can just check validity on the fly?
            # No, we need to modify the list.
            
            # Optimization: Only try deletion if the character is "redundant".
            # A character is redundant if it is part of a permutation that appears multiple times.
            # This requires the freq map.
            
            # Let's stick to the simple re-verification but limit the number of attempts.
            # Or, use the freq map approach.
            
            # Let's try the freq map approach. It's faster.
            
            # Identify windows to remove
            remove_windows = []
            start_min = max(0, i - n + 1)
            start_max = min(i, len(current_word) - n)
            for s in range(start_min, start_max + 1):
                remove_windows.append(tuple(current_word[s:s+n]))
            
            # Check if any of these has freq 1
            can_delete = True
            for w in remove_windows:
                if freq.get(w, 0) == 1:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                # Update freq for removed windows
                for w in remove_windows:
                    freq[w] -= 1
                    if freq[w] == 0:
                        del freq[w]
                
                # Update freq for new windows
                # The new word is current_word[:i] + current_word[i+1:]
                # The new windows are those that cross the boundary i-1 | i
                # In the new word, the boundary is at index i-1 (0-indexed in new word? No)
                # New word length is len(current_word) - 1.
                # The gap is between new index i-1 and i.
                # Windows crossing the gap start at i-1, i-2, ...
                
                new_word = current_word[:i] + current_word[i+1:]
                # New windows crossing the gap:
                # They start at indices j in new_word such that j < i and j+n > i
                # j < i and j > i - n
                # So j in [max(0, i-n+1), i-1]
                
                # Wait, the indices in new_word are shifted.
                # The elements before i are the same.
                # The elements after i are shifted left by 1.
                
                # Windows in new_word that are "new" (didn't exist in old_word)
                # are those that contain the connection between old[i-1] and old[i+1].
                
                # Let's just recalculate the freq for the affected region?
                # No, let's just update.
                
                # The windows that are added are those starting at:
                # s in new_word such that the window includes new_word[i-1] and new_word[i]
                # i.e., s <= i-1 and s+n-1 >= i
                # s <= i-1 and s >= i-n+1
                
                add_windows = []
                for s in range(max(0, i - n + 1), i):
                    # Ensure s is a valid start index in new_word
                    if s + n <= len(new_word):
                        add_windows.append(tuple(new_word[s:s+n]))
                
                for w in add_windows:
                    freq[w] = freq.get(w, 0) + 1
                
                # Update current_word
                current_word = new_word
                improved = True
            # If not can_delete, do nothing
    return current_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if n <= 5:
        return "".join(map(str, _classic_superpermutation(n)))
    
    # Start with classic
    word = _classic_superpermutation(n)
    
    # If n=6, classic is optimal (873). No need to search.
    if n == 6:
        return "".join(map(str, word))
    
    # For n=7, try to shorten
    # We can try multiple passes or randomizations
    best_word = word
    
    # Run greedy deletion
    # To avoid getting stuck in a local optimum, we can try deleting in different orders
    # 1. Right to Left
    # 2. Left to Right
    # 3. Random
    
    # Pass 1: Right to Left
    w1 = _greedy_delete(best_word, n, deadline)
    if len(w1) < len(best_word):
        best_word = w1
    
    # Pass 2: Left to Right (on the result of Pass 1)
    # Note: _greedy_delete modifies the list in place or returns a new one?
    # My implementation returns a new list.
    # But it uses a freq map initialized from the input word.
    # So we need to pass the new word.
    
    if time.time() < deadline:
        w2 = _greedy_delete(best_word, n, deadline)
        if len(w2) < len(best_word):
            best_word = w2
    
    # Pass 3: Randomized deletion order
    if time.time() < deadline:
        # We need a version of greedy delete that takes an order
        # Or just shuffle the indices?
        # My current _greedy_delete iterates 0..N or N..0.
        # Let's just do another pass, maybe it finds more.
        w3 = _greedy_delete(best_word, n, deadline)
        if len(w3) < len(best_word):
            best_word = w3
            
    return "".join(map(str, best_word))
