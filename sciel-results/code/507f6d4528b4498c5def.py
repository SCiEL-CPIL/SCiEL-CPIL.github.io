import time
import math
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Target: contain all n! permutations of 1..n as consecutive substrings
    target_perms = math.factorial(n)
    
    # For small n, we can do exact search or known optimal values
    # But the task is specifically for n=7 (5040 perms), length ~5905
    
    # Step 1: Generate the classic recursive superpermutation
    # This produces length sum_{k=1}^n k! = 5913 for n=7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of new_word
            overlap = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Initial length for n=7 is 5913
    # Known shortest is 5905, so we need to remove 8 symbols
    
    # Step 2: Greedy deletion with efficient validation
    # We maintain a set of all permutations seen in the word
    # When we delete a symbol, only windows containing that position are affected
    
    def get_all_perms(w):
        """Get set of all length-n permutations in word w"""
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return seen
    
    # Initial set of permutations
    current_perms = get_all_perms(word)
    
    # Helper to check if a deletion is valid
    # When deleting position i, the affected windows are those starting from 
    # max(0, i-n+1) to min(i, len(w)-n)
    def try_delete(w, i, perms_set):
        """Try to delete position i from w, return (new_w, new_perms, success)"""
        if i < 0 or i >= len(w):
            return w, perms_set, False
        
        # Windows that contain position i are those starting at j where:
        # j <= i <= j+n-1, so j is in [i-n+1, i]
        # And j must be valid: 0 <= j <= len(w)-n
        
        # Remove old windows containing position i
        old_windows = []
        for j in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
            old_windows.append(tuple(w[j:j+n]))
        
        # Create new word
        new_w = w[:i] + w[i+1:]
        
        if len(new_w) < n:
            return w, perms_set, False
        
        # Add new windows formed by the shift
        # After deletion, windows that were starting at positions > i shift left
        # New windows are those that now exist and weren't there before
        # Specifically, windows starting at positions j where the window now includes 
        # the element that was at position i+1 (shifted to i)
        # The new windows are those starting at j in [i-n+1, i] in the new word
        # But we need to be careful about boundaries
        
        new_windows = []
        for j in range(max(0, i - n + 1), min(i, len(new_w) - n) + 1):
            new_windows.append(tuple(new_w[j:j+n]))
        
        # Check if we still have all target_perms
        # We need to verify that no permutation is lost
        # A permutation is lost if it only appeared in one of the old_windows 
        # and is not in new_windows or other parts of the word
        
        # More efficient: check if the set of all perms still has size target_perms
        # This is O(len(w)) which is ~6000, acceptable for a few thousand deletions
        
        # Quick check: if any old window is unique and not in new windows, we might lose it
        # But this is complex. Let's just do the full check for now.
        
        # Full validation
        seen = set()
        for j in range(len(new_w) - n + 1):
            seen.add(tuple(new_w[j:j+n]))
        
        if len(seen) == target_perms:
            return new_w, seen, True
        else:
            return w, perms_set, False
    
    # Multiple passes of greedy deletion
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Try deletions from right to left (or random order for diversity)
        # Right-to-left is often better because it doesn't shift indices
        i = len(word) - 1
        while i >= 0 and time.time() < deadline:
            new_word, new_perms, success = try_delete(word, i, current_perms)
            if success:
                word = new_word
                current_perms = new_perms
                improved = True
                # Don't decrement i, check same position again (new element shifted in)
            else:
                i -= 1
        
        # Also try left-to-right
        if time.time() < deadline:
            i = 0
            while i < len(word) and time.time() < deadline:
                new_word, new_perms, success = try_delete(word, i, current_perms)
                if success:
                    word = new_word
                    current_perms = new_perms
                    improved = True
                    # Don't increment i
                else:
                    i += 1
    
    # Step 3: Local search - try swapping adjacent symbols
    # This can sometimes allow further deletions
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(word) - 1):
                if time.time() > deadline:
                    break
                # Try swapping word[i] and word[i+1]
                if word[i] != word[i+1]:
                    word[i], word[i+1] = word[i+1], word[i]
                    # Check validity
                    seen = set()
                    for j in range(len(word) - n + 1):
                        seen.add(tuple(word[j:j+n]))
                    if len(seen) == target_perms:
                        improved = True
                        # Keep the swap
                    else:
                        # Revert
                        word[i], word[i+1] = word[i+1], word[i]
    
    # Step 4: Another round of deletions after swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            i = len(word) - 1
            while i >= 0 and time.time() < deadline:
                new_word, new_perms, success = try_delete(word, i, current_perms)
                if success:
                    word = new_word
                    current_perms = new_perms
                    improved = True
                else:
                    i -= 1
    
    # Final validation
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i+n]))
    
    # If invalid, return the original classic construction
    if len(seen) != target_perms:
        # Regenerate classic
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            word = new_word
    
    return "".join(map(str, word))
