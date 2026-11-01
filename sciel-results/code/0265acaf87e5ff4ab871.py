import time
import itertools
from collections import deque

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n <= 1:
        return "1"

    # 1. Generate the classic recursive superpermutation
    # Length is 1! + 2! + ... + n!
    # For n=7, this is 1 + 2 + 6 + 24 + 120 + 720 + 5040 = 5913
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        # They appear in the order of the recursive construction
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Only consider windows that are permutations (all distinct)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # The block to insert is p, m, p
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of the current new word
            # The overlap must be a suffix of new and prefix of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Optimization: Greedy deletion
    # The classic construction is suboptimal. We can try to remove characters
    # one by one and check if all permutations are still present.
    
    # Pre-compute all target permutations for efficiency
    all_perms = set(itertools.permutations(range(1, n + 1)))
    target_count = len(all_perms)

    # Helper to check validity efficiently
    # Instead of rebuilding the set from scratch every time, we could maintain counts,
    # but for a single pass greedy deletion, rebuilding might be okay if n is small
    # and word is not huge. However, for n=7, word is ~5900.
    # Rebuilding set of 5900 tuples of length 7 is fast.
    # Checking issubset is fast.
    
    def is_valid(w):
        if len(w) < n:
            return False
        present = set()
        for i in range(len(w) - n + 1):
            present.add(tuple(w[i:i + n]))
        return all_perms.issubset(present)

    # Initial check (should be true)
    if not is_valid(word):
        return "".join(map(str, word))

    # Greedy deletion loop
    # We scan from left to right. If we can remove word[i], we do and restart from 0.
    # This is O(L^2 * N) roughly, but L is small enough.
    # 5913^2 * 7 operations ~ 240M ops. Might be too slow in Python.
    # Let's optimize the validity check or the search.
    
    # A faster validity check:
    # Maintain a counter of how many times each permutation appears.
    # If we remove a character at index i, the windows affected are i-n+1 to i.
    # Specifically, windows starting at i-n+1, ..., i are invalidated (removed).
    # New windows starting at i-n+1, ..., i-1 are shifted? No.
    # If we remove w[i], the word becomes w[:i] + w[i+1:].
    # The windows that disappear are those that included w[i].
    # These are windows starting at max(0, i-n+1) to min(i, L-n).
    # The windows that remain are unchanged except for their position, but their content is the same.
    # Wait, the content of windows *after* i shifts left, but the tuple content is the same.
    # So we only need to remove the tuples corresponding to the windows that contained w[i].
    # And we don't add any new windows because the length decreases by 1, so the number of windows decreases by 1.
    # Actually, the set of windows is a subset of the previous set minus the ones containing the removed char?
    # No.
    # Let W be the word. Windows are W[j:j+n].
    # Remove W[i]. New word W'.
    # Windows in W' are W'[j:j+n].
    # For j < i-n+1, W'[j:j+n] == W[j:j+n].
    # For j >= i-n+1 and j < i, W'[j:j+n] == W[j:j+n-1] + W[j+n]? No.
    # W' is W with one char removed.
    # W'[j] = W[j] for j < i.
    # W'[j] = W[j+1] for j >= i.
    # So W'[j:j+n] for j < i-n+1 is same as W[j:j+n].
    # For j in [i-n+1, i-1], the window in W' is composed of parts from W before and after i.
    # This is getting complicated to maintain incrementally.
    
    # Given the time limit, let's try a simpler approach:
    # Only attempt deletion if we have plenty of time.
    # And maybe limit the number of deletions or use a faster check.
    
    # Let's try to optimize the is_valid check by using a numpy array or similar?
    # Or just accept that for n=7, it might be slow and add a time check.
    
    # Alternative: Only try to delete if the character is "redundant".
    # A character is redundant if it is not part of any "essential" permutation.
    # This is hard to determine.
    
    # Let's stick to the simple greedy but with a time check inside the loop.
    # And maybe break early if we don't find a deletion in a while.
    
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        # Iterate through indices
        # To speed up, we can check from left to right.
        for i in range(len(word)):
            if time.time() > deadline:
                break
            # Try removing word[i]
            # Optimization: If word[i] is the only occurrence of some symbol in a window, 
            # and that window is the only occurrence of that permutation, it's essential.
            # Too complex.
            
            # Simple check:
            # Create a copy? No, slicing is O(N).
            # is_valid is O(N).
            # So each step is O(N). Total O(N^2).
            # 5900^2 = 35M. Python can do ~10-50M simple ops/sec.
            # So 35M tuple creations and set inserts might take 5-10 seconds per pass.
            # If we do a few passes, we might run out of time.
            
            # Let's try one pass first.
            # If we remove, restart.
            
            # Quick heuristic: if removing this char doesn't change the set of permutations?
            # We can check which permutations are "at risk".
            # The windows containing index i are i-n+1 to i.
            # If any of these windows is the *only* occurrence of that permutation, we cannot remove i.
            
            # Let's pre-compute counts for the current word.
            # This is expensive to update, but we can do it once per outer loop iteration?
            # No, the word changes.
            
            # Let's just try the deletion and validate.
            # But to save time, let's check if the word is already minimal?
            # No, we know it's not.
            
            # Let's try removing i.
            # We can avoid creating the full new word list if we just check validity on the fly?
            # No, is_valid needs the list.
            
            # Let's try to optimize is_valid.
            # Instead of set, use a dictionary of counts?
            # Or just set.
            
            # Let's try a different strategy:
            # Only try to delete if the character is '1' or something? No.
            
            # Let's just do it and hope it's fast enough for n<=6 or short n=7.
            # For n=7, it might be too slow.
            # The task says "at most n * n! symbols".
            # For n=7, 7*5040 = 35280.
            # Our word is 5913.
            
            # If time is tight, we should return the classic one.
            # Let's add a check: if n > 6, skip deletion?
            # Or limit deletions?
            
            # Let's try to make is_valid faster.
            # Using a set of tuples is the standard way.
            
            # Let's try to remove i.
            temp_word = word[:i] + word[i+1:]
            if is_valid(temp_word):
                word = temp_word
                changed = True
                break # Restart scan as indices changed
    
    return "".join(map(str, word))
