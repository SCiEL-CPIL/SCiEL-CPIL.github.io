import time
import itertools
from collections import Counter

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # Precompute all valid permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
        
    # Greedy deletion with fast updates
    # We will try to delete characters.
    # To avoid getting stuck, we can try multiple passes or randomize order.
    # But first, let's do a simple left-to-right pass.
    
    def try_delete(word, i, perm_counts, all_perms, n):
        """
        Checks if deleting word[i] is safe.
        Returns (is_safe, new_word, new_counts) if safe, else (False, None, None).
        """
        # Windows affected: start s in [i-n+1, i]
        s_min = max(0, i - n + 1)
        s_max = min(i, len(word) - n)
        
        if s_min > s_max:
            # No windows contain i? This shouldn't happen if i is within bounds and n>=1
            # But if i is at the very end and n is large?
            # If i > len(word)-n, then windows starting at i don't exist.
            # The windows containing i are those s such that s <= i <= s+n-1.
            # s >= i-n+1 and s <= i.
            # Also s must be a valid start index: 0 <= s <= len(word)-n.
            # So if s_min > s_max, no windows contain i.
            # This implies i is "outside" the range of any window?
            # If i is the last char, and n=7, len=10.
            # Windows start 0..3.
            # i=9. s in [3, 9] intersect [0,3] -> [3,3].
            # So s_min=3, s_max=3.
            # If i=0, s in [-6, 0] intersect [0, len-n] -> [0,0].
            # So it should always find at least one window if n >= 1 and len >= n.
            pass

        # Check safety
        for s in range(s_min, s_max + 1):
            window = tuple(word[s:s + n])
            if window in all_perms:
                if perm_counts[window] == 1:
                    return False, None, None
        
        # Safe to delete
        # Perform deletion
        new_word = word[:i] + word[i+1:]
        
        # Update counts
        # Decrement old windows
        for s in range(s_min, s_max + 1):
            window = tuple(word[s:s + n])
            perm_counts[window] -= 1
        
        # Increment new windows
        # New windows start at s in [i-n+1, i-1]
        # But limited to valid start indices in new_word
        # new_len = len(new_word)
        # valid starts: 0 to new_len - n
        new_s_min = max(0, i - n + 1)
        new_s_max = min(i - 1, len(new_word) - n)
        
        for s in range(new_s_min, new_s_max + 1):
            # The window in new_word at s
            # It uses new_word[s : s+n]
            window = tuple(new_word[s:s + n])
            perm_counts[window] += 1
            
        return True, new_word, perm_counts

    changed = True
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        for i in range(len(word)):
            if time.time() - start_time > deadline - 5:
                break
            
            safe, new_word, new_counts = try_delete(word, i, perm_counts, all_perms, n)
            if safe:
                word = new_word
                perm_counts = new_counts
                changed = True
                break  # Restart scan
    
    return "".join(map(str, word))
