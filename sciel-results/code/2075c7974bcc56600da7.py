import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Classic Recursive Construction
    # Length is 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break

    # For n <= 6, the recursive construction is often optimal or close enough
    # and the overhead of complex local search might not be worth it or might risk timeout.
    # However, for n=7, we have plenty of time (150s) and the recursive construction
    # is 5905, while optimal is ~5884+. We can try to shave off characters.
    
    if n >= 5 and time.time() < deadline - 20.0:
        # Build initial counts
        counts = defaultdict(int)
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                counts[p] += 1
        
        # Verify validity
        fact = 1
        for i in range(1, n+1):
            fact *= i
        if len(counts) < fact:
            return "".join(map(str, word))

        current_word = word
        
        def try_delete(i):
            """
            Attempts to delete current_word[i].
            Returns True if successful.
            Modifies current_word and counts if successful.
            """
            L = len(current_word)
            if L <= n:
                return False
            
            # 1. Identify old windows containing i
            # k in [max(0, i-n+1), min(i, L-n)]
            k_start_old = max(0, i - n + 1)
            k_end_old = min(i, L - n)
            
            old_perms = []
            for k in range(k_start_old, k_end_old + 1):
                p = tuple(current_word[k:k+n])
                if len(set(p)) == n:
                    old_perms.append(p)
            
            # 2. Identify new windows formed
            k_start_new = max(0, i - n + 1)
            k_end_new = min(i - 1, L - n - 1)
            
            new_perms = []
            for k in range(k_start_new, k_end_new + 1):
                w1 = current_word[k:i]
                w2 = current_word[i+1:k+n+1]
                p = tuple(w1 + w2)
                if len(set(p)) == n:
                    new_perms.append(p)
            
            # 3. Check validity without modifying counts yet
            # Build a temporary delta map
            delta = defaultdict(int)
            for p in old_perms:
                delta[p] -= 1
            for p in new_perms:
                delta[p] += 1
            
            valid = True
            for p, d in delta.items():
                if counts[p] + d <= 0:
                    valid = False
                    break
            
            if valid:
                # Commit deletion and update counts
                for p, d in delta.items():
                    counts[p] += d
                    if counts[p] == 0:
                        del counts[p]
                del current_word[i]
                return True
            else:
                return False

        # Main deletion loop with improved scan strategy
        # After deleting at position i, the affected region in the new word is
        # [max(0, i-n+1), min(i, len(current_word)-n)]
        # We resume scanning from the start of this affected region.
        
        i = 0
        while i < len(current_word):
            if time.time() > deadline - 5.0:
                break
            
            if try_delete(i):
                # Deletion successful. Resume from the start of the affected region.
                # In the new word, the affected region starts at max(0, i-n+1).
                i = max(0, i - n + 1)
            else:
                i += 1

        # Second pass: try again from the beginning to catch any new opportunities
        if time.time() < deadline - 5.0:
            i = 0
            while i < len(current_word):
                if time.time() > deadline - 5.0:
                    break
                
                if try_delete(i):
                    i = max(0, i - n + 1)
                else:
                    i += 1

    return "".join(map(str, current_word))
