import time
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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
    # However, for n=7, we have plenty of time (110s) and the recursive construction
    # is 5905, while optimal is ~5884+. We can try to shave off characters.
    
    if n >= 5 and time.time() < deadline - 10.0:
        # Build initial counts
        counts = defaultdict(int)
        unique_perms = set()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                counts[p] += 1
                if counts[p] == 1:
                    unique_perms.add(p)
        
        # Verify validity
        fact = 1
        for i in range(1, n+1):
            fact *= i
        if len(counts) < fact:
            return "".join(map(str, word))
        
        current_word = word
        L = len(current_word)
        
        def try_delete(i):
            """
            Attempts to delete current_word[i].
            Returns True if deletion was performed.
            """
            nonlocal current_word, L
            nonlocal counts, unique_perms
            
            if L <= n:
                return False
            
            # 1. Identify old windows containing i
            k_start_old = max(0, i - n + 1)
            k_end_old = min(i, L - n)
            
            old_perms = []
            for k in range(k_start_old, k_end_old + 1):
                p = tuple(current_word[k:k+n])
                # We only care if it's a valid permutation
                if len(set(p)) == n:
                    old_perms.append(p)
            
            # 2. Identify new windows formed
            # New windows starting at k < i that cross the gap
            new_perms = []
            k_start_new = max(0, i - n + 1)
            k_end_new = min(i - 1, L - n - 1)
            
            for k in range(k_start_new, k_end_new + 1):
                w1 = current_word[k:i]
                w2 = current_word[i+1:k+n+1]
                p = tuple(w1 + w2)
                if len(set(p)) == n:
                    new_perms.append(p)
            
            # 3. Check validity
            # A deletion is unsafe if it removes the last occurrence of any permutation.
            # i.e., if there exists p in old_perms such that:
            #   counts[p] == 1 AND p is NOT in new_perms
            
            # Optimization: Only check p in old_perms if it is in unique_perms
            for p in old_perms:
                if p in unique_perms:
                    # This permutation has count 1. We are deleting its only occurrence.
                    # We must ensure it appears in new_perms.
                    if p not in new_perms:
                        return False
            
            # 4. Commit deletion
            # Update counts
            for p in old_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
                elif counts[p] == 1:
                    unique_perms.add(p)
                
            for p in new_perms:
                if p in counts:
                    counts[p] += 1
                    if counts[p] == 2:
                        # Was 1, now 2. No longer unique.
                        if p in unique_perms:
                            unique_perms.remove(p)
            
            del current_word[i]
            L -= 1
            return True
        
        # Perform multiple passes of greedy deletion
        max_passes = 100
        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            
            i = 0
            made_deletion = False
            while i < L:
                if time.time() > deadline - 2.0:
                    break
                
                if try_delete(i):
                    made_deletion = True
                    # Do not increment i, as the next element shifted into position i
                    # might now be deletable.
                else:
                    i += 1
            
            if not made_deletion:
                break

    return "".join(map(str, current_word))
