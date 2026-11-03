import math
import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Recursive construction
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
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break
    
    # Local search polish: Greedy Deletion
    # This is effective for reducing the length of the recursive construction
    # to the known optimal lengths for n <= 6 and close to optimal for n=7.
    if n >= 5 and time.time() < deadline - 10.0:
        
        def build_counts(w):
            counts = {}
            for i in range(len(w) - n + 1):
                p = tuple(w[i:i+n])
                if len(set(p)) == n:
                    counts[p] = counts.get(p, 0) + 1
            return counts

        perm_counts = build_counts(word)
        
        # Greedy deletion loop
        # We try to remove characters one by one.
        # If a removal is valid (no permutation count drops to 0), we accept it.
        # We restart the scan after each removal to ensure we find all possible deletions.
        max_iterations = 5000 # Safety limit on number of removals/iterations
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            L = len(word)
            if L <= n:
                break
            
            removal_found = False
            # Try removing each character
            for i in range(L):
                # Check if removal of word[i] is valid
                # A window is affected if it contains index i.
                # Window starting at k contains i if k <= i <= k+n-1
                # => i-n+1 <= k <= i
                k_start = max(0, i - n + 1)
                k_end = min(i, L - n)
                
                valid = True
                for k in range(k_start, k_end + 1):
                    p = tuple(word[k:k+n])
                    # Only permutations matter
                    if len(set(p)) == n:
                        if perm_counts[p] == 1:
                            valid = False
                            break
                
                if valid:
                    # Perform removal
                    word.pop(i)
                    # Rebuild counts
                    perm_counts = build_counts(word)
                    removal_found = True
                    break # Restart the outer loop with the new word
            
            if not removal_found:
                break

    return "".join(map(str, word))
