"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This yields a valid superpermutation.
    # For n=7, length is 5913.
    # For n=6, length is 873.
    # For n=5, length is 153.
    # For n=4, length is 33.
    
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            # If we timeout, we must return a valid string.
            # The current `word` is a valid superpermutation for m-1.
            # But we need one for n.
            # This branch is unlikely to be hit for n<=7 as it's very fast.
            # If hit, we return the best we have (which might be invalid for n).
            # To be safe, we just proceed fast.
            pass
        
        perms, seen = [], set()
        # Extract all (m-1)-permutations from the current word in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between the end of `new` and start of `block`
            # Optimization: The overlap is typically large.
            # We can limit the search range if we know the structure, 
            # but for n=7, linear scan is fine.
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            # In the classic construction, overlap is at least m-1 usually.
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
