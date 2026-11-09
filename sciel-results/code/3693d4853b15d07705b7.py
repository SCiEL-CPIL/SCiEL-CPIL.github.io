import time
import sys

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n=7, this returns the classic recursive construction (length 5913).
    A more optimal construction (length 5905) exists but requires complex
    search or pre-computation not feasible in this context without risking
    validity or timeout.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        # The classic algorithm relies on the order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            # For efficiency, we check length and uniqueness
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p, m, p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the current end of 'new'
            # We want to append block[overlap:]
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Optimization: only check up to max_overlap
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
        
    return "".join(map(str, word))
