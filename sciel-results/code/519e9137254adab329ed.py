import time
import sys

def construct(n):
    # Classic recursive construction
    # This generates a superpermutation of length 1! + 2! + ... + n!
    # For n=7, this is 5913.
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        # that appear as windows
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # A valid window must contain distinct elements
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p, m, p
            block = list(p) + [m] + list(p)
            # Find maximum overlap between the end of 'new' and start of 'block'
            # We check all possible overlap lengths from max down to 0
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
