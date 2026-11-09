"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n):
    """
    Constructs a superpermutation of length 1! + 2! + ... + n!.
    This length is 5913 for n=7, which is 8 symbols longer than the optimal 5905.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if window is a permutation of 1..m-1
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between current end of 'new' and start of 'block'
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
