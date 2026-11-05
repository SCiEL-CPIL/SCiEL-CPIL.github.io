"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the classic recursive method.
    For n <= 4, this is optimal. For n=5,6 it is also optimal. For n=7, it yields 5913, 
    while the best known is 5905.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Base case
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Find all unique permutations of 1..m-1 in the current word
        perms, seen = [], set()
        # The word length is L. Windows of size m-1.
        # Note: In the classic construction, we extract permutations of size m-1 from the word of size m-1.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            if len(set(window)) == m - 1 and all(1 <= x <= m - 1 for x in window):
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Form the block: p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
