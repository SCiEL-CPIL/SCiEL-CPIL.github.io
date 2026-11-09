"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n):
    """
    Construct a superpermutation of length 1!+2!+...+n! using the standard
    recursive construction (the 'classic' one). This is the only known
    deterministic construction that works for all n and finishes in time.
    
    The classic construction produces a word of length sum_{k=1}^n k!.
    For n=7 this is 5913, which is close to the known optimum of 5905.
    """
    # Classic recursive construction: word starts as [1], then for each m=2..n,
    # we take the order of (m-1)-permutations as they appear in the current word,
    # and for each permutation p, we insert the block p, m, p with maximum overlap.
    
    word = [1]
    for m in range(2, n + 1):
        # Find the order in which (m-1)-permutations appear in the current word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build the new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
