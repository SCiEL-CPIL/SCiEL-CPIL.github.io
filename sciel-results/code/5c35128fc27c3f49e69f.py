import time
import sys

def construct(n):
    """
    Constructs a superpermutation of length n! + (n-1)! + ... + 1!
    using the standard recursive method.
    """
    if n == 1:
        return "1"
    
    # For small n, we could hardcode optimal solutions if known, 
    # but the recursive method is optimal for n<=6 and near-optimal for n=7.
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        # Construct the new word by inserting m
        for p in perms:
            # Block is p, m, p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            # The overlap must be a suffix of new_word and prefix of block
            # Max possible overlap is len(block) or len(new_word), but logically
            # it's constrained by the structure.
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
    
    return "".join(map(str, word))
