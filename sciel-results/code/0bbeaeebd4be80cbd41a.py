"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # Use the classic recursive construction which is fast and valid
    # For n=7, this gives length 1!+2!+...+7! = 5916
    # The shortest known is 5905, but finding it requires extensive search
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        # Use a strategy that maximizes overlap
        new = []
        
        for p in perms:
            # Create the block: p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with current end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
