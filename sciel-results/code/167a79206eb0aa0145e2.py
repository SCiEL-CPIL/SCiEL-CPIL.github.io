"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is the standard "Teodorescu" construction (though he found shorter ones for larger n,
    # this is the base construction often referenced).
    # For n=7, this yields length 5913.
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # Return current best if we run out of time (unlikely for this algo)
            break
            
        perms = []
        seen = set()
        
        # Extract all unique permutations of length m-1 from the current word
        # The current word contains all permutations of length m-1.
        # We iterate through windows of size m-1.
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        # Construct the new word
        # For each permutation p, we append p + [m] + p, overlapping with the end of the current new word
        new_word = []
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
            
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
