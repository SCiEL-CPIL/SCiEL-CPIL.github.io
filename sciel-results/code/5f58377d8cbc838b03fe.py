"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the recursive method.
    Returns a string of digits 1..n.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 150  # seconds
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Extract all unique permutations of length m-1 from the current word
        # in order of appearance
        perms = []
        seen = set()
        # The window size is m-1
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            # A valid permutation must have distinct elements
            # Since word is built from permutations, this should hold, 
            # but checking len(set) is safe and cheap for small m.
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Construct the block: p, m, p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            # Overlap length k means new_word[-k:] == block[:k]
            max_possible_overlap = min(len(new_word), len(block))
            overlap = 0
            # We can optimize this by checking from largest possible overlap downwards
            # But len(block) is 2*(m-1)+1, which is small (max 13 for m=7)
            # So a linear scan or nested loop is fine.
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_word[-k:] == block[:k]:
                    overlap = k
                    break
            
            # Append the non-overlapping part
            new_word.extend(block[overlap:])
        
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
