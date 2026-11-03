import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Classical recursive construction
    # Length is sum_{i=1}^n i!
    # For n=7, this is 5913.
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Extract all unique permutations of length m-1 from the current word
        # that consist of distinct symbols.
        # These are the "nodes" we will expand.
        perms = []
        seen = set()
        # Optimization: Use a sliding window and set check
        # The current word length is sum_{i=1}^{m-1} i!
        # We scan for windows of length m-1
        
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if distinct
            # Since m is small (up to 7), len(set) is fast enough
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = []
        # The first permutation is already in the word, but the algorithm 
        # typically prepends or appends. 
        # The standard algorithm:
        # For each perm p in order of appearance:
        #   block = p + [m] + p
        #   Overlap block with new_word
        
        # Note: The first 'p' is effectively the start.
        # Let's follow the seed logic which worked.
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            # Max possible overlap is min(len(new_word), len(block))
            # But block starts with p. 
            # Actually, we can just check suffixes of new_word against prefixes of block.
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Optimization: 
            # The overlap must be such that new_word[-overlap:] == block[:overlap]
            # Since block starts with p (length m-1), and new_word ends with 
            # the previous block's end, which is the previous p.
            # Often the overlap is m-1.
            
            # We can check from max_overlap down to 0.
            # But for m=7, len(block) = 13.
            # This is very fast.
            
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
            
        word = new_word
        
    # If we broke early due to deadline, we might have an incomplete word.
    # But for n=7, the loop is very fast.
    # 1! + 2! + ... + 6! is small.
    # The m=7 step is the big one.
    # Number of perms is 6! = 720.
    # 720 iterations, each with small list ops.
    # This is well under 1 second.
    
    return "".join(map(str, word))
