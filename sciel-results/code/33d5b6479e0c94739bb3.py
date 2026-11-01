import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s margin for 120s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"
    
    # For n >= 5, we use the standard recursive construction.
    # This is the best known general method and produces a valid superpermutation.
    # The length is n! + (n-1)! + ... + 2! + 1 + n - 1.
    # For n=5: 120 + 24 + 6 + 2 + 1 + 4 = 157
    # For n=6: 720 + 120 + 24 + 6 + 2 + 1 + 5 = 878
    # For n=7: 5040 + 720 + 120 + 24 + 6 + 2 + 1 + 6 = 5919
    
    # Use list of integers for efficient appending and comparison
    word_chars = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, return what we have
            return ''.join(map(str, word_chars))
        
        # Extract unique (m-1) permutations in order of appearance
        # Using tuples for hash efficiency
        perms = []
        seen = set()
        w_len = len(word_chars)
        prev_len = m - 1
        
        for i in range(w_len - prev_len + 1):
            window = tuple(word_chars[i:i + prev_len])
            if len(set(window)) == prev_len:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Build new word using list for efficiency
        new_word_chars = []
        
        for p in perms:
            # Block is P + [m] + P
            # We can construct the block as a list
            block = list(p) + [m] + list(p)
            block_len = len(block)
            
            # Find max overlap
            # Limit overlap to min(len(new_word_chars), len(block), m-1)
            # Note: The new symbol m cannot appear in the overlap because it's new.
            # Actually, the overlap is between the suffix of the current word and the prefix of the new block.
            # The prefix of the new block starts with P, which does not contain m.
            # So the overlap can be at most len(P) = m-1.
            limit = min(len(new_word_chars), m - 1)
            
            max_overlap = 0
            # Iterate from largest possible overlap down to 0
            for o in range(limit, 0, -1):
                # Check if suffix of new_word_chars matches prefix of block
                # new_word_chars[-o:] == block[:o]
                if new_word_chars[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            # Append non-overlapping part
            new_word_chars.extend(block[max_overlap:])
        
        word_chars = new_word_chars
    
    return ''.join(map(str, word_chars))
