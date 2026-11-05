import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    
    for m in range(2, n + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Extract permutations of m-1 symbols from the current word
        perms = []
        seen = set()
        # The previous word contains all permutations of m-1 symbols
        # We iterate through all windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # A valid permutation has m-1 distinct symbols
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # The block is p, m, p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            # Overlap can be at most len(block) or len(new_word)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            
            # Append the non-overlapping part
            new_word.extend(block[overlap:])
        
        word = new_word
        
    return "".join(map(str, word))
