import time
import itertools

def construct(n):
    """
    Constructs the shortest known superpermutation using the classic recursive method.
    Length is sum_{k=1}^n k!.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Handle small n with hardcoded optimal solutions where possible
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        # Shortest is length 9: 123121321
        return "123121321"
    if n == 4:
        # Shortest is length 33
        # Using recursive construction
        pass
    
    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Find all distinct permutations of size m-1 in the current word, in order of appearance
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
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
