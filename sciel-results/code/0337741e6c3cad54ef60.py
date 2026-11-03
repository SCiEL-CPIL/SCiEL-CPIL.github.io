import time
import itertools

def construct(n):
    """
    Constructs a superpermutation using the recursive method.
    Hardcodes small n to ensure optimal base cases.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate S_4, S_5, S_6 recursively from S_3
    word = list("123121321")
    
    for m in range(4, n + 1):
        if time.time() > deadline:
            break
        
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = list(word)
        for p in perms:
            block = list(p) + [m] + list(p)
            
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
