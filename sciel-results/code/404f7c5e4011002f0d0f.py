import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if time.time() > deadline:
        return "".join(map(str, word))
        
    # 2. Calculate frequency of all permutations in the word
    # Since n is small (<=7), we can store perms as tuples
    freq = {}
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    # 3. Trim from ends
    # We need to be careful: if we trim from start, the "end" indices shift?
    # No, we are just slicing the list.
    # But the freq dict is for the ORIGINAL word.
    # If we trim from start, the windows in the new word are a subset of the original windows.
    # Specifically, if we remove the first symbol, the window $W[0:n]$ is removed.
    # The window $W[1:n+1]$ becomes the new first window.
    # The freq of $W[0:n]$ decreases by 1.
    # The freq of all other windows remains the same?
    # Yes, because the other windows are unchanged.
    
    # So we can use the same logic.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try trimming from start
        if len(word) > n:
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
                changed = True
                
        # Try trimming from end
        if len(word) > n:
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
                changed = True
                
    return "".join(map(str, word))
