import itertools

def construct(n):
    if n == 1:
        return "1"
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    seen = set()
    
    word = list(range(1, n + 1))
    seen.add(tuple(word))
    
    # Last position each symbol appeared
    last_pos = {i: 0 for i in range(1, n + 1)}
    for i, s in enumerate(word):
        last_pos[s] = i
    
    # Maximum steps to prevent infinite loop
    max_steps = n * total_perms * 2
    
    for _ in range(max_steps):
        if len(seen) == total_perms:
            break
        
        current_len = len(word)
        best_sym = None
        best_key = None
        
        for sym in range(1, n + 1):
            # Form the candidate window
            new_window = tuple(word[-(n-1):] + [sym])
            
            is_new_perm = new_window in all_perms and new_window not in seen
            
            # LRU: lower last_pos means less recently used (better)
            lru_priority = -last_pos[sym]
            
            # Key: prefer new perms, then LRU
            if is_new_perm:
                key = (0, lru_priority)
            else:
                key = (1, lru_priority)
            
            if best_key is None or key < best_key:
                best_key = key
                best_sym = sym
        
        word.append(best_sym)
        last_pos[best_sym] = len(word) - 1
        
        if len(word) >= n:
            w = tuple(word[-n:])
            if w in all_perms:
                seen.add(w)
    
    return "".join(map(str, word))
