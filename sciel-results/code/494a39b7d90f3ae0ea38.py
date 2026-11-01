import time
import random
from itertools import permutations

def construct(n):
    # Standard recursive construction
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
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n < 5:
        return "".join(map(str, word))
        
    # Precompute permutation indices for fast validity check
    # Map each permutation tuple to an integer 0..n!-1
    perm_map = {}
    idx = 0
    for p in permutations(range(1, n + 1)):
        perm_map[p] = idx
        idx += 1
    target_count = idx
        
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            # Tuple creation is still needed for lookup
            # We can optimize by encoding the window directly if we maintain state, 
            # but for a simple check, tuple lookup is fine for ~6000 windows.
            window = tuple(w[i:i + n])
            # If window is not a permutation, it's not in perm_map
            if window in perm_map:
                seen.add(perm_map[window])
            if len(seen) == target_count:
                return True
        return len(seen) == target_count

    if not is_valid(word):
        return "".join(map(str, word))
        
    best_word = word
    best_len = len(word)
    
    start_time = time.time()
    deadline = start_time + 115
    
    current_word = best_word[:]
    
    while time.time() < deadline:
        move_type = random.random()
        if move_type < 0.8:
            if len(current_word) <= n:
                continue
            idx_del = random.randint(0, len(current_word) - 1)
            candidate = current_word[:]
            candidate.pop(idx_del)
        else:
            if len(current_word) < 2:
                continue
            i, j = random.sample(range(len(current_word)), 2)
            candidate = current_word[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
        if is_valid(candidate):
            current_word = candidate
            if len(current_word) < best_len:
                best_len = len(current_word)
                best_word = current_word[:]
                if n == 7 and best_len <= 5905:
                    break
        else:
            pass
            
    return "".join(map(str, best_word))
