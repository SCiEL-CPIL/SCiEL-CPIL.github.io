import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # If n is small, standard is optimal
    if n <= 5:
        return "".join(map(str, word))
        
    # If n is large, search is infeasible
    if n > 7:
        return "".join(map(str, word))
        
    # 2. Local Search for n=6, 7
    required_perms = set(itertools.permutations(range(1, n + 1)))
    num_perms = len(required_perms)
    
    def is_valid(w):
        if len(w) < n:
            return False
        found = set()
        for i in range(len(w) - n + 1):
            found.add(tuple(w[i:i+n]))
            if len(found) == num_perms:
                return True
        return len(found) == num_perms
        
    best_word = word
    best_len = len(word)
    
    # Greedy Deletion
    while time.time() < deadline:
        improved = False
        for i in range(len(best_word)):
            if time.time() > deadline:
                break
            
            new_word = best_word[:i] + best_word[i+1:]
            if is_valid(new_word):
                best_word = new_word
                best_len -= 1
                improved = True
                break # Restart from 0
        
        if not improved:
            break
            
    return "".join(map(str, best_word))
