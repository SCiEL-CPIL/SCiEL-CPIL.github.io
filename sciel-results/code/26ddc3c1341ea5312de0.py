"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    """
    Construct a superpermutation of length 1! + 2! + ... + n! with optimization
    at the final stage to approach the known optimum of 5905 for n=7.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    
    # Build up to n-1 using standard construction
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n == 2:
        return "".join(map(str, word))
    
    # For the final stage (adding symbol n), extract all (n-1)-permutations
    # from the current word and try to order them to minimize total length
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    if not perms:
        # Fallback: standard construction for this level
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        return "".join(map(str, word))
    
    # Standard order gives a baseline
    def build_with_order(order):
        new = []
        for p in order:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    # Try the standard order first
    baseline_word = build_with_order(perms)
    baseline_len = len(baseline_word)
    
    # For n=7, try to find a better ordering via greedy local search
    # The idea: swap adjacent permutations if it reduces total length
    if n <= 7 and time.time() < deadline - 5:
        best_order = list(perms)
        best_len = baseline_len
        improved = True
        max_iters = 50
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_order) - 1):
                # Try swapping i and i+1
                new_order = best_order[:]
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                candidate = build_with_order(new_order)
                if len(candidate) < best_len:
                    best_len = len(candidate)
                    best_order = new_order
                    improved = True
                    break  # Restart from beginning after improvement
        
        word = build_with_order(best_order)
    else:
        word = baseline_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
