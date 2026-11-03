"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _construct_classic(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            # Find maximum overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
    return word

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def construct(n):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we want to finish by 145s
    deadline = start_time + 145.0
    
    # 1. Build standard superpermutation
    word = _construct_classic(n)
    
    # If n is large, the search space is too big for heuristic optimization within time limits
    # The classic construction is O(n * n!) which is huge for n >= 9.
    # For n <= 8, we can attempt local search.
    if n > 8:
        return "".join(map(str, word))
    
    # Convert to list of ints for efficient manipulation
    w = word
    L = len(w)
    target = _factorial(n)
    
    # 2. Setup permutation counting
    # Generate all permutations and map to index
    # For n=7, 5040 perms. This is fast.
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Initialize counts
    # counts[i] = number of times permutation i appears in word
    counts = [0] * target
    
    # Initial counts
    for i in range(L - n + 1):
        window = tuple(w[i:i + n])
        idx = perm_to_idx.get(window)
        if idx is not None:
            counts[idx] += 1
    
    # 3. Greedy Pruning Loop with improved efficiency
    # We repeatedly try to remove characters.
    # To be efficient, we scan the word and try to remove each char.
    # If a removal is valid, we do it and restart the scan from the beginning.
    
    current_word = w
    current_len = len(current_word)
    
    # To avoid infinite loops or getting stuck in a local minimum,
    # we can use a random restart or a randomized scan order.
    # However, simple greedy restart is often effective for superpermutations.
    
    # Optimization: Pre-compute the set of positions that are "critical"
    # i.e., part of a unique window? No, that's too expensive to maintain dynamically.
    
    # Instead, we do the standard O(L * n) check per removal candidate.
    # But we can break early in the check.
    
    # Let's add a small randomization to the scan order to escape local minima,
    # or just do multiple passes.
    
    # A more sophisticated approach:
    # Keep a list of "removable" positions.
    # But maintaining this list is complex.
    
    # Let's stick to the restart strategy but make the check faster.
    
    # We can use a bitmask to check if a window is a permutation quickly?
    # No, we need the specific permutation index to check counts.
    
    # Let's try to optimize the inner loop of the validity check.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle the order of positions to try?
        # If we always try from left to right, we might remove a char that prevents
        # a later removal which would have been beneficial.
        # But since we restart, it's a hill-climbing on length.
        
        # To improve escape from local minima, we can randomize the start position
        # or the order.
        
        # Let's try scanning from left to right, but if we fail, try right to left?
        # Or just randomize the order of positions to check.
        
        positions = list(range(1, current_len - 1))
        random.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline - 5.0:
                break
            
            # Check if removal is valid
            # Windows affected: starting at [pos-n+1, pos]
            start_min = max(0, pos - n + 1)
            start_max = min(pos, current_len - n)
            
            valid = True
            lost_indices = []
            
            for i in range(start_min, start_max + 1):
                window = tuple(current_word[i:i + n])
                idx = perm_to_idx.get(window)
                if idx is not None:
                    if counts[idx] == 1:
                        valid = False
                        break
                    else:
                        lost_indices.append(idx)
            
            if valid:
                # Perform removal
                for idx in lost_indices:
                    counts[idx] -= 1
                
                # Update word
                current_word = current_word[:pos] + current_word[pos+1:]
                current_len -= 1
                improved = True
                break # Restart scan with new word
    
    return "".join(map(str, current_word))

# EVOLVE-BLOCK-END
