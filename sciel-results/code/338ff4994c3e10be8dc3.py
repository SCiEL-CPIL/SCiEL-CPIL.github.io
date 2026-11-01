import time
import itertools
import sys
from collections import Counter
import random

def _classic_superperm(n):
    """Classic recursive construction: length 1!+2!+...+n!."""
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n, all_perms):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    return all_perms.issubset(seen)

def _greedy_deletion_randomized(word, n, all_perms, deadline, seed=0):
    """
    Greedy deletion of redundant symbols, picking candidates randomly to avoid local minima.
    """
    if len(word) < n:
        return word
    
    current_word = list(word)
    L = len(current_word)
    
    # Precompute counts
    perm_counts = Counter()
    for i in range(L - n + 1):
        perm_counts[tuple(current_word[i:i + n])] += 1
        
    # We will keep deleting until no more safe deletions or time runs out
    # To avoid infinite loops or getting stuck, we limit the number of "no progress" restarts?
    # No, we just stop if no safe deletion is found.
    
    while time.time() < deadline:
        L = len(current_word)
        if L < n:
            break
            
        # Find all safe deletable indices
        # An index i is safe if for all windows containing i, count > 1
        safe_indices = []
        for i in range(L):
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(current_word[s:s + n])
                if perm_counts[w] == 1:
                    safe = False
                    break
            if safe:
                safe_indices.append(i)
                
        if not safe_indices:
            break
            
        # Pick a random safe index
        idx = random.choice(safe_indices)
        
        # Delete it
        current_word.pop(idx)
        
        # Update counts
        # Recompute is simple and fast enough for N~6000
        L = len(current_word)
        perm_counts = Counter()
        for i in range(L - n + 1):
            perm_counts[tuple(current_word[i:i + n])] += 1
            
    return current_word

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Generate classic
    word = _classic_superperm(n)
    
    # Strategy:
    # 1. Run one deterministic pass (left-to-right) to get a quick baseline.
    # 2. If time remains, run several randomized passes to try to find a shorter path.
    
    best_word = list(word)
    best_len = len(word)
    
    # Deterministic pass (greedy, first available)
    # This is fast and gives a good starting point
    # We can reuse the logic but force left-to-right
    # Actually, the randomized function with seed=None or specific logic can do this, 
    # but let's just run the randomized function with a few different seeds.
    
    # Since the gap is small (8 chars for n=7), we might need to be lucky.
    # Let's run up to 5 iterations if time permits.
    
    num_iterations = 5
    # Estimate time per iteration. 
    # One iteration might take ~0.1s - 1s depending on how many deletions.
    # We have 115s.
    
    for i in range(num_iterations):
        if time.time() > deadline - 10: # Stop if we are close to limit
            break
            
        # Use a different seed for each iteration
        seed = i * 12345 + 42
        polished = _greedy_deletion_randomized(word, n, all_perms, deadline - 5, seed)
        
        if _is_valid_superperm(polished, n, all_perms):
            if len(polished) < best_len:
                best_len = len(polished)
                best_word = polished
                # If we reached the theoretical lower bound or known optimum, we can stop?
                # Known optimum for n=7 is 5905.
                if n == 7 and best_len == 5905:
                    break

    return "".join(map(str, best_word))
