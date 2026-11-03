"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def _classic_superperm(n):
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superperm(word, n, target_perms):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    return seen == target_perms

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # For small n, classic is optimal or near optimal, and search is fast
    if n <= 4:
        return "".join(map(str, word))
        
    target_perms = set(permutations(range(1, n + 1)))
    
    # Phase 1: Greedy Right-to-Left Deletion
    # Scanning right-to-left allows us to continue scanning left after a deletion
    # without restarting the entire loop, as indices to the left are unaffected.
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # We scan from the end to the beginning
        for i in range(len(word) - 1, -1, -1):
            if (time.time() - start_time) >= deadline:
                break
            # Try removing character at position i
            test_word = word[:i] + word[i + 1:]
            if _is_valid_superperm(test_word, n, target_perms):
                word = test_word
                improved = True
                # Continue the for loop, which will naturally move to i-1
        
    # Phase 2: Random Local Search
    # If greedy got stuck, try random deletions to escape local minima
    best_word = word
    best_len = len(best_word)
    
    while (time.time() - start_time) < deadline:
        if len(best_word) <= 0:
            break
            
        # Pick a random index to delete
        idx = random.randint(0, len(best_word) - 1)
        test_word = best_word[:idx] + best_word[idx + 1:]
        
        if _is_valid_superperm(test_word, n, target_perms):
            best_word = test_word
            best_len -= 1
            # We could optionally restart greedy here, but random is fine
            # If we want to be aggressive, we can check if we improved significantly
            
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
