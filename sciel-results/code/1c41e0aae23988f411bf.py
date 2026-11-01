import random
import time
import sys
from itertools import permutations
import math

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation of 1..n.
    
    Uses the standard recursive construction as a starting point,
    then attempts local search to reduce length.
    """
    if n <= 1:
        return "1"
    
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Local search to try to reduce length
    word = _local_search(word, n, time_budget=115)
    
    return "".join(map(str, word))

def _local_search(word, n, time_budget=115):
    """Try to shorten the word while maintaining superpermutation property."""
    start_time = time.time()
    word = list(word)
    
    target = math.factorial(n)
    
    def count_perms(w):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)
    
    current_count = count_perms(word)
    if current_count < target:
        return word  # Not valid, return as-is
    
    # Greedy deletion: repeatedly try to delete symbols
    # Use a more systematic approach: try positions in order of "redundancy"
    
    for outer_iter in range(500):
        if time.time() - start_time > time_budget:
            break
        
        if len(word) <= n:
            break
        
        found_deletion = False
        
        # Try random positions first (faster to find easy deletions)
        for attempt in range(200):
            if time.time() - start_time > time_budget:
                break
            
            pos = random.randint(1, len(word) - 2)
            new_word = word[:pos] + word[pos+1:]
            
            if count_perms(new_word) >= target:
                word = new_word
                found_deletion = True
                break
        
        if not found_deletion:
            # Try sequential scan from the middle outward
            mid = len(word) // 2
            for offset in range(1, len(word) // 2):
                if time.time() - start_time > time_budget:
                    break
                
                for pos in [mid - offset, mid + offset]:
                    if pos < 1 or pos > len(word) - 2:
                        continue
                    new_word = word[:pos] + word[pos+1:]
                    if count_perms(new_word) >= target:
                        word = new_word
                        found_deletion = True
                        break
                if found_deletion:
                    break
        
        if not found_deletion:
            # If we can't find any deletion, try a few more random attempts
            for attempt in range(100):
                if time.time() - start_time > time_budget:
                    break
                pos = random.randint(1, len(word) - 2)
                new_word = word[:pos] + word[pos+1:]
                if count_perms(new_word) >= target:
                    word = new_word
                    found_deletion = True
                    break
            
            if not found_deletion:
                # Give up if we've tried hard enough
                if outer_iter > 50:
                    break
    
    return word

# EVOLVE-BLOCK-END
