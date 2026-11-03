"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Classic recursive construction
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
    
    # Try to shorten the word by finding and removing redundant parts
    if n >= 3 and time.time() < deadline:
        word = try_shorten(word, n, deadline)
    
    return "".join(map(str, word))

def get_permutations_covered(word, n):
    """Get set of all permutations covered by the word."""
    covered = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            covered.add(window)
    return covered

def try_shorten(word, n, deadline):
    """Try to shorten a superpermutation by removing redundant symbols."""
    best = list(word)
    best_len = len(best)
    best_perms = get_permutations_covered(best, n)
    
    # Greedy local search: try removing each position and keep if all perms still covered
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best)
        current_perms = set(best_perms)
        
        # Try removing each position from right to left (to avoid index shifts)
        for pos in range(len(current) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            # Skip if removing this would make word too short
            if len(current) - 1 < n:
                continue
                
            # Try removing position pos
            trial = current[:pos] + current[pos + 1:]
            if len(trial) < n:
                continue
                
            # Check if all permutations are still covered
            trial_perms = get_permutations_covered(trial, n)
            if len(trial_perms) == len(current_perms):
                # Check if it's the same set
                if trial_perms == current_perms:
                    current = trial
                    current_perms = trial_perms
                    improved = True
                    break
        
        if improved and current != best:
            best = current
            best_perms = current_perms
    
    return best
# EVOLVE-BLOCK-END
