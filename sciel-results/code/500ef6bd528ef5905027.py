"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 125.0  # Safety margin before 150s kill
    
    # Classic recursive construction with optimized final step
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        if m == n:
            # For the final step, try multiple orderings to maximize overlap
            best_word = None
            best_len = float('inf')
            
            # Try the original ordering first
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            
            if len(new) < best_len:
                best_word = new
                best_len = len(new)
            
            # Try random reordering if we have time
            if time.time() < deadline - 30:
                num_trials = 50
                for trial in range(num_trials):
                    if time.time() > deadline - 10:
                        break
                    shuffled_perms = perms[:]
                    random.shuffle(shuffled_perms)
                    
                    new = []
                    for p in shuffled_perms:
                        block = list(p) + [m] + list(p)
                        overlap = 0
                        for o in range(min(len(new), len(block)), -1, -1):
                            if new[len(new) - o:] == block[:o]:
                                overlap = o
                                break
                        new += block[overlap:]
                    
                    if len(new) < best_len:
                        best_word = new
                        best_len = len(new)
            
            word = best_word
        else:
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    target = 1
    for i in range(1, n + 1):
        target *= i
    
    current = list(word)
    
    # Build permutation count map efficiently
    def build_perm_map(w, n):
        perm_count = defaultdict(int)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                perm_count[window] += 1
        return perm_count
    
    perm_map = build_perm_map(current, n)
    if len(perm_map) < target:
        return "".join(map(str, current))
    
    # Fast deletion pass: try removing each position, check if all perms still covered
    # Use a counter to track coverage efficiently
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        # Windows affected by removing position i
        start_w = max(0, i - n + 1)
        end_w = min(i, len(current) - n)
        
        # Check if any affected perm has count 1 (would be lost)
        can_remove = True
        for w in range(start_w, end_w + 1):
            window = tuple(current[w:w + n])
            if len(set(window)) == n:
                if perm_map[window] == 1:
                    can_remove = False
                    break
        
        if can_remove:
            new_word = current[:i] + current[i + 1:]
            # Quick verification
            new_pm = build_perm_map(new_word, n)
            if len(new_pm) == target:
                current = new_word
                perm_map = new_pm
    
    return "".join(map(str, current))
# EVOLVE-BLOCK-END
