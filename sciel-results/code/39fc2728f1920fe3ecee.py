"""Improved superpermutation construction using greedy + local search."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def _build_classic(n):
    """Classic recursive construction."""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _validate(word, n):
    """Check if word contains all n! permutations."""
    if len(word) < n:
        return False
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            found.add(window)
        if len(found) == len(list(permutations(range(1, n+1)))):
            return True
    return len(found) == n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5) * (n-6) if n == 7 else False

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 1:
        return str(n) if n == 1 else ""
    
    # Start with classic construction
    word = _build_classic(n)
    best_len = len(word)
    best_word = list(word)
    
    # Local search: try to improve by reordering segments
    # The idea: find consecutive windows that are permutations, 
    # and try alternative orderings that create more overlap
    
    # For n=7, the classic gives 13349, target is ~5905
    # We need a fundamentally better approach
    
    # Alternative: greedy construction with backtracking
    # Build word by always choosing next permutation that maximizes overlap
    # while adding a new permutation
    
    if n <= 7:
        # Use greedy + local search approach
        target_perms = list(permutations(range(1, n+1)))
        target_set = set(target_perms)
        
        # Greedy construction
        def greedy_construct(n, seed=42):
            rng = random.Random(seed)
            word = list(range(1, n+1))  # start with first permutation
            found = {tuple(word)}
            
            while len(found) < n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5) * (n-6) if n == 7 else len(found) < len(target_set):
                # Find all permutations that overlap with the end of current word
                # i.e., permutations P where P[:n-1] == word[-(n-1):]
                tail = tuple(word[-(n-1):])
                candidates = []
                for perm in target_set:
                    if perm not in found and perm[:n-1] == tail:
                        candidates.append(perm)
                
                if not candidates:
                    # Try to extend by finding a permutation that overlaps
                    # with some suffix of the word
                    for overlap_len in range(n-2, 0, -1):
                        suffix = tuple(word[-overlap_len:])
                        for perm in target_set:
                            if perm not in found and perm[n-overlap_len:] == suffix:
                                # We can append perm[overlap_len:]
                                candidates.append(perm)
                                break
                        if candidates:
                            break
                
                if not candidates:
                    break  # Stuck
                
                # Choose candidate with best overlap (most overlap = shorter extension)
                best_perm = max(candidates, key=lambda p: n-1)  # all have n-1 overlap
                word.extend(list(best_perm[1:]))
                found.add(tuple(best_perm))
            
            return word
        
        # Try greedy with different seeds
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            gw = greedy_construct(n, seed)
            if len(gw) < best_len and _validate(gw, n):
                best_word = gw
                best_len = len(gw)
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
