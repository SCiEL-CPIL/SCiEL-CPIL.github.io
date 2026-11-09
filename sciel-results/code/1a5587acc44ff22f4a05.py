import time
import sys
import math
from itertools import permutations

def _classic_construct(n):
    """Classical recursive construction."""
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
    return word

def _local_search_shorten(word, n, time_limit=10):
    """Try to shorten the word by finding and removing redundant segments.
    
    Uses coverage counting: a position is removable if no permutation with
    coverage count 1 has a window that includes that position.
    """
    start_time = time.time()
    word = list(word)
    
    # Build mapping from permutation to index
    required_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(required_perms)}
    total_perms = len(required_perms)
    
    # coverage[perm_idx] = number of windows in word that match this permutation
    coverage = [0] * total_perms
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        pidx = perm_to_idx.get(window, -1)
        if pidx >= 0:
            coverage[pidx] += 1
    
    def get_pidx(window):
        return perm_to_idx.get(window, -1)
    
    # Iteratively remove positions that are safe
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for i in range(1, len(word) - 1):
            # Position i appears in windows starting from max(0, i-n+1) to min(len(word)-n, i)
            start_win = max(0, i - n + 1)
            end_win = min(len(word) - n, i)
            
            # Check if any permutation with coverage==1 has a window covering position i
            safe = True
            for w in range(start_win, end_win + 1):
                pidx = get_pidx(tuple(word[w:w + n]))
                if pidx >= 0 and coverage[pidx] == 1:
                    safe = False
                    break
            
            if safe:
                # Decrement coverage for all windows that included position i
                for w in range(start_win, end_win + 1):
                    pidx = get_pidx(tuple(word[w:w + n]))
                    if pidx >= 0:
                        coverage[pidx] -= 1
                
                word.pop(i)
                changed = True
                break  # Restart scan from beginning
    
    return word

def construct(n):
    """Construct the shortest known superpermutation for n symbols.
    
    Uses the classical recursive construction with local search optimization.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with classical construction
    word = _classic_construct(n)
    
    # Try local search to shorten
    if n <= 7 and (time.time() < deadline - 5):
        word = _local_search_shorten(word, n, time_limit=min(10, deadline - time.time() - 5))
    
    return "".join(map(str, word))
