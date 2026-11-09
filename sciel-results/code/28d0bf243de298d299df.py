"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations as _perms
from collections import Counter, defaultdict

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    deadline = time.time() + 115.0
    
    # Classic construction
    word = _classic_construct(n)
    L = len(word)
    
    # Convert to list for easier manipulation
    word = list(word)
    L = len(word)
    
    # Precompute all windows and their counts
    perm_count = Counter()
    for i in range(L - n + 1):
        perm_count[tuple(word[i:i+n])] += 1
    
    # Build reverse index: for each permutation, which window positions contain it
    perm_positions = defaultdict(list)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        perm_positions[w].append(i)
    
    # Multiple aggressive removal passes
    removed = [False] * L
    for pass_num in range(50):
        if time.time() > deadline - 5.0:
            break
        
        progress = False
        # Try removing positions from right to left (to maintain window indices)
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            if removed[pos]:
                continue
            
            # Check if all windows containing pos have count > 1
            valid = True
            min_i = max(0, pos - n + 1)
            max_i = min(pos, L - n)
            for wi in range(min_i, max_i + 1):
                if removed[wi] or removed[wi + n - 1]:
                    continue
                w = tuple(word[wi:wi+n])
                if perm_count[w] <= 1:
                    valid = False
                    break
            
            if valid:
                removed[pos] = True
                progress = True
                # Update counts for windows containing pos
                for wi in range(min_i, max_i + 1):
                    if removed[wi] or removed[wi + n - 1]:
                        continue
                    w = tuple(word[wi:wi+n])
                    perm_count[w] -= 1
    
    # Build intermediate word
    word = [word[i] for i in range(L) if not removed[i]]
    L = len(word)
    
    # Rebuild perm_count for the new word
    perm_count = Counter()
    for i in range(L - n + 1):
        perm_count[tuple(word[i:i+n])] += 1
    
    # Second phase: more aggressive removal with multiple passes
    removed = [False] * L
    for pass_num in range(100):
        if time.time() > deadline - 3.0:
            break
        
        progress = False
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 3.0:
                break
            if removed[pos]:
                continue
            
            valid = True
            min_i = max(0, pos - n + 1)
            max_i = min(pos, L - n)
            for wi in range(min_i, max_i + 1):
                if removed[wi] or removed[wi + n - 1]:
                    continue
                w = tuple(word[wi:wi+n])
                if perm_count[w] <= 1:
                    valid = False
                    break
            
            if valid:
                removed[pos] = True
                progress = True
                for wi in range(min_i, max_i + 1):
                    if removed[wi] or removed[wi + n - 1]:
                        continue
                    w = tuple(word[wi:wi+n])
                    perm_count[w] -= 1
        
        if not progress:
            break
    
    # Build final word
    word = [word[i] for i in range(L) if not removed[i]]
    return "".join(map(str, word))


def _classic_construct(n):
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word
# EVOLVE-BLOCK-END
