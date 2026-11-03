"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
from math import factorial

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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _build_perm_lookup(n):
    """Build a dictionary mapping permutation tuples to their integer indices."""
    lookup = {}
    for i, p in enumerate(permutations(range(1, n + 1))):
        lookup[p] = i
    return lookup

def _compute_covered(word, n, perm_lookup):
    """Compute the set of permutation indices covered by the word."""
    covered = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            idx = perm_lookup.get(window)
            if idx is not None:
                covered.add(idx)
    return covered

def _try_remove_chars_v2(word, n, time_limit, perm_lookup):
    """Try to remove redundant characters using precomputed window data."""
    start_time = time.time()
    word = list(word)
    total_perms = factorial(n)
    
    # Precompute: for each position, which permutations does it participate in?
    # A character at position i is in windows starting at [i-n+1, i]
    
    def get_affected_windows(pos, L):
        """Get window start positions that include character at pos."""
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)
        return list(range(lo, hi + 1))
    
    # Build initial window perm indices
    L = len(word)
    win_idx = []
    for w in range(L - n + 1):
        window = tuple(word[w:w + n])
        if len(set(window)) == n:
            idx = perm_lookup.get(window)
            win_idx.append(idx if idx is not None else -1)
        else:
            win_idx.append(-1)
    
    perm_count = [0] * total_perms
    for idx in win_idx:
        if idx >= 0:
            perm_count[idx] += 1
    covered_count = sum(1 for c in perm_count if c > 0)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        L = len(word)
        # Scan from end to start
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return word
            if L - 1 < n:
                continue
            
            # Check if removal at position i is safe
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            safe = True
            for w in range(lo, hi + 1):
                if w < len(win_idx) and win_idx[w] >= 0:
                    if perm_count[win_idx[w]] == 1:
                        safe = False
                        break
            
            if safe:
                # Perform removal
                removed_char = word.pop(i)
                L = len(word)
                
                # Rebuild win_idx (expensive but correct)
                win_idx = []
                for w in range(L - n + 1):
                    window = tuple(word[w:w + n])
                    if len(set(window)) == n:
                        idx = perm_lookup.get(window)
                        win_idx.append(idx if idx is not None else -1)
                    else:
                        win_idx.append(-1)
                
                perm_count = [0] * total_perms
                for idx in win_idx:
                    if idx >= 0:
                        perm_count[idx] += 1
                new_covered = sum(1 for c in perm_count if c > 0)
                
                if new_covered == covered_count:
                    covered_count = new_covered
                    improved = True
                    break
                else:
                    # Undo removal
                    word.insert(i, removed_char)
                    # Rebuild win_idx back
                    L = len(word)
                    win_idx = []
                    for w in range(L - n + 1):
                        window = tuple(word[w:w + n])
                        if len(set(window)) == n:
                            idx = perm_lookup.get(window)
                            win_idx.append(idx if idx is not None else -1)
                        else:
                            win_idx.append(-1)
                    perm_count = [0] * total_perms
                    for idx in win_idx:
                        if idx >= 0:
                            perm_count[idx] += 1
                    covered_count = sum(1 for c in perm_count if c > 0)
                    # Continue scanning
            # If not safe, continue
    
    return word

def _try_swap_chars(word, n, time_limit, perm_lookup):
    """Try swapping adjacent characters to potentially enable further removals."""
    start_time = time.time()
    word = list(word)
    total_perms = factorial(n)
    covered = _compute_covered(word, n, perm_lookup)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(word) - 1):
            if (time.time() - start_time) >= time_limit:
                return word
            # Try swapping characters at positions i and i+1
            if word[i] != word[i + 1]:
                word[i], word[i + 1] = word[i + 1], word[i]
                new_covered = _compute_covered(word, n, perm_lookup)
                if len(new_covered) >= len(covered):
                    covered = new_covered
                    improved = True
                    break
                else:
                    # Swap back
                    word[i], word[i + 1] = word[i + 1], word[i]
    
    return word

def construct(n):
    start_time = time.time()
    time_limit = 130  # Leave 20s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # For small n, the classic construction might already be optimal or close
    if n <= 4:
        return "".join(map(str, word))
    
    # Build permutation lookup for fast coverage checking
    perm_lookup = _build_perm_lookup(n)
    
    # Try to remove redundant characters using v2 approach
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _try_remove_chars_v2(word, n, remaining_time * 0.6, perm_lookup)
    
    # Try swaps to enable further removals
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _try_swap_chars(word, n, remaining_time * 0.3, perm_lookup)
    
    # Final pass: try removals again after swaps
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 3:
        word = _try_remove_chars_v2(word, n, remaining_time, perm_lookup)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
