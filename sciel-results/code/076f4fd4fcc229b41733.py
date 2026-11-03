"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _check_valid(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    target = _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == target

def _try_remove(word, n, start_idx=0):
    """Try to remove symbols from word using incremental coverage tracking."""
    word = list(word)
    target = _factorial(n)
    L = len(word)
    
    # Build coverage: for each permutation, track count of occurrences
    from collections import defaultdict
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            perm_count[window] += 1
    
    if len(perm_count) < target:
        return word
    
    i = start_idx
    while i < L - n + 1:
        # Determine which windows are affected by removing position i
        # Window at position j covers positions [j, j+n-1]
        # Position i is in window j iff j <= i <= j+n-1, i.e., j in [i-n+1, i]
        j_min = max(0, i - n + 1)
        j_max = min(L - n, i)
        
        # Check if any permutation would lose its last occurrence
        safe = True
        for j in range(j_min, j_max + 1):
            window = tuple(word[j:j + n])
            if len(set(window)) == n:
                if perm_count[window] == 1:
                    safe = False
                    break
        
        if safe:
            # Remove position i
            word.pop(i)
            L -= 1
            # Update perm_count for affected windows
            # After removal, old windows at j_min..j_max are gone
            # New windows at j_min..j_max-1 (shifted) may be different
            # Simplest: rebuild affected portion
            for j in range(j_min, min(j_max + 1, L - n + 1)):
                # Old window at j (before removal) is no longer valid
                pass
            # Rebuild coverage for the affected region
            # For simplicity, rebuild entire coverage
            perm_count = defaultdict(int)
            for j in range(L - n + 1):
                window = tuple(word[j:j + n])
                if len(set(window)) == n:
                    perm_count[window] += 1
            # Don't increment i
        else:
            i += 1
    
    return word

def _try_remove_all_positions(word, n):
    """Try removing from all positions, multiple passes."""
    word = list(word)
    improved = True
    while improved:
        improved = False
        word2 = _try_remove(word, n, 0)
        if len(word2) < len(word):
            word = word2
            improved = True
    return word

def construct(n):
    deadline = time.time() + 140  # 140s safety margin for 150s limit
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
    
    best_word = word
    best_len = len(word)
    
    # Try to shorten by removing symbols
    if n <= 7:
        # Single pass removal
        word2 = _try_remove(word, n)
        if len(word2) < best_len:
            best_word = word2
            best_len = len(word2)
        
        # Multiple passes with different starting offsets if time allows
        while time.time() < deadline - 10:
            improved = False
            for offset in range(0, n):
                if time.time() > deadline - 5:
                    break
                word3 = _try_remove(best_word, n, offset)
                if len(word3) < best_len:
                    best_word = word3
                    best_len = len(word3)
                    improved = True
            if not improved:
                break
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
