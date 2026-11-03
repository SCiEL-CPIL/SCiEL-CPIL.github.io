"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START

def _check_valid(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    target = n * n  # n!
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == target

def _try_remove(word, n, start_idx=0):
    """Try to remove symbols from word, starting at start_idx, greedily."""
    word = list(word)
    i = start_idx
    while i < len(word) - n + 1:
        # Try removing word[i]
        test = word[:i] + word[i+1:]
        if _check_valid(test, n):
            word = test
            # Don't increment i since we removed an element
        else:
            i += 1
    return word

def _try_remove_all_positions(word, n):
    """Try removing from all positions, multiple passes."""
    word = list(word)
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(word) - n + 1:
            test = word[:i] + word[i+1:]
            if _check_valid(test, n):
                word = test
                improved = True
                # Don't increment i
            else:
                i += 1
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
        
        # Multiple passes if time allows
        while time.time() < deadline:
            word3 = _try_remove_all_positions(best_word, n)
            if len(word3) < best_len:
                best_word = word3
                best_len = len(word3)
            else:
                break
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
