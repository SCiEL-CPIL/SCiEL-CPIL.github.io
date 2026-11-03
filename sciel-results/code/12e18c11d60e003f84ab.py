"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
# EVOLVE-BLOCK-START

def _build_required_perms(n):
    """Build the set of all n! permutations as tuples."""
    return set(itertools.permutations(range(1, n + 1)))

def _count_windows(word, n, required_perms):
    """Count how many required permutations appear as windows in word."""
    count = 0
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if window in required_perms:
            count += 1
    return count

def _try_remove_fast(word, n, required_perms, deadline):
    """Try to remove symbols from word using fast counting. Returns shortened word."""
    word = list(word)
    target = len(required_perms)
    current_count = _count_windows(word, n, required_perms)
    if current_count < target:
        return word
    
    i = 0
    while i < len(word) - n + 1 and time.time() < deadline:
        # Try removing word[i]
        test = word[:i] + word[i+1:]
        new_count = _count_windows(test, n, required_perms)
        if new_count >= target:
            word = test
            current_count = new_count
            # Don't increment i since we removed an element
        else:
            i += 1
    return word

def _try_remove_multi_pass(word, n, required_perms, deadline):
    """Multiple passes of removal until no improvement or time limit."""
    word = list(word)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(word) - n + 1 and time.time() < deadline:
            test = word[:i] + word[i+1:]
            if _count_windows(test, n, required_perms) >= len(required_perms):
                word = test
                improved = True
                # Don't increment i
            else:
                i += 1
    return word

def construct(n):
    deadline = time.time() + 135  # 135s safety margin for 150s limit
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
        required_perms = _build_required_perms(n)
        
        # First pass removal
        word2 = _try_remove_fast(word, n, required_perms, deadline)
        if len(word2) < best_len:
            best_word = word2
            best_len = len(word2)
        
        # Multiple passes if time allows
        while time.time() < deadline:
            word3 = _try_remove_multi_pass(best_word, n, required_perms, deadline)
            if len(word3) < best_len:
                best_word = word3
                best_len = len(word3)
            else:
                break
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
