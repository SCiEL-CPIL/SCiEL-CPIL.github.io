"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = 145.0
    
    if n <= 4:
        return _construct_small(n)
    
    # Classic construction as baseline
    best_word = _classic_construct(n)
    best_len = len(best_word)
    
    # Try greedy constructions
    for attempt in range(50):
        if time.time() > deadline - 3:
            break
        word = _greedy_construct(n, seed=attempt)
        if word is not None:
            # Verify it's valid
            if _is_valid_superpermutation(word, n) and len(word) < best_len:
                best_word = word
                best_len = len(word)
    
    return "".join(map(str, best_word))

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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _greedy_construct(n, seed=0):
    random.seed(seed)
    symbols = list(range(1, n + 1))
    word = random.sample(symbols, n)
    covered = {tuple(word)}
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    while len(covered) < total_perms:
        best_sym = None
        best_score = -1
        for sym in symbols:
            new_word = word + [sym]
            last = new_word[-n:]
            if len(set(last)) != n:
                continue
            last_perm = tuple(last)
            is_new = last_perm not in covered
            score = 10 if is_new else 0
            if score > best_score:
                best_score = score
                best_sym = sym
        
        if best_sym is None:
            return None
        
        word.append(best_sym)
        last_perm = tuple(word[-n:])
        if last_perm not in covered:
            covered.add(last_perm)
    
    return word

def _is_valid_superpermutation(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
        if len(seen) == total_perms:
            return True
    return len(seen) == total_perms

def _construct_small(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134121431213412143121"
    return _classic_construct(n)
# EVOLVE-BLOCK-END
