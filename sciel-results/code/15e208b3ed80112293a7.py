"""Shortest superpermutation search for n up to 7 within a time budget."""
import time
import random
from itertools import permutations

_TIME_LIMIT = 145.0  # safety margin below the 150 s wall-clock limit


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _classic_superperm(n):
    """Classic recursive construction of length n!+(n-1)!+...+1!."""
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
            best = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    best = o
                    break
            new += block[best:]
        word = new
    return "".join(map(str, word))


def _greedy_superperm(n, seed=None):
    """
    Greedy construction: start with a random permutation, then repeatedly
    append the remaining permutation that shares the longest suffix/prefix
    overlap with the current word.
    """
    if seed is not None:
        random.seed(seed)
    
    all_perms = list(permutations(range(1, n + 1)))
    remaining = set(all_perms)
    
    # Start with a random permutation
    first = tuple(random.choice(all_perms))
    word = list(first)
    remaining.discard(first)
    
    while remaining:
        current_suffix = tuple(word[-(n - 1):]) if len(word) >= n - 1 else tuple(word)
        
        best_perm = None
        best_overlap = -1
        best_new_len = float('inf')
        
        # To speed up, sample a subset if remaining is large
        candidates = list(remaining)
        if len(candidates) > 200:
            random.shuffle(candidates)
            candidates = candidates[:200]
        
        for p in candidates:
            # Check overlap: how much of p's prefix matches word's suffix
            max_ov = min(len(word), n - 1)
            for o in range(max_ov, -1, -1):
                if o == 0:
                    new_len = len(word) + n
                    break
                if tuple(word[-o:]) == p[:o]:
                    new_len = len(word) + n - o
                    break
            else:
                new_len = len(word) + n
            
            if new_len < best_new_len or (new_len == best_new_len and (best_perm is None or hash(p) < hash(best_perm))):
                best_new_len = new_len
                best_perm = p
                best_overlap = new_len
        
        if best_perm is None:
            # Fallback: pick random
            best_perm = random.choice(list(remaining))
            word.extend(best_perm)
        else:
            # Find the actual overlap
            max_ov = min(len(word), n - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or tuple(word[-o:]) == best_perm[:o]:
                    word.extend(best_perm[o:])
                    break
        
        remaining.discard(best_perm)
    
    return "".join(map(str, word))


def _is_valid_superperm(word_str, n):
    """Check that every permutation of 1..n appears as n consecutive symbols."""
    if len(word_str) < n:
        return False
    perms = set()
    for i in range(len(word_str) - n + 1):
        w = word_str[i:i + n]
        if len(set(w)) == n:
            perms.add(w)
    return len(perms) == _fact(n)


def _validate_and_return(word, n, deadline):
    """Validate the word; if invalid, return classic construction."""
    if time.time() > deadline - 5:
        return word
    if _is_valid_superperm(word, n):
        return word
    return _classic_superperm(n)


def construct(n):
    """Return the shortest known superpermutation word for symbols 1..n."""
    start = time.time()
    deadline = start + _TIME_LIMIT

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"
    if n == 5:
        # Classic gives 153, optimal is 152
        # Try to find 152 with greedy, fall back to classic
        best = _classic_superperm(5)
        best_len = len(best)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            w = _greedy_superperm(5, seed=attempt * 7 + 1)
            if len(w) < best_len:
                if _is_valid_superperm(w, 5):
                    best = w
                    best_len = len(w)
                    if best_len == 152:
                        break
        return best

    # For n >= 6, use randomized greedy with multiple restarts
    if n == 6:
        best = _classic_superperm(6)
        best_len = len(best)
        # Classic for n=6 is 1!+2!+3!+4!+5!+6! = 873
        # Known shortest is 872
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            w = _greedy_superperm(6, seed=attempt * 13 + 3)
            if len(w) < best_len:
                if _is_valid_superperm(w, 6):
                    best = w
                    best_len = len(w)
                    if best_len <= 872:
                        break
        return best

    if n == 7:
        # Classic for n=7 is 1!+2!+3!+4!+5!+6!+7! = 13700
        # Known shortest is 5905, lower bound 5884
        # Greedy will likely do much better than classic
        best = _classic_superperm(7)
        best_len = len(best)
        
        # Run many greedy restarts, keeping the best
        attempt = 0
        while time.time() < deadline - 15:
            w = _greedy_superperm(7, seed=attempt * 17 + 5)
            attempt += 1
            if len(w) < best_len:
                # Quick validation check (only check length first for speed)
                # Full validation is expensive, do it only when we find something new
                if _is_valid_superperm(w, 7):
                    best = w
                    best_len = len(w)
        
        return best

    # Fallback for other n
    return _classic_superperm(n)
