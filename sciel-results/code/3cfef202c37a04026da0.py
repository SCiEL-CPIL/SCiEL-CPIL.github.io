"""Superpermutation search: build a classic superpermutation then try to shorten it."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations


def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            # find maximum overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _all_perm_set(n):
    """Return a set of all permutations of 1..n as tuples."""
    return set(permutations(range(1, n + 1)))


def _windows_set(word, n):
    """Return the set of all n-length windows of word as tuples."""
    s = set()
    for i in range(len(word) - n + 1):
        s.add(tuple(word[i:i + n]))
    return s


def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as n-consecutive symbols."""
    if len(word) < n:
        return False
    windows = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        # quick check: must have all n distinct symbols
        if len(set(w)) != n:
            continue
        windows.add(w)
        if len(windows) == n ** 0:  # placeholder
            pass
    # Better: count distinct valid permutations
    count = 0
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n and w not in seen:
            seen.add(w)
            count += 1
            if count == _factorial(n):
                return True
    return count == _factorial(n)


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _try_remove_chars(word, n, time_limit):
    """
    Try to shorten the superpermutation by removing characters one at a time,
    or by swapping/removing blocks. Uses a greedy + local search approach.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    best_len = len(word)
    
    def window_count(w):
        """Count distinct valid n-permutations in w."""
        seen = set()
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i + n])
            if len(set(t)) == n:
                seen.add(t)
        return len(seen)
    
    # First, verify we start valid
    cur_count = window_count(word)
    if cur_count < target_perms:
        return word
    
    # Greedy removal: try to remove each position, keep if still valid
    # Do multiple passes
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        # Try removing each character (from end to start to avoid index shifting issues)
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Skip if removing would make length < n
            if len(word) - 1 < n:
                break
            # Try removing position i
            new_word = word[:i] + word[i + 1:]
            if window_count(new_word) >= target_perms:
                word = new_word
                improved = True
                best_len = len(word)
    
    # Now try swapping adjacent characters or small block moves
    # Local search: try moving a character to another position
    for _round in range(200):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(word)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(word)):
                if time.time() > deadline - 3:
                    break
                # Try moving word[i] to position j
                if i == j:
                    continue
                new_word = word[:]
                val = new_word.pop(i)
                new_word.insert(j, val)
                if window_count(new_word) >= target_perms:
                    word = new_word
                    improved = True
                    best_len = len(word)
                    break
            if improved:
                break
        if not improved:
            break
    
    return word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 120  # evaluator limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Step 1: Build classic superpermutation
    word = _classic_superperm(n)
    
    # Step 2: Try to shorten it
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 10  # safety margin
    
    if remaining > 5:
        word = _try_remove_chars(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
