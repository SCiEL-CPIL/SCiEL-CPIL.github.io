import time
import sys
from itertools import permutations

def _recursive_superperm(n):
    """Classic recursive construction of length sum(1!..n!)."""
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _verify(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive n-tuples."""
    total = 1
    for i in range(2, n + 1):
        total *= i
    found = set()
    for i in range(len(word) - n + 1):
        t = tuple(word[i:i + n])
        if len(set(t)) == n and all(1 <= x <= n for x in t):
            found.add(t)
    return len(found) == total, len(found)

def _local_improve_fast(word, n, deadline):
    """
    Fast local improvement: find and remove redundant internal permutations.
    A permutation at position pos is redundant if the (n-1)-window before it
    equals the (n-1)-window after it, allowing us to remove n-1 characters.
    """
    word = list(word)
    if len(word) < 2 * n:
        return word
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 10:
            break
        
        # Build permutation positions efficiently
        perm_positions = {}
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i + n])
            if len(set(p)) == n and all(1 <= x <= n for x in p):
                perm_positions.setdefault(p, []).append(i)
        
        # Find redundant permutations (appearing more than once)
        redundant = []
        for p, positions in perm_positions.items():
            if len(positions) > 1:
                redundant.append((p, positions))
        
        # Try to remove internal occurrences
        for p, positions in redundant:
            if time.time() > deadline - 10:
                break
            internal = positions[1:-1]
            for pos in internal:
                if pos < n - 1 or pos + n > len(word) - (n - 1):
                    continue
                before = tuple(word[pos - (n - 1):pos])
                after = tuple(word[pos + n - 1:pos + 2 * (n - 1)])
                if len(set(before)) == n - 1 and len(set(after)) == n - 1:
                    if before == after:
                        new_word = word[:pos] + word[pos + n - 1:]
                        valid, _ = _verify(new_word, n)
                        if valid:
                            word = new_word
                            improved = True
                            break
            if improved:
                break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Use classic recursive construction
    word = _recursive_superperm(n)
    
    # Apply local improvements
    if len(word) > n and time.time() < deadline - 20:
        word = _local_improve_fast(word, n, deadline)
    
    return "".join(map(str, word))
