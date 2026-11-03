"""Shortest superpermutation solver using greedy insertion with a time budget."""
import time
import math
from itertools import permutations
import random

# Global deadline guard
_START_TIME = None
TIME_LIMIT = 115.0  # safety margin below 120s

def _check_deadline():
    """Return True if we should stop searching."""
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > TIME_LIMIT

def _build_greedy(n, symbol_order):
    """
    Build a superpermutation by greedily inserting permutations.
    symbol_order: list of permutations (tuples) in the order they should be added.
    Returns the final word as a list of ints.
    """
    # Start with the first permutation
    word = list(symbol_order[0])
    
    # For efficiency, maintain a set of already-covered permutations
    covered = set()
    covered.add(tuple(symbol_order[0]))
    
    # Precompute suffix/prefix matching for faster overlap detection
    # We'll use a dictionary mapping suffix of word to positions
    
    for perm in symbol_order[1:]:
        if _check_deadline():
            break
        perm = list(perm)
        best_overlap = 0
        best_pos = len(word)
        
        # Try all possible overlap lengths from max to min
        # Overlap of length k means word[-k:] == perm[:k]
        # Max possible overlap is n-1 (since we need to add at least 1 new symbol)
        for k in range(min(n - 1, len(word)), 0, -1):
            suffix = tuple(word[-k:])
            prefix = tuple(perm[:k])
            if suffix == prefix:
                best_overlap = k
                best_pos = len(word) - k
                break
        
        # Append the non-overlapping part
        if best_overlap > 0:
            word.extend(perm[best_overlap:])
        else:
            word.extend(perm)
        
        # Update covered set (only add if not already covered)
        # For large n, checking all windows is expensive, so we skip this for n > 5
        if n <= 5:
            # Check all windows in the new portion
            start_idx = len(word) - len(perm)
            for i in range(start_idx, len(word) - n + 1):
                w = tuple(word[i:i+n])
                covered.add(w)
    
    return word

def _greedy_heuristic(n, order_type="random", seed=42):
    """
    Generate a superpermutation using greedy insertion with a specific order.
    """
    if n <= 1:
        return [1]
    
    perms = list(permutations(range(1, n + 1)))
    
    if order_type == "random":
        random.seed(seed)
        random.shuffle(perms)
    elif order_type == "lex":
        perms.sort()
    elif order_type == "reverse_lex":
        perms.sort(reverse=True)
    elif order_type == "gray_code":
        # Generate permutations in Gray code order (adjacent permutations differ by one swap)
        def gray_code_perms(n):
            """Generate permutations in Gray code order using Steinhaus-Johnson-Trotter."""
            if n == 1:
                yield (1,)
                return
            # Simple approach: use the fact that we can generate all perms and sort by a key
            # For simplicity, use lex order as fallback
            yield from sorted(permutations(range(1, n + 1)))
        perms = list(gray_code_perms(n))
    
    return _build_greedy(n, perms)

def _verify_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    if len(word) < n:
        return False
    all_perms = set(permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        found.add(tuple(word[i:i+n]))
    return found == all_perms

def construct(n):
    global _START_TIME
    _START_TIME = time.time()
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest is 33
        # Use classic construction
        return _classic_construction(n)
    
    # For n >= 5, use greedy insertion with multiple strategies
    # Strategy 1: Classic recursive construction (known to be good for small n)
    if n <= 5:
        classic = _classic_construction(n)
        if _verify_superperm(classic, n):
            return "".join(map(str, classic))
    
    # For larger n, try greedy with different orderings
    best_word = None
    best_len = float('inf')
    
    # Try several random seeds
    for seed in [42, 123, 999, 777, 555]:
        if _check_deadline():
            break
        word = _greedy_heuristic(n, "random", seed)
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    # Also try lex order
    if not _check_deadline():
        word = _greedy_heuristic(n, "lex")
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    # Try reverse lex
    if not _check_deadline():
        word = _greedy_heuristic(n, "reverse_lex")
        if len(word) < best_len:
            best_word = word
            best_len = len(word)
    
    if best_word is None:
        # Fallback: classic construction
        best_word = _classic_construction(n)
    
    return "".join(map(str, best_word))

def _classic_construction(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word
