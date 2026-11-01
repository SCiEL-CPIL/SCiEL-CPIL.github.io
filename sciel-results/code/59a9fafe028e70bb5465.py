"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Classic recursive construction as baseline
    word = _classic_superpermutation(n)
    best_word = list(word)
    
    if n < 4:
        return "".join(map(str, best_word))
    
    # Try to improve via local search on the permutation order
    # For n=7, the classic construction has length 13699
    # We try to find a shorter valid superpermutation
    
    # Strategy: For each new symbol m, try different insertion orders
    # to minimize total length
    
    # First, let's try a smarter recursive construction
    # where we optimize the order in which permutations are processed
    
    improved = _optimized_recursive(n, deadline)
    if improved is not None and len(improved) < len(best_word):
        best_word = improved
    
    # Try local search to swap adjacent blocks in the construction
    # This can sometimes reduce the total length
    
    if time.time() < deadline - 5.0:
        locally_improved = _local_search_superpermutation(n, best_word, deadline)
        if locally_improved is not None and len(locally_improved) < len(best_word):
            best_word = locally_improved
    
    return "".join(map(str, best_word))

def _classic_superpermutation(n):
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
            overlap = _max_overlap(new, block)
            new += block[overlap:]
        word = new
    return word

def _max_overlap(current, block):
    """Find maximum overlap between end of current and start of block."""
    max_possible = min(len(current), len(block))
    for o in range(max_possible, -1, -1):
        if o == 0:
            return 0
        if current[len(current) - o:] == block[:o]:
            return o
    return 0

def _optimized_recursive(n, deadline):
    """Try to optimize the recursive construction by choosing better orders."""
    # For small n, try all possible orderings of permutations
    # For larger n, use greedy heuristics
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            return None
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For small number of perms, try to find optimal order
        if len(perms) <= 8:
            best_order = _find_best_order(word, perms, m, deadline)
            if best_order is None:
                best_order = perms
        else:
            best_order = _greedy_order(word, perms, m, deadline)
        
        new = []
        for p in best_order:
            block = list(p) + [m] + list(p)
            overlap = _max_overlap(new, block)
            new += block[overlap:]
        word = new
    
    return word

def _find_best_order(base_word, perms, m, deadline):
    """Find the best ordering of permutations to minimize total length."""
    if len(perms) <= 4:
        # Try all permutations
        best_len = float('inf')
        best_order = None
        for order in itertools.permutations(perms):
            if time.time() > deadline:
                return None
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                overlap = _max_overlap(new, block)
                new += block[overlap:]
            if len(new) < best_len:
                best_len = len(new)
                best_order = list(order)
        return best_order
    else:
        # Use greedy for larger cases
        return _greedy_order([], perms, m, deadline)

def _greedy_order(base_word, perms, m, deadline):
    """Greedy ordering: always pick the next permutation that gives maximum overlap."""
    remaining = list(perms)
    order = []
    current = []
    
    while remaining:
        if time.time() > deadline:
            return order if order else perms
        
        best_idx = 0
        best_overlap = -1
        for i, p in enumerate(remaining):
            block = list(p) + [m] + list(p)
            overlap = _max_overlap(current, block)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        p = remaining.pop(best_idx)
        block = list(p) + [m] + list(p)
        overlap = _max_overlap(current, block)
        current += block[overlap:]
        order.append(p)
    
    return order

def _local_search_superpermutation(n, initial_word, deadline):
    """Try local search to improve the superpermutation."""
    # This is a simple approach: try to find shorter valid superpermutations
    # by modifying the structure
    
    word = list(initial_word)
    
    # Try to remove redundant characters where possible
    # A character at position i is redundant if removing it still leaves
    # all permutations present
    
    # For efficiency, only try on smaller n
    if n > 5:
        return None
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(len(word) - 1, -1, -1):
            # Check if removing word[i] still gives valid superpermutation
            test_word = word[:i] + word[i+1:]
            if _is_valid_superpermutation(test_word, n):
                word = test_word
                improved = True
                break
    
    return word

def _is_valid_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive symbols."""
    if len(word) < n:
        return False
    
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
    
    return len(seen) == _factorial(n)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
# EVOLVE-BLOCK-END
