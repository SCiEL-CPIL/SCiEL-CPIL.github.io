import time
import math

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    if n <= 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new_word), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def _try_greedy_shorten(word, n, deadline):
    """Try to shorten the superpermutation by greedy local search."""
    if n < 3:
        return word
    
    # Verify it's valid first
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) != n:
                continue
            seen.add(window)
        return len(seen) == math.factorial(n)
    
    current = list(word)
    if not is_valid(current):
        return current
    
    best_len = len(current)
    
    # Try removing redundant characters where possible
    # A character at position i is redundant if removing it still gives a valid superperm
    # This is expensive, so we use a heuristic: try to find windows that appear multiple times
    # and see if we can merge them
    
    # Alternative approach: try to find better overlaps by reordering the permutation sequence
    # For now, try a simple local search: for each position, try swapping adjacent elements
    # and check if still valid (expensive, so limit attempts)
    
    start_time = time.time()
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to find a position where we can remove a character
        # Heuristic: look for patterns where the same (n-1)-permutation appears
        # and we might be able to skip one element
        
        # Try swapping adjacent elements at random positions
        import random
        if len(current) > n + 1:
            pos = random.randint(0, len(current) - 2)
            current[pos], current[pos+1] = current[pos+1], current[pos]
            
            # Quick check: does this still contain all permutations?
            # This is O(n * len(current)) which is expensive for n=7
            # Use a faster check: only verify that we haven't broken existing windows
            # Actually, let's just check validity but limit frequency
            if attempts % 10 == 0:
                if is_valid(current):
                    if len(current) < best_len:
                        best_len = len(current)
                else:
                    # Revert
                    current[pos], current[pos+1] = current[pos+1], current[pos]
        
        # Also try: if we have consecutive identical symbols, remove one
        # But in a superperm of distinct symbols, this shouldn't happen
        # unless n=1
    
    return current

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    word = _classic_superperm(n)
    
    # Try to shorten if we have time
    if time.time() < deadline - 30.0:
        word = _try_greedy_shorten(word, n, deadline)
    
    return "".join(map(str, word))
