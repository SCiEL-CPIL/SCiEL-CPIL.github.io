"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _perm_to_int(p, n):
    """Convert a permutation tuple to a unique integer using factorial number system."""
    result = 0
    for i, v in enumerate(p):
        result = result * (n - i) + (v - 1)
    return result

def _try_shorten(word, n, deadline):
    """Try to shorten the superpermutation by removing redundant symbols using efficient checking."""
    word = list(word)
    target_perms = _factorial(n)
    
    # Precompute all valid permutations as a set for fast membership
    all_perms = set(permutations(range(1, n + 1)))
    
    def compute_seen_count(w):
        """Count distinct permutations in word w."""
        seen = set()
        for j in range(len(w) - n + 1):
            window = tuple(w[j:j+n])
            if window in all_perms:
                seen.add(window)
        return len(seen)
    
    # First pass: try removing each position
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        L = len(word)
        # Try positions in a somewhat random order to avoid getting stuck
        for i in range(L - 1, -1, -1):
            if time.time() > deadline - 10:
                return word
            if len(word) <= n:
                break
            
            # Quick check: if removing position i, check if any permutation is lost
            # A permutation is lost if it appears exactly once and uses position i
            # This is expensive, so just try the removal and check
            new_word = word[:i] + word[i+1:]
            if len(new_word) < n:
                continue
            
            seen = set()
            for j in range(len(new_word) - n + 1):
                window = tuple(new_word[j:j+n])
                if window in all_perms:
                    seen.add(window)
            
            if len(seen) == target_perms:
                word = new_word
                improved = True
                break
    
    return word

def _try_greedy_construction(n, deadline):
    """Try a greedy construction that might produce a shorter superpermutation."""
    # This is a heuristic approach: build the word greedily
    # At each step, try to extend by one symbol that maximizes new permutations
    
    # Start with a permutation of 1..n
    start_perm = list(range(1, n + 1))
    word = start_perm[:]
    seen = {tuple(start_perm)}
    target = _factorial(n)
    
    # Try extending by appending symbols
    while len(seen) < target and time.time() < deadline - 15:
        best_ext = None
        best_count = 0
        for sym in range(1, n + 1):
            # Try appending sym
            new_word = word + [sym]
            new_seen = set(seen)
            for j in range(len(new_word) - n + 1):
                window = tuple(new_word[j:j+n])
                if window not in new_seen:
                    if len(set(window)) == n:
                        new_seen.add(window)
            new_count = len(new_seen) - len(seen)
            if new_count > best_count:
                best_count = new_count
                best_ext = sym
        
        if best_ext is None or best_count == 0:
            break
        
        word.append(best_ext)
        # Update seen
        for j in range(len(word) - n + 1):
            window = tuple(word[j:j+n])
            if window not in seen:
                if len(set(window)) == n:
                    seen.add(window)
    
    if len(seen) == target:
        return word
    return None

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    if n <= 4:
        return "".join(map(str, word))
    
    # For n=5,6 the classic is optimal
    if n <= 6:
        return "".join(map(str, word))
    
    # For n=7, try to shorten
    if time.time() < deadline - 20:
        word = _try_shorten(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
