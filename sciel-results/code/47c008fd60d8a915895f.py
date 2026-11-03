"""Improved superpermutation construction using a more efficient approach."""
import time
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    """
    Construct a shortest-known superpermutation for n symbols.
    
    For n <= 4, we can use the known optimal constructions.
    For n = 5, 6, 7, we use the classical PnP (permutation-next-permutation) 
    construction which gives length 1! + 2! + ... + n!, but we try to optimize
    the overlap.
    
    The classical construction for n symbols has length:
      n=1: 1
      n=2: 3 (121)
      n=3: 9 (123121321)
      n=4: 33
      n=5: 153
      n=6: 873
      n=7: 5913 (classical), but the shortest known is 5905
    
    We implement the classical recursive construction with maximum overlap,
    and for n=7 we try to find a slightly shorter word if time permits.
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    if n <= 1:
        return "1"
    
    # For small n, use precomputed optimal or near-optimal words
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classical construction gives 33, which is optimal
        return _classic_superperm(4)
    
    # For n = 5, 6, 7, use the classical PnP construction
    # This gives 1!+2!+...+n! which for n=7 is 5913
    # The known shortest is 5905, so we try to improve if time allows
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 2:
        return word
    
    # Try to improve the word by removing redundant characters
    # This is a heuristic local search
    if n <= 7 and (deadline - time.time()) > 5:
        word = _try_improve(word, n, deadline)
    
    return word


def _classic_superperm(n):
    """
    Classical recursive construction of superpermutation.
    Length is 1! + 2! + ... + n!
    """
    word = [1]
    for m in range(2, n + 1):
        # Find all (m-1)-permutations that appear in the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m after each permutation
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))


def _try_improve(word_str, n, deadline):
    """
    Try to improve the superpermutation by removing redundant characters.
    This is a heuristic approach that tries to find shorter valid words.
    """
    word = list(map(int, word_str))
    original_len = len(word)
    
    # Verify it's a valid superpermutation first
    if not _is_valid_superperm(word, n):
        return word_str
    
    # Try local search: for each position, try to remove the character
    # and check if the result is still valid
    improved = True
    while improved and (time.time() < deadline - 2):
        improved = False
        for i in range(1, len(word) - 1):  # Don't remove first or last
            # Try removing position i
            candidate = word[:i] + word[i+1:]
            if len(candidate) < n:
                continue
            if _is_valid_superperm(candidate, n):
                word = candidate
                improved = True
                break
    
    return "".join(map(str, word))


def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
    
    return len(seen) == _factorial(n)


def _factorial(n):
    """Compute n!"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
# EVOLVE-BLOCK-END
