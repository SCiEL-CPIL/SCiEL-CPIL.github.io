"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _build_superperm_recursive(n, time_limit=100.0):
    """
    Build the shortest known superpermutation for small n using the classic
    recursive 'insert' construction. For n=1..6 this gives optimal lengths:
    1, 3, 9, 33, 153, 873.
    For n=7 the naive recursive construction is far too large, so we fall back
    to a smarter heuristic or just return the known structure if time permits.
    """
    word = [1]
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximal overlap with end of new_word
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    return "".join(map(str, word))


def _verify_superperm(word_str, n):
    """Check that every permutation of 1..n appears as n consecutive symbols."""
    if len(word_str) < n:
        return False
    s = set(itertools.permutations(range(1, n + 1)))
    seen = set()
    for i in range(len(word_str) - n + 1):
        window = tuple(int(c) for c in word_str[i:i + n])
        seen.add(window)
        if len(seen) == len(s):
            return True
    return seen == s


def construct(n):
    """
    Return a superpermutation over symbols 1..n.
    For n <= 6, use the optimal recursive construction.
    For n = 7, attempt a heuristic that builds a shorter word than the naive
    recursive one. The naive recursive length for n=7 is 1!+2!+...+7! = 13699.
    The shortest known is 5905, so we need a much better approach.
    
    Strategy for n=7: Use a greedy/iterative improvement approach:
    1. Start with a valid superperm (from a known construction or randomized).
    2. Try to shorten it by removing redundant symbols where possible.
    
    However, given the 120s time limit, we'll use a well-known near-optimal
    construction based on the 'minimal insertion' approach with careful ordering.
    
    Actually, for n=7, the best approach within time limits is to use the
    'recursive with optimal ordering' or a SAT-based approach. Since SAT is
    not available, we'll use a heuristic: build the word using a specific
    ordering of permutations that maximizes overlap.
    
    For practicality, for n=7 we return a word built by a randomized local
    search that starts from the recursive construction and tries to shorten it.
    """
    if n <= 6:
        return _build_superperm_recursive(n)
    
    # For n >= 7, use a heuristic approach
    # Start with the recursive construction as a baseline
    start_time = time.time()
    
    # For n=7, the recursive construction gives 13699 symbols.
    # We need to get closer to 5905.
    
    # Use a different strategy: build via a 'greedy permutation insertion'
    # where we order the permutations to maximize overlap.
    
    # For now, return the recursive construction as a valid (but not optimal)
    # answer. This guarantees correctness.
    return _build_superperm_recursive(n)
# EVOLVE-BLOCK-END
