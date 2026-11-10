"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def _build_superperm(n):
    """Build a superpermutation using recursive construction with overlap optimization."""
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        # For each perm P, the block is P + [m] + P
        # We want to maximize overlap between consecutive blocks
        new = []
        for idx, p in enumerate(perms):
            block = list(p) + [m] + list(p)
            if not new:
                new = block
            else:
                # Find maximum overlap: suffix of new matches prefix of block
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, 0, -1):
                    if new[-o:] == block[:o]:
                        overlap = o
                        break
                new.extend(block[overlap:])
        word = new
    return word

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Build the standard recursive superpermutation
    word = _build_superperm(n)
    
    # Try to improve by local search: look for redundant patterns
    # A pattern like "abcXabc" where X is a single symbol can sometimes be shortened
    # if the surrounding context allows it.
    
    # For n=7, try to find savings in the final layer
    # The idea: when we have ...P+m+P...Q+m+Q... where P and Q are (m-1)-perms,
    # if P and Q share a long prefix/suffix, we might be able to merge.
    
    if n >= 5:
        # Try greedy improvement: scan for patterns that can be shortened
        # Pattern: if we see X Y X where Y is a single new symbol and the context
        # allows removing one symbol, do it.
        # More practically: look for "abca" patterns where 'a' appears and we can
        # sometimes remove the middle if it's redundant.
        
        # A known trick: in the superpermutation, look for substrings of the form
        # "xy...z" where removing one character still preserves all permutations.
        # This is expensive to verify, so we use a heuristic.
        
        # For n=7, the main saving opportunity is in how the 7s are placed.
        # Let's try a different construction for the last step.
        
        # Build up to n-1
        word_n1 = _build_superperm(n - 1)
        
        # Now insert n into word_n1 with optimized overlaps
        m = n
        perms = []
        seen = set()
        for i in range(len(word_n1) - (m - 1) + 1):
            window = tuple(word_n1[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Standard construction
        standard_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            if not standard_word:
                standard_word = block
            else:
                max_ov = min(len(standard_word), len(block))
                ov = 0
                for o in range(max_ov, 0, -1):
                    if standard_word[-o:] == block[:o]:
                        ov = o
                        break
                standard_word.extend(block[ov:])
        
        # Try to find a better ordering or insertion strategy
        # Heuristic: if two consecutive perms P and Q have P[-k:] == Q[:k] for large k,
        # the overlap is already maximal. The issue is when overlap is small.
        
        # Alternative: try reversing the order of some permutations
        # This is complex; for now, use the standard construction which gives 5913.
        
        # Check if we're within time budget to try improvements
        if time.time() < deadline - 5:
            # Try a simple local optimization: 
            # Look for "abXab" where X is a single char and ab is part of a permutation.
            # If we can remove X and still have all (n)-perms, do it.
            # This is hard to verify efficiently, so skip for now.
            pass
        
        word = standard_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
