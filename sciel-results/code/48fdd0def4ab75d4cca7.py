"""Improved superpermutation construction using better overlap strategies."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def _overlap_len(suffix_src, block):
    """Max o such that suffix_src[-o:] == block[:o], or 0."""
    max_ov = min(len(suffix_src), len(block))
    for o in range(max_ov, -1, -1):
        if o == 0 or suffix_src[-o:] == block[:o]:
            return o
    return 0

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    # The classic method gives sum_{k=1}^n k! which for n=7 is 5916
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        new = []
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with current end of new
            ov = _overlap_len(new, block)
            new += block[ov:]
        
        word = new
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
