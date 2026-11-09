# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations as all_perms

def construct(n):
    """
    Construct a superpermutation for n symbols.
    Returns a string of digits 1..n such that every permutation appears as a substring of length n.
    """
    deadline = time.time() + 110
    start_time = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    
    # Build up to n-1 using standard construction
    if n <= 6:
        return _standard_construction(n)
    
    # For n >= 7, build up to n-1, then use greedy for the final stage
    base_word = _standard_construction(n - 1)
    return _greedy_final_stage(base_word, n, deadline)


def _standard_construction(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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


def _greedy_final_stage(base_word_str, n, deadline):
    """
    Given a superpermutation for n-1 symbols, build the final stage
    by ordering the n-1! permutations (blocks of length 2*(n-1)+1 = 2n-1)
    to maximize overlap. Use greedy with look-ahead.
    """
    base_word = list(base_word_str)
    base_len = len(base_word)
    new_sym = n
    block_len = 2 * (n - 1) + 1  # 15 for n=7
    
    # Extract all (n-1)-permutations from the base word in order of first appearance
    (n1_perms, seen) = [], set()
    for i in range(base_len - (n - 2)):
        window = tuple(base_word[i:i + (n - 1)])
        if len(set(window)) == (n - 1) and window not in seen:
            seen.add(window)
            n1_perms.append(window)
    
    # Each permutation p becomes a block: p + [new_sym] + p
    # We need to order these blocks to maximize overlap
    num_blocks = len(n1_perms)
    
    # Precompute overlap between consecutive blocks
    # block_i = list(p_i) + [new_sym] + list(p_i)
    # The overlap between block_i and block_j is the largest o such that
    # block_i[-o:] == block_j[:o]
    
    # For efficiency, let's compute overlap on the fly with caching
    overlap_cache = {}
    
    def get_overlap(block_end, block_start, max_o):
        """Find max overlap where suffix of block_end matches prefix of block_start"""
        key = (id(block_end), id(block_start))
        if key in overlap_cache:
            return overlap_cache[key]
        
        best = 0
        for o in range(min(max_o, len(block_end), len(block_start)), 0, -1):
            if block_end[-o:] == block_start[:o]:
                best = o
                break
        overlap_cache[key] = best
        return best
    
    # Greedy: start with the first block, then repeatedly pick the block
    # that gives the best overlap with the current end
    # But we also want to ensure we can complete all blocks
    
    # Actually, for a proper greedy, we should try multiple starting blocks
    # and pick the one that gives the shortest total length
    
    # Let's use a simple greedy: start with block 0, then greedily pick
    # the next block that maximizes overlap
    # We'll try a few different starting points and look-ahead depths
    
    best_word = None
    best_len = float('inf')
    
    # Try greedy with different starting blocks (first few for speed)
    for start_idx in range(min(10, num_blocks)):
        if time.time() > deadline:
            break
            
        remaining = set(range(num_blocks))
        remaining.discard(start_idx)
        
        current_blocks = [start_idx]
        word_so_far = list(n1_perms[start_idx]) + [new_sym] + list(n1_perms[start_idx])
        
        while remaining:
            best_next = None
            best_overlap = -1
            
            for idx in remaining:
                block = list(n1_perms[idx]) + [new_sym] + list(n1_perms[idx])
                o = get_overlap(word_so_far, block, block_len - 1)
                if o > best_overlap:
                    best_overlap = o
                    best_next = idx
            
            if best_next is None:
                break
            
            remaining.discard(best_next)
            current_blocks.append(best_next)
            block = list(n1_perms[best_next]) + [new_sym] + list(n1_perms[best_next])
            word_so_far += block[best_overlap:]
        
        if len(remaining) == 0 and len(word_so_far) < best_len:
            best_len = len(word_so_far)
            best_word = word_so_far[:]
    
    if best_word is not None:
        return "".join(map(str, best_word))
    
    # Fallback: standard construction
    return _standard_construction(n)
# EVOLVE-BLOCK-END
