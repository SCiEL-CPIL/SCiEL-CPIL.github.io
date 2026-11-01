import random
import time
import sys
from itertools import permutations
import math

# EVOLVE-BLOCK-START

def construct(n):
    """
    Construct a superpermutation of 1..n using fixed-symbol anchor decomposition.
    
    Partition n! permutations into n groups by position of symbol n.
    Each group is a superpermutation of the remaining n-1 symbols with
    anchor inserted at the fixed position. Optimize overlaps between groups.
    """
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    # Build the standard superpermutation for n-1 symbols
    def build_superperm(m):
        """Build standard recursive superpermutation for symbols 1..m."""
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        word = build_superperm(m - 1)
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    # Build superpermutation for n-1 symbols
    base_sp = build_superperm(n - 1)
    base_symbols = list(range(1, n))  # 1..n-1
    
    # For each position pos (0-indexed) of symbol n, build a block
    # that covers all permutations where n is at position pos
    def build_anchor_block(pos, anchor_sym):
        """
        Build a word containing all permutations of 1..n where anchor_sym
        is at position pos (0-indexed). The other n-1 symbols form a
        superpermutation pattern with the anchor inserted.
        """
        # We need to cover all (n-1)! permutations of the remaining symbols
        # with anchor_sym at position pos
        remaining_syms = [s for s in range(1, n + 1) if s != anchor_sym]
        
        # Get all permutations of remaining symbols
        perms = list(permutations(remaining_syms))
        
        # For each permutation, create the full permutation with anchor at pos
        blocks = []
        for perm in perms:
            full_perm = list(perm[:pos]) + [anchor_sym] + list(perm[pos:])
            blocks.append(full_perm)
        
        # Now we need to concatenate these blocks with maximum overlap
        # This is like a shortest common superstring problem
        result = []
        for block in blocks:
            if not result:
                result = list(block)
                continue
            # Find maximum overlap
            max_ov = min(len(result), len(block))
            best_ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    best_ov = 0
                    break
                if result[len(result) - o:] == block[:o]:
                    best_ov = o
                    break
            result += block[best_ov:]
        
        return result
    
    # Build blocks for each position of the anchor symbol
    anchor_sym = n
    blocks = []
    for pos in range(n):
        if time.time() > deadline:
            break
        blocks.append(build_anchor_block(pos, anchor_sym))
    
    # If we ran out of time, fall back to standard construction
    if len(blocks) < n:
        word = build_superperm(n)
        return "".join(map(str, word))
    
    # Now concatenate all n blocks with optimal ordering and maximum overlap
    # Use greedy: try all starting positions, then greedily add blocks
    def max_overlap_suffix_prefix(a, b):
        """Find maximum overlap where suffix of a matches prefix of b."""
        max_ov = min(len(a), len(b))
        for o in range(max_ov, -1, -1):
            if o == 0:
                return 0
            if a[len(a) - o:] == b[:o]:
                return o
        return 0
    
    def concatenate_blocks(block_list):
        """Concatenate blocks with maximum overlap, trying different orderings."""
        best_result = None
        best_len = float('inf')
        
        # Try all permutations of block order (n is small, at most 7)
        # But n! = 5040 is manageable
        for order in permutations(range(len(block_list))):
            if time.time() > deadline:
                break
            result = []
            for idx in order:
                block = block_list[idx]
                if not result:
                    result = list(block)
                    continue
                ov = max_overlap_suffix_prefix(result, block)
                result += block[ov:]
            
            if len(result) < best_len:
                best_len = len(result)
                best_result = result
        
        return best_result if best_result else block_list[0]
    
    word = concatenate_blocks(blocks)
    
    # Local search to try to reduce length further
    if time.time() < deadline - 10:
        word = _local_search(word, n, time_budget=int(deadline - time.time() - 5))
    
    return "".join(map(str, word))

def _local_search(word, n, time_budget=115):
    """Try to shorten the word while maintaining superpermutation property."""
    start_time = time.time()
    word = list(word)
    
    target = math.factorial(n)
    
    def count_perms(w):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen)
    
    current_count = count_perms(word)
    if current_count < target:
        return word  # Not valid, return as-is
    
    # Greedy deletion: repeatedly try to delete symbols
    for outer_iter in range(200):
        if time.time() - start_time > time_budget:
            break
        
        if len(word) <= n:
            break
        
        found_deletion = False
        
        # Try random positions
        for attempt in range(100):
            if time.time() - start_time > time_budget:
                break
            
            pos = random.randint(1, len(word) - 2)
            new_word = word[:pos] + word[pos+1:]
            
            if count_perms(new_word) >= target:
                word = new_word
                found_deletion = True
                break
        
        if not found_deletion:
            # Try sequential scan
            mid = len(word) // 2
            for offset in range(1, len(word) // 2):
                if time.time() - start_time > time_budget:
                    break
                
                for pos in [mid - offset, mid + offset]:
                    if pos < 1 or pos > len(word) - 2:
                        continue
                    new_word = word[:pos] + word[pos+1:]
                    if count_perms(new_word) >= target:
                        word = new_word
                        found_deletion = True
                        break
                if found_deletion:
                    break
        
        if not found_deletion:
            if outer_iter > 30:
                break
    
    return word

# EVOLVE-BLOCK-END
