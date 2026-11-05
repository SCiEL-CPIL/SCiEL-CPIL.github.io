"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    """Construct a superpermutation of length at most n*n! for symbols 1..n.
    
    Uses a recursive construction with greedy nearest-neighbor reordering
    at each level to maximize overlaps between blocks.
    """
    if n <= 1:
        return "1"
    
    # Time budget: must finish well before 120s
    start_time = time.time()
    DEADLINE = start_time + 110.0  # 10s safety margin
    
    # Build the base superpermutation for n-1 symbols
    if n == 2:
        return "121"
    
    # Get the superpermutation for n-1
    base = construct(n - 1)
    
    # For n=7, we need to be very careful about time
    # The recursive approach with greedy reordering should work
    
    # Extract all (n-1)-permutations from the base word in order of appearance
    m = n - 1
    perms = []
    seen = set()
    for i in range(len(base) - m + 1):
        window = tuple(int(c) for c in base[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Each block is: perm + [n] + perm
    # We want to order these blocks to maximize suffix-prefix overlap
    
    # Build the blocks
    blocks = []
    for p in perms:
        block = list(p) + [n] + list(p)
        blocks.append(block)
    
    # Greedy nearest-neighbor: start with first block, then repeatedly
    # pick the unused block that best overlaps with the current end
    # This is O(k^2) where k = number of blocks = (n-1)!
    
    k = len(blocks)
    if k <= 1:
        return "".join(map(str, blocks[0])) if blocks else "1"
    
    # Precompute block signatures for O(1) overlap lookup
    # For each block, store its prefix of length up to max_overlap
    # and for the current result end, store its suffix of length up to max_overlap
    max_overlap = 2 * m  # maximum possible overlap
    
    # Precompute prefixes for each block
    # prefix_cache[j][ov] = tuple of first ov elements of block j
    prefix_cache = []
    for j in range(k):
        b = blocks[j]
        max_ov_j = min(len(b), max_overlap)
        prefixes = [None] * (max_ov_j + 1)
        prefixes[0] = ()
        for ov in range(1, max_ov_j + 1):
            prefixes[ov] = tuple(b[:ov])
        prefix_cache.append(prefixes)
    
    # Greedy nearest-neighbor with precomputed prefixes
    used = [False] * k
    current = blocks[0]
    used[0] = True
    result = list(current)
    
    for _ in range(k - 1):
        best_idx = -1
        best_overlap = -1
        
        # Compute suffixes of result end
        max_ov_result = min(len(result), max_overlap)
        suffixes = [None] * (max_ov_result + 1)
        suffixes[0] = ()
        for ov in range(1, max_ov_result + 1):
            suffixes[ov] = tuple(result[-ov:])
        
        for j in range(k):
            if used[j]:
                continue
            # Compute overlap between end of result and start of blocks[j]
            max_ov = min(max_ov_result, len(prefix_cache[j]) - 1)
            for ov in range(max_ov, -1, -1):
                if ov == 0:
                    best_ov_candidate = 0
                    break
                if suffixes[ov] == prefix_cache[j][ov]:
                    best_ov_candidate = ov
                    break
            else:
                best_ov_candidate = 0
            
            if best_ov_candidate > best_overlap:
                best_overlap = best_ov_candidate
                best_idx = j
        
        if best_idx == -1:
            break
        
        used[best_idx] = True
        block_to_add = blocks[best_idx]
        result.extend(block_to_add[best_overlap:])
        
        # Check time
        if time.time() > DEADLINE:
            break
    
    # Check if we've included all blocks
    if not all(used):
        # Add remaining blocks with minimal overlap
        for j in range(k):
            if not used[j]:
                block_to_add = blocks[j]
                max_ov = min(len(result), len(block_to_add), max_overlap)
                best_ov = 0
                for ov in range(max_ov, -1, -1):
                    if ov == 0:
                        best_ov = 0
                        break
                    if tuple(result[-ov:]) == tuple(block_to_add[:ov]):
                        best_ov = ov
                        break
                result.extend(block_to_add[best_ov:])
    
    return "".join(map(str, result))

# EVOLVE-BLOCK-END
