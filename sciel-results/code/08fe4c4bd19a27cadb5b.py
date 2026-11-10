import itertools
import random

def construct(n):
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute prefix maps
    # prefix_map[(k, prefix)] = set of indices
    prefix_map = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if (k, prefix) not in prefix_map:
                prefix_map[(k, prefix)] = set()
            prefix_map[(k, prefix)].add(i)
    
    visited = set()
    # Start with the first permutation (1..n)
    current_idx = 0
    visited.add(0)
    
    # Remove current_idx from all prefix maps
    p = perms[current_idx]
    for k in range(1, n):
        prefix = p[:k]
        prefix_map[(k, prefix)].discard(current_idx)
    
    word = list(p)
    
    for _ in range(N - 1):
        next_idx = -1
        overlap_len = 0
        
        # Try to find the next permutation with maximum overlap
        # Check k from n-1 down to 1
        for k in range(n - 1, 0, -1):
            suffix = perms[current_idx][n - k:]
            key = (k, suffix)
            candidates = prefix_map.get(key, set())
            
            if candidates:
                # Pick an arbitrary candidate
                # Using next(iter()) is O(1)
                next_idx = next(iter(candidates))
                overlap_len = k
                break
        
        if next_idx == -1:
            # Fallback: pick any unvisited permutation
            # This should rarely happen if the graph is connected
            # Find an unvisited index
            for i in range(N):
                if i not in visited:
                    next_idx = i
                    break
            if next_idx == -1:
                break # Should not happen
            overlap_len = 0 # No overlap
        
        # Add to visited
        visited.add(next_idx)
        
        # Remove next_idx from all prefix maps
        p_next = perms[next_idx]
        for k in range(1, n):
            prefix = p_next[:k]
            key = (k, prefix)
            if key in prefix_map:
                prefix_map[key].discard(next_idx)
        
        # Append the non-overlapping part
        word.extend(perms[next_idx][overlap_len:])
        
        current_idx = next_idx
        
    return "".join(map(str, word))
