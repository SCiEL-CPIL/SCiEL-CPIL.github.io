"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import heapq
import random

def construct(n):
    if n <= 1:
        return '1' * n
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Map each permutation tuple to its index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # For each suffix of length k (1 <= k < n), store the list of perm indices
    # that have this suffix as their last k elements
    # suffix_to_perms[suffix] = list of perm indices
    suffix_to_perms = {k: {} for k in range(1, n)}
    for idx, p in enumerate(perms):
        for k in range(1, n):
            suffix = p[n-k:]
            if suffix not in suffix_to_perms[k]:
                suffix_to_perms[k][suffix] = []
            suffix_to_perms[k][suffix].append(idx)
    
    # Also build prefix_to_perms for potential use
    prefix_to_perms = {k: {} for k in range(1, n)}
    for idx, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_perms[k]:
                prefix_to_perms[k][prefix] = []
            prefix_to_perms[k][prefix].append(idx)
    
    # Greedy construction with overlap
    # We build the word by always appending a permutation that maximizes overlap
    # with the current suffix of the word
    
    uncovered = set(range(num_perms))
    word = []
    
    # Start with the first permutation
    first_perm = perms[0]
    word.extend(first_perm)
    uncovered.remove(0)
    
    # Check if any other permutations are covered by the initial word
    # Only possible if n == 1, but we handle n >= 2 here
    # For n >= 2, only the first permutation is covered
    
    while uncovered:
        current_suffix = tuple(word[-(n-1):]) if len(word) >= n-1 else tuple(word)
        
        best_perm_idx = None
        best_overlap = 0
        best_perm = None
        
        # Try to find a permutation with maximum overlap
        # Overlap k means the last k elements of the word match the first k elements of the perm
        # Equivalently, the last k elements of the word are a prefix of the perm
        
        # Try from maximum possible overlap down to 1
        max_overlap = min(n-1, len(word))
        for k in range(max_overlap, 0, -1):
            if k > len(word):
                continue
            prefix_needed = tuple(word[-k:])
            # Find permutations that start with this prefix
            if k in prefix_to_perms and prefix_needed in prefix_to_perms[k]:
                candidates = prefix_to_perms[k][prefix_needed]
                for pidx in candidates:
                    if pidx in uncovered:
                        best_perm_idx = pidx
                        best_overlap = k
                        best_perm = perms[pidx]
                        break
            if best_perm_idx is not None:
                break
        
        if best_perm_idx is None:
            # No overlap found, pick any uncovered perm
            # Use a heuristic: pick the one that has the most uncovered neighbors
            # to maximize future coverage
            best_perm_idx = min(uncovered, key=lambda i: (len(prefix_to_perms.get(1, {}).get(perms[i][:1], [])) if 1 in prefix_to_perms else 0))
            best_perm = perms[best_perm_idx]
            best_overlap = 0
        
        # Append the non-overlapping part
        to_add = best_perm[best_overlap:]
        word.extend(to_add)
        uncovered.remove(best_perm_idx)
        
        # Check for other newly covered permutations
        # New windows are those that include at least one of the newly added characters
        # The last window starts at len(word) - n
        # We need to check windows that end in the newly added region
        start_pos = len(word) - len(to_add)
        for end_pos in range(start_pos, len(word)):
            start = end_pos - n + 1
            if start < 0:
                continue
            window = tuple(word[start:end_pos+1])
            if window in perm_to_idx:
                widx = perm_to_idx[window]
                if widx in uncovered:
                    uncovered.remove(widx)
    
    return ''.join(map(str, word))
# EVOLVE-BLOCK-END
