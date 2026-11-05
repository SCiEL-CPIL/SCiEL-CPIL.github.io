"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    if n <= 1:
        return '1' * n
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Build suffix-to-perm-index mapping for fast lookup
    # suffix_to_perm[suffix] = set of perm indices that start with this suffix
    suffix_to_perm = {}
    for idx, p in enumerate(perms):
        for k in range(1, n):
            suffix = tuple(p[:k])
            if suffix not in suffix_to_perm:
                suffix_to_perm[suffix] = []
            suffix_to_perm[suffix].append(idx)
    
    uncovered = set(range(num_perms))
    current_suffix = ()  # empty tuple initially
    word_chars = []
    
    def get_suffix(k):
        if k <= 0:
            return ()
        if len(word_chars) < k:
            return tuple(word_chars)
        return tuple(word_chars[-k:])
    
    while uncovered:
        best_perm_idx = None
        best_overlap = 0
        
        # Check for maximum overlap with the current suffix
        for k in range(n - 1, 0, -1):
            if len(word_chars) < k:
                continue
            suffix = tuple(word_chars[-k:])
            if suffix in suffix_to_perm:
                candidates = suffix_to_perm[suffix]
                for pidx in candidates:
                    if pidx in uncovered:
                        best_perm_idx = pidx
                        best_overlap = k
                        break
            if best_perm_idx is not None:
                break
        
        if best_perm_idx is None:
            # No overlap found, pick any uncovered perm
            best_perm_idx = next(iter(uncovered))
            best_overlap = 0
        
        # Append the non-overlapping part
        p = perms[best_perm_idx]
        to_add = p[best_overlap:]
        word_chars.extend(to_add)
        uncovered.remove(best_perm_idx)
        
        # Check if any other uncovered perms are now fully contained in the new word
        # Only check perms that could be formed by the recent additions
        # The last window of length n is at position len(word_chars) - n
        # Check all windows that include at least one of the newly added characters
        start_check = max(0, len(word_chars) - n - len(to_add) + 1)
        end_check = len(word_chars) - n + 1
        
        if end_check > start_check:
            for i in range(start_check, end_check):
                window = tuple(word_chars[i:i+n])
                # Check if this window is a permutation
                if len(set(window)) == n and all(1 <= x <= n for x in window):
                    # Find which perm this is
                    # We need to map window to perm index
                    # This is expensive, so let's use a different approach
                    pass
        
    return ''.join(map(str, word_chars))
# EVOLVE-BLOCK-END
