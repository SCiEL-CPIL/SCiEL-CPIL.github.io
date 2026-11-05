"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    # Generate all permutations of 1..n
    symbols = list(range(1, n + 1))
    perms = [''.join(map(str, p)) for p in itertools.permutations(symbols)]
    
    # Pre-compute prefix index for fast overlap checking
    # prefix_index[prefix] = list of permutations starting with prefix
    prefix_index = {}
    for p in perms:
        for i in range(1, n + 1):
            prefix = p[:i]
            if prefix not in prefix_index:
                prefix_index[prefix] = []
            prefix_index[prefix].append(p)
    
    uncovered = set(perms)
    word = ""
    
    while uncovered:
        best_perm = None
        best_overlap = 0
        
        # Check for maximum overlap with the current suffix
        # Overlap can be at most n (if the next perm is fully contained in the suffix, 
        # but since we remove covered perms, this usually means we just need to extend)
        # Actually, if a perm is fully contained in the current word, it should have been removed.
        # So we look for overlaps of length k where 1 <= k < n.
        # If k=n, the perm is already in the word? 
        # The check "if window is a permutation" in the task implies we check all windows.
        # If we append a perm, we ensure it's in the word.
        
        # We iterate overlap length from n-1 down to 1
        # If overlap is n, the perm is a substring of the current word.
        # We should have removed it when it was formed.
        # However, due to how we append, it's possible a perm is formed by the overlap of two appended parts.
        # To be safe, we check if any uncovered perm is a substring of the current word.
        # This is expensive. 
        # Instead, the standard SCS greedy assumes we only append non-overlapping parts.
        
        # Let's check overlaps from n-1 down to 1.
        for k in range(n - 1, 0, -1):
            if len(word) < k:
                continue
            suffix = word[-k:]
            if suffix in prefix_index:
                candidates = prefix_index[suffix]
                for p in candidates:
                    if p in uncovered:
                        best_perm = p
                        best_overlap = k
                        break
            if best_perm:
                break
        
        if not best_perm:
            # No overlap found, pick any uncovered perm
            # To be deterministic, pick the first one in sorted order or arbitrary
            best_perm = next(iter(uncovered))
            best_overlap = 0
            
        # Append the non-overlapping part
        word += best_perm[best_overlap:]
        uncovered.remove(best_perm)
        
        # Optimization: Check if any other uncovered perms are now fully contained in the new word
        # This is O(N * n) per step, which might be slow (5040 * 7 * 5040 ~ 176M ops).
        # We can skip this for speed, as the greedy selection tends to cover them soon.
        # Or we can do a cheap check:
        # Only check perms that share a prefix with the newly added part?
        # For simplicity and speed, we skip this. The greedy step usually handles it.
        
    return word
# EVOLVE-BLOCK-END
