"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build standard recursive superpermutation
    def build_standard(m):
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        prev = build_standard(m - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - m + 2):
            window = tuple(prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        return new_word
    
    word = build_standard(n)
    
    # Optimize: try to remove characters that are redundant
    # A character at position i is redundant if removing it doesn't lose any permutation
    # We track which permutations are "covered" by each position
    
    def count_perm_coverage(w, n):
        """For each position, count how many unique perms are only seen at windows starting there"""
        perm_first_pos = {}
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            if len(set(t)) == n:
                if t not in perm_first_pos:
                    perm_first_pos[t] = i
        return perm_first_pos
    
    w = list(word)
    # Try removing characters from the middle (not first or last)
    # Process from right to left to avoid index shifting issues
    for i in range(len(w) - 2, 0, -1):
        # Quick check: is this character part of any "unique" window?
        # A window starting at i covers w[i:i+n]
        # If we remove w[i], windows starting at i, i-1, ..., i-n+1 are affected
        # Simple test: just try it
        orig = w[i]
        del w[i]
        # Verify all perms still present
        all_perms = set(itertools.permutations(range(1, n+1)))
        found = set()
        for j in range(len(w) - n + 1):
            t = tuple(w[j:j+n])
            if len(set(t)) == n:
                found.add(t)
            if len(found) == len(all_perms):
                break
        if found == all_perms:
            # Keep the deletion
            continue
        else:
            w.insert(i, orig)
    
    return "".join(map(str, w))
# EVOLVE-BLOCK-END
