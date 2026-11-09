"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import permutations

def construct(n):
    # Standard recursive construction
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
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    s = "".join(map(str, word))
    
    # Local optimization: try removing each symbol and check validity
    # Only for n <= 7 to keep time reasonable
    if n <= 7:
        # Precompute set of valid permutation strings
        all_perms_str = set(''.join(map(str, p)) for p in permutations(range(1, n + 1)))
        
        def is_valid(ss):
            if len(ss) < n:
                return False
            # Use a set to track found permutations to avoid duplicates in check
            # But we just need to know if all are present.
            # Iterating and checking membership is O(L * n) for slicing, O(1) for set lookup.
            found = set()
            for i in range(len(ss) - n + 1):
                w = ss[i:i+n]
                if w in all_perms_str:
                    found.add(w)
            return len(found) == len(all_perms_str)
        
        # Try single-symbol removals repeatedly
        improved = True
        while improved:
            improved = False
            for pos in range(len(s) - 1):
                # Optimization: Skip if removing 'pos' doesn't affect any permutation?
                # Hard to check quickly. Just try.
                trial = s[:pos] + s[pos+1:]
                if is_valid(trial):
                    s = trial
                    improved = True
                    break
    
    return s
# EVOLVE-BLOCK-END
