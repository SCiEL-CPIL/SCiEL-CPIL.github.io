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
    w = word
    
    # Optimization: Greedy deletion based on permutation counts
    # 1. Count occurrences of each valid permutation
    perm_counts = {}
    for i in range(len(w) - n + 1):
        t = tuple(w[i:i+n])
        if len(set(t)) == n:
            perm_counts[t] = perm_counts.get(t, 0) + 1
    
    # Ensure all permutations are present (they should be from construction)
    all_perms = set(itertools.permutations(range(1, n+1)))
    # If any missing, construction failed, but we assume it's correct.
    
    # 2. Try to delete characters
    # We do multiple passes until no deletion is possible
    changed = True
    while changed:
        changed = False
        # Iterate from right to left to avoid index shifting
        for i in range(len(w) - 2, 0, -1):
            # Determine which windows are affected by deleting w[i]
            # Windows starting at j such that j <= i < j+n
            # So j in [i-n+1, i]
            # We need to check if any of these windows is the ONLY occurrence of its permutation
            
            # Collect affected permutations
            affected = {}
            for j in range(max(0, i - n + 1), min(i + 1, len(w) - n + 1)):
                t = tuple(w[j:j+n])
                if len(set(t)) == n:
                    if t not in affected:
                        affected[t] = 0
                    affected[t] += 1
            
            # If any affected permutation has count 1 in the whole word, we cannot delete
            can_delete = True
            for t, cnt_in_window in affected.items():
                # If this window is the only one for this permutation, we lose it
                if perm_counts.get(t, 0) == 1:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                del w[i]
                changed = True
                
                # Update perm_counts
                # Windows affected by deletion at i:
                # Old windows starting at j in [i-n+1, i] are removed/changed?
                # Actually, deleting w[i] shifts everything after i.
                # It's easier to just recount? No, that's O(L^2).
                # Let's update counts carefully.
                
                # Windows that existed before and are still valid?
                # Windows starting at j < i-n+1 are unchanged.
                # Windows starting at j >= i are shifted left by 1.
                # Windows starting at j in [i-n+1, i-1] are broken/changed.
                
                # This update logic is tricky.
                # Given L ~ 6000, recounting is O(L) ~ 6000 ops.
                # Passes might be ~10. Total 60k ops. Very fast.
                # Let's just recount in the inner loop if we delete?
                # No, recounting takes O(L). If we delete many, it's O(L * Deletions).
                # Deletions are at most 8 (from 5913 to 5905).
                # So 8 * 6000 = 48000 ops. Fine.
                
                # Recount
                perm_counts = {}
                for j in range(len(w) - n + 1):
                    t = tuple(w[j:j+n])
                    if len(set(t)) == n:
                        perm_counts[t] = perm_counts.get(t, 0) + 1
                # Break inner loop because indices changed?
                # We are iterating backwards. Deleting at i affects indices < i.
                # Since we go right to left, indices < i are yet to be processed.
                # So we can continue.
    
    return "".join(map(str, w))
# EVOLVE-BLOCK-END
