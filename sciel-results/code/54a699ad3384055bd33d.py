<<<<<<< SEARCH
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
=======
    # Fast optimization using coverage counts
    w = list(word)
    if n <= 1:
        return "".join(map(str, w))
        
    # Map permutations to indices for fast lookup
    # We only care about valid permutations (distinct symbols)
    perm_to_idx = {}
    idx_to_perm = []
    all_perms = list(itertools.permutations(range(1, n+1)))
    for i, p in enumerate(all_perms):
        perm_to_idx[p] = i
        idx_to_perm.append(p)
        
    num_perms = len(all_perms)
    
    # count[k] is the number of windows in w that correspond to permutation k
    counts = [0] * num_perms
    
    def update_counts(w, start, end):
        """Update counts for windows starting in [start, end)"""
        for i in range(start, end):
            if 0 <= i <= len(w) - n:
                t = tuple(w[i:i+n])
                if t in perm_to_idx:
                    counts[perm_to_idx[t]] += 1
                    
    def decrement_counts(w, start, end):
        """Decrement counts for windows starting in [start, end)"""
        for i in range(start, end):
            if 0 <= i <= len(w) - n:
                t = tuple(w[i:i+n])
                if t in perm_to_idx:
                    counts[perm_to_idx[t]] -= 1

    # Initialize counts for the full word
    for i in range(len(w) - n + 1):
        t = tuple(w[i:i+n])
        if t in perm_to_idx:
            counts[perm_to_idx[t]] += 1
            
    # Track how many permutations are currently covered (count > 0)
    covered = sum(1 for c in counts if c > 0)
    
    # Try to remove characters
    # We iterate multiple times to allow cascading deletions
    # But for safety and time, we do a single pass from right to left
    # If we delete w[i], windows starting at i-n+1 to i are affected
    # We need to check if any permutation count drops to 0
    
    for i in range(len(w) - 2, 0, -1):
        # Affected windows: start indices i-n+1 to i
        # But note: windows starting at < 0 or > len(w)-n are invalid
        start_idx = max(0, i - n + 1)
        end_idx = min(i, len(w) - n)
        
        if start_idx > end_idx:
            continue
            
        # Store old counts for affected windows
        old_counts = []
        for j in range(start_idx, end_idx + 1):
            t = tuple(w[j:j+n])
            if t in perm_to_idx:
                idx = perm_to_idx[t]
                old_counts.append((j, idx, counts[idx]))
            else:
                old_counts.append((j, -1, 0))
                
        # Remove w[i]
        del w[i]
        
        # Check if any permutation becomes uncovered
        # We need to re-evaluate the windows that were affected
        # The windows that existed before and are still valid after deletion
        # Windows starting at j in [start_idx, i-1] are shifted? No, indices < i are unchanged.
        # Windows starting at j in [i, ...] are shifted left by 1.
        # Actually, deleting w[i] affects:
        # 1. Windows that included w[i] are broken or changed.
        # 2. Windows starting at i now start at w[i+1] (old w[i+1] is now at i).
        
        # Simpler: Just recompute the counts for the affected range in the new w
        # Affected range in new w:
        # Old windows starting at i-n+1 ... i
        # New windows starting at i-n+1 ... i-1 (these are the same as old windows starting at i-n+1 ... i-1? No)
        
        # Let's just recompute counts for the whole affected zone in the new w
        # Zone: [max(0, i-n+1), min(i-1, len(w)-n)]
        
        # This is getting complicated. Let's use a simpler heuristic:
        # Only try to delete if the character is not part of any "critical" permutation.
        # A permutation is critical if it appears exactly once.
        
        # Restore w[i] for now and use a simpler check
        w.insert(i, w[i] if i < len(w) else orig) # This logic is flawed, let's rollback properly
        
        # Rollback
        # We didn't actually delete in the logic above correctly. 
        # Let's restart the loop body with a correct deletion logic.
        
        pass # Placeholder to indicate logic change needed
>>>>>>> REPLACE
The above replacement is messy. I will provide a clean, efficient implementation of the local search.

