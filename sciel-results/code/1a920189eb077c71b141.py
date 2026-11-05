import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # --- 1. Generate Baseline Recursive Superpermutation ---
    def recursive_superperm(n_val):
        word = [1]
        for m in range(2, n_val + 1):
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
                # Find max overlap between end of new and start of block
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    baseline_word = recursive_superperm(n)
    baseline_len = len(baseline_word)
    
    # If n is small, the baseline is likely optimal or near-optimal and fast enough.
    # For n=7, baseline is 13699. We want to improve.
    # For n<=4, we can just return baseline as it's very small and hard to beat significantly in 110s.
    if n <= 4:
        return "".join(map(str, baseline_word))

    S = baseline_word[:]
    best_S = S[:]
    best_len = len(S)

    # --- 2. Local Search: Removal Passes ---
    # Precompute counts of permutations in current S
    def build_counts(word, n_val):
        counts = {}
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            counts[p] = counts.get(p, 0) + 1
        return counts

    counts = build_counts(S, n)
    
    # Optimization: Precompute permutation indices? 
    # For n=7, 5040 perms. Tuples of 7 ints are hashable and reasonably fast.
    # We can map tuples to IDs for faster comparison if needed, but let's start simple.
    
    # To speed up removal checks, we can maintain a list of "critical" windows (count == 1).
    # If we remove a char, we only invalidate windows that include that char.
    # A window is critical if count==1.
    # If any critical window is destroyed, the removal is invalid.
    
    # Let's use a simpler approach first: check validity by iterating affected windows.
    # Affected windows for removal at index i:
    # Old windows: [i-n+1, i]
    # New windows: [i-n+1, i-1] (shifted)
    
    # Actually, the most efficient way for n=7 is:
    # 1. Identify all windows with count 1.
    # 2. For a removal at i, check if any window with count 1 is among the old windows.
    # 3. If not, the removal is valid.
    
    # Let's maintain a set of critical window indices.
    # critical_windows: set of indices i such that count(window[i]) == 1
    # Note: multiple windows can have count 1.
    
    def get_critical_indices(counts, word, n_val):
        crit = set()
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            if counts[p] == 1:
                crit.add(i)
        return crit

    critical_indices = get_critical_indices(counts, S, n)
    
    stuck_passes = 0
    max_stuck_passes = 50 # Allow some randomization attempts if stuck
    
    while time.time() < deadline:
        removed_in_pass = 0
        
        # Iterate from right to left to avoid index shifting issues for left part
        # But we need to re-calculate critical indices or update them efficiently.
        # Updating critical indices dynamically is complex.
        # Given n=7 and ~6000 length, full rescan of critical indices after each removal
        # is O(L*n) which is ~42000 ops. If we do 1000 removals, that's 42M ops.
        # Python might be too slow for 1000 full rescans in 110s?
        # Let's try to be smarter or just accept the cost if removals are few.
        # Actually, removals will be few because the baseline is already dense.
        # Let's try the simple check:
        # For index i (from len-1 down to 0):
        #   Check if any critical window covers i.
        #   A window covers i if i is in [j, j+n-1] for some critical j.
        #   So i is covered if there exists j in critical_indices such that j <= i <= j+n-1.
        #   This is equivalent to: min(critical_indices in [i-n+1, i]) exists.
        
        # To make this O(1) or O(log N), we can sort critical indices.
        # Or just iterate? Critical indices are likely sparse?
        # In a superpermutation, most windows appear exactly once?
        # No, in the recursive construction, many windows appear multiple times.
        # But let's assume we can check this reasonably.
        
        # Let's use a boolean array or a set for fast lookup?
        # If we have a set of critical indices, checking if any j in [i-n+1, i] is in the set
        # requires checking up to n values. n=7. So O(7) per index.
        # Total O(7 * L) per pass. L ~ 6000. 42k ops per pass.
        # If we do 100 passes, 4.2M ops. Very fast.
        
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Check if removal at i is valid
            # Valid if no critical window is destroyed.
            # A window is destroyed if it is an old window and its count was 1.
            # Old windows affected: indices j such that j <= i <= j+n-1.
            # i.e., j in [i-n+1, i].
            
            valid = True
            for j in range(max(0, i - n + 1), min(i, len(S) - n) + 1):
                if j in critical_indices:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Removal is valid. Perform it.
            # 1. Remove S[i]
            S.pop(i)
            removed_in_pass += 1
            
            # 2. Update counts and critical_indices
            # This is the expensive part.
            # We need to update counts for old and new windows.
            
            # Old windows to decrement:
            old_windows = []
            for j in range(max(0, i - n + 1), min(i, len(S) + n) + 1): # Note: S is already popped?
                # Wait, S is popped. Indices > i are shifted left.
                # Let's reconstruct the logic carefully.
                
                # It's safer to compute the effect before popping, or handle indices carefully.
                # Let's revert: Don't pop yet. Compute changes.
                
                pass
            
            # Let's restart the logic for the update step to be safe.
            # We will not pop in the loop above, but prepare to.
            # Actually, the code above already popped. Let's undo? No, let's just fix the logic.
            
            # Better approach:
            # 1. Identify old windows (indices in original S)
            # 2. Identify new windows (indices in S after removal)
            # 3. Update counts
            # 4. Update critical set
            
            # Since we already popped, let's just rebuild counts?
            # Rebuilding counts is O(L*n). If we do this for every removal, it's O(R * L * n).
            # If R is small (e.g., < 100), this is fine.
            # If R is large, it might be slow.
            # Let's assume R is small for now. If it's not, we optimize.
            
            # Actually, let's just rebuild counts and critical_indices from scratch.
            # It's the simplest and least error-prone.
            counts = build_counts(S, n)
            critical_indices = get_critical_indices(counts, S, n)
            
            best_S = S[:]
            best_len = len(S)
            
        if removed_in_pass == 0:
            stuck_passes += 1
            if stuck_passes > max_stuck_passes:
                break
        else:
            stuck_passes = 0
            
    if best_len < baseline_len:
        return "".join(map(str, best_S))
    else:
        return "".join(map(str, baseline_word))
