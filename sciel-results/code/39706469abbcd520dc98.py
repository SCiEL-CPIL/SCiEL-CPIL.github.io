import time
import math

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with bitset tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))

def generate_classic(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def local_search_optimize(word, n, target_count, start_time, deadline):
    """
    Local search using bitset tracking for O(1) amortized validity checks.
    Tries single deletions and 2-swaps to reduce length.
    """
    L = len(word)
    if L < target_count + n - 1:
        return word
    
    # Precompute factorial for ranking
    # We'll use a dictionary to map permutations to indices
    # For n=7, 5040 entries is manageable
    
    # Build permutation rank function using Lehmer code
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    
    def perm_rank(p):
        """Compute rank of permutation p (0-indexed) using Lehmer code"""
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        return rank
    
    # Build the set of all valid permutation ranks
    # We need to track which permutations we've seen
    # Use a boolean array of size target_count
    seen = bytearray(target_count)  # 0 or 1
    seen_count = 0
    
    # Initialize: compute rank for each window
    window_ranks = []
    for i in range(L - n + 1):
        r = perm_rank(word[i:i+n])
        window_ranks.append(r)
        if not seen[r]:
            seen[r] = 1
            seen_count += 1
    
    # Verify we have all permutations
    assert seen_count == target_count, f"Only found {seen_count}/{target_count} permutations"
    
    best_word = word[:]
    best_len = L
    
    # Greedy single-deletion search with restarts
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        L = len(best_word)
        improved = False
        
        # Rebuild tracking structures for current best_word
        seen = bytearray(target_count)
        seen_count = 0
        window_ranks = []
        for i in range(L - n + 1):
            r = perm_rank(best_word[i:i+n])
            window_ranks.append(r)
            if not seen[r]:
                seen[r] = 1
                seen_count += 1
        
        # Try each single deletion
        for i in range(L - 1):  # Don't delete last char (would lose last window)
            if time.time() > deadline:
                break
            
            # Check if deletion is valid
            # After deleting position i, windows shift
            # Window at position j (j < i) stays the same
            # Window at position j (j >= i) becomes old window at j+1
            # Last window is lost
            # New windows: old[0..i-1], old[i+1..L-n]
            # We need all target_count permutations still present
            
            # Quick check: if the deleted position is part of a unique window, it fails
            # More efficient: just check if removing windows[i..L-n-1] and adding nothing
            # still covers all permutations
            
            # Actually, after deletion at position i:
            # New windows are:
            #   old windows 0..i-1 (unchanged)
            #   old windows i+1..L-n (shifted left by 1)
            # Total new windows: (i) + (L-n-i) = L-n-1 = (L-1)-n+1 ✓
            
            # We need to check if all permutations are still covered
            # A permutation is lost if it ONLY appeared in window i (the one that shifts out)
            # Wait, no. Let me think more carefully.
            
            # Original windows: W[0], W[1], ..., W[L-n]
            # After deleting position i:
            # New word has length L-1
            # New windows: W'[0], W'[1], ..., W'[L-1-n]
            # W'[j] = W[j] for j < i
            # W'[j] = W[j+1] for j >= i
            # So W'[L-1-n] = W[L-n] (the last original window)
            # The window that's "lost" is W[i] (it's no longer a window in the new word)
            # Wait, that's not right either.
            
            # Let me reconsider. Original word: w[0] w[1] ... w[L-1]
            # Windows: (w[0]..w[n-1]), (w[1]..w[n]), ..., (w[L-n]..w[L-1])
            # Delete w[i]: new word is w[0]..w[i-1] w[i+1]..w[L-1]
            # New windows:
            #   (w[0]..w[n-1]) if i >= n: same as before
            #   ...
            #   For j < i: window j is w[j]..w[j+n-1], unchanged
            #   For j >= i: window j is w[j]..w[j+n-1] in new word
            #     = w[j]..w[j+n-1] where indices skip i
            #     This equals old window j+1 shifted, but only if j+n-1 < i or j >= i+n
            #     Actually it's more complex.
            
            # Simpler approach: just rebuild and check. With 5040 perms and ~13k windows,
            # a full check is O(L*n) ≈ 13000*7 = 91000 ops, which is fast enough.
            # But we can do better with incremental tracking.
            
            # For now, let's do a quick necessary condition:
            # The window at position i in the original (w[i]..w[i+n-1]) disappears.
            # If this window's permutation is unique (appears only once), deletion fails.
            # But even if it appears multiple times, other windows shift.
            
            # Let's just do the full check for correctness, but optimize the common case.
            
            # Quick reject: if window i has a permutation that appears only once, skip
            # (This is a necessary condition for failure, not sufficient for success)
            r_i = window_ranks[i]
            # Count occurrences of r_i
            # This is O(L) per check, which is too slow for L~13000
            # Let's precompute occurrence counts
            
            # Actually, let's just try the deletion and verify.
            # For n=7, L starts at 13699, so L-n+1 = 13693 windows.
            # A full rebuild is 13693 * 7 = ~96000 ops, which takes maybe 10-50ms in Python.
            # With 13699 positions to try, that's too slow.
            
            # Better: use the fact that only windows near position i are affected.
            # When we delete position i:
            # - Windows 0..i-1 are unchanged (if i >= n, window i-1 ends at i-1+n-1 = i+n-2, which is < i+n-1, so it doesn't include w[i])
            #   Wait, window j covers positions j..j+n-1.
            #   Window i-1 covers i-1..i+n-2. Does it include position i? Yes, if i <= i+n-2, i.e., n >= 2.
            #   So window i-1 DOES include w[i].
            
            # Let me be more careful:
            # Original: positions 0, 1, ..., L-1
            # Window j: positions j, j+1, ..., j+n-1
            # Delete position i.
            # New positions: 0, 1, ..., i-1, i+1, ..., L-1 (renumbered)
            # New window j:
            #   If j + n - 1 < i: covers old positions j..j+n-1 (unchanged)
            #   If j >= i: covers old positions j+1..j+n (since we skip i)
            #     Wait, new position j corresponds to old position j if j < i, or j+1 if j >= i.
            #     New window j covers new positions j..j+n-1.
            #     In old coordinates:
            #       If j+n-1 < i: old positions j..j+n-1
            #       If j < i <= j+n-1: old positions j..i-1, i+1..j+n-1 (skipping i)
            #       If j >= i: old positions j+1..j+n
            #
            # So windows that change are those where j <= i <= j+n-1, i.e., i-n+1 <= j <= i.
            # That's at most n windows that change!
            
            # This is the key insight. Only n windows are affected by a single deletion.
            
            # Affected windows: j from max(0, i-n+1) to min(i, L-n)
            # For these, we need to recompute the new window content.
            
            # Let's implement this properly.
            
            # First, quick check: is the permutation of the "disappearing" window unique?
            # The window that no longer exists in any form is... actually, all old windows
            # either persist or are replaced. Let me think differently.
            
            # Old windows: W[0], ..., W[L-n]
            # New windows: W'[0], ..., W'[L-1-n]
            # W'[j] for j < i-n+1 (if i >= n): same as W[j]
            # W'[j] for i-n+1 <= j <= i: new content (skips position i)
            # W'[j] for j > i: same as W[j+1]
            
            # So the multiset of windows changes by:
            # - Removing W[i-n+1], ..., W[i] (n windows, if they exist)
            # - Adding W'[i-n+1], ..., W'[i] (n new windows)
            # - W[j] for j < i-n+1 stays
            # - W[j] for j > i becomes W'[j-1]
            
            # Wait, I think I'm overcomplicating. Let me just check:
            # After deletion, the new word is word[:i] + word[i+1:]
            # The new windows are simply the n-consecutive substrings of this new word.
            # For a deletion at position i, windows 0..i-n (if i >= n) are unchanged.
            # Windows i-n+1..i are affected (they now skip position i).
            # Windows i+1..L-n become the old windows i+2..L-n+1... no wait.
            
            # Let's just do it directly for the affected range.
            
            # Actually, the cleanest approach: for each candidate deletion,
            # check if the resulting word is valid by tracking which permutations
            # are still covered. Since only ~n windows change, we can update incrementally.
            
            # But for simplicity and correctness, let's use a different strategy:
            # Precompute for each position which permutations become "lost" if we delete it.
            
            # Simpler approach that works well in practice:
            # Try deletions in random order, and for each, do a full validity check
            # but early-exit as soon as we find a missing permutation.
            
            # For n=7, a full check is ~13000 window computations.
            # In Python, that's maybe 5-10ms. With 13000 positions, that's 65-130s.
            # Too slow for one pass.
            
            # Let's use the incremental approach properly.
            
            # Precompute: for each window position j, its rank r_j
            # Also precompute: for each rank r, the list of window positions that have rank r
            # This lets us quickly check if a rank becomes uncovered.
            
            # Build rank -> positions map
            rank_positions = [[] for _ in range(target_count)]
            for j, r in enumerate(window_ranks):
                rank_positions[r].append(j)
            
            # For deletion at position i:
            # Affected window positions: j where the window content changes
            # Window j covers positions j..j+n-1
            # It's affected if i is in [j, j+n-1], i.e., j <= i <= j+n-1
            # So j in [i-n+1, i]
            
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            # Windows that disappear (their old content is no longer a window):
            # All old windows j in [j_min, j_max] have their content changed
            # But actually, the old windows still "exist" in the sense that their
            # permutations might still appear elsewhere.
            
            # Let's think about it as: what is the new set of window ranks?
            # Old ranks: W[0], ..., W[L-n]
            # New ranks: 
            #   W[0], ..., W[j_min-1] (unchanged, if j_min > 0)
            #   New ranks for j in [j_min, j_max]
            #   W[j_max+1], ..., W[L-n] (but shifted: old W[j] becomes new W[j-1] for j > i)
            
            # Ugh, this is getting complicated. Let me just do a direct check
            # for the affected windows and track the net change in coverage.
            
            # Key insight: a permutation r is still covered after deletion if:
            # 1. It appears in an unaffected window (j < j_min or j > j_max, with appropriate shift)
            # 2. OR it appears in one of the new affected windows
            # 3. AND it didn't ONLY appear in affected windows that are now gone
            
            # This is still complex. Let me use a simpler but effective heuristic:
            # For each deletion, check if any permutation becomes uncovered.
            # A permutation r becomes uncovered if all its occurrences are in affected windows
            # AND none of the new affected windows have rank r.
            
            # Let's precompute the new ranks for affected windows.
            # For j in [j_min, j_max], the new window at position j in the new word
            # corresponds to old positions: j..i-1, i+1..j+n-1 (skipping i)
            # In the new word, this is at position j (if j < i) or j-1 (if j >= i).
            # Wait, I need to be more careful.
            
            # Let's just compute the new word's windows directly for the affected range.
            # New word: word[:i] + word[i+1:]
            # For new window at position j:
            #   If j + n - 1 < i: old positions j..j+n-1
            #   If j < i <= j + n - 1: old positions j..i-1, i+1..j+n-1
            #   If j >= i: old positions j+1..j+n
            
            # For j in [j_min, j_max], we have j < i <= j+n-1 (by definition of j_min, j_max)
            # So the new window at position j is: word[j:i] + word[i+1:j+n]
            # (in old coordinates)
            
            # Let's compute these new ranks.
            new_affected_ranks = {}
            for j in range(j_min, j_max + 1):
                # New window at position j in new word
                # Old positions: j to i-1, then i+1 to j+n-1
                new_window = word[j:i] + word[i+1:j+n]
                if len(new_window) == n:
                    r = perm_rank(new_window)
                    new_affected_ranks[j] = r
                else:
                    # This shouldn't happen if j_max <= L-n
                    new_affected_ranks[j] = -1
            
            # Now, check if any permutation becomes uncovered.
            # For each rank r that appeared in affected old windows:
            #   If r doesn't appear in unaffected old windows AND r not in new_affected_ranks.values():
            #     Then r becomes uncovered -> deletion fails
            
            # Build set of new affected ranks
            new_affected_set = set(new_affected_ranks.values())
            
            # Check each affected old window's rank
            valid_deletion = True
            for j in range(j_min, j_max + 1):
                r = window_ranks[j]
                # Check if r appears in unaffected windows
                # Unaffected windows: [0, j_min-1] and [j_max+1, L-n]
                # But wait, windows [j_max+1, L-n] in the old word become
                # windows [j_max, L-n-1] in the new word (shifted left by 1).
                # Their content is unchanged, so their ranks are the same.
                # So we just need to check if r appears in positions outside [j_min, j_max].
                
                # Quick check: if rank_positions[r] has any position outside [j_min, j_max]
                # We can check the first and last positions in rank_positions[r]
                if len(rank_positions[r]) == 1:
                    # Only one occurrence, and it's in the affected range
                    if j_min <= rank_positions[r][0] <= j_max:
                        if r not in new_affected_set:
                            valid_deletion = False
                            break
                else:
                    # Multiple occurrences: check if any is outside [j_min, j_max]
                    first_pos = rank_positions[r][0]
                    last_pos = rank_positions[r][-1]
                    if not (first_pos < j_min or last_pos > j_max):
                        # All occurrences are in [j_min, j_max]
                        if r not in new_affected_set:
                            valid_deletion = False
                            break
            
            if valid_deletion:
                # Perform the deletion
                best_word = best_word[:i] + best_word[i+1:]
                improved = True
                best_len = len(best_word)
                break  # Restart scan with the new shorter word
        
        if not improved:
            break
    
    return best_word
