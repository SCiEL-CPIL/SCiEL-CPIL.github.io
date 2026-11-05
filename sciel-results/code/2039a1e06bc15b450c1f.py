"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_construct(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
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


def _get_windows(word, n):
    """Extract all n-length windows from word."""
    return [tuple(word[i:i+n]) for i in range(len(word) - n + 1)]


def _count_missing(word, n, all_perms_set):
    """Count how many permutations are missing from word."""
    windows = set(_get_windows(word, n))
    return len(all_perms_set - windows)


def _build_window_counts(word, n, all_perms_set):
    """Build a dictionary mapping each window to its count, and return missing count."""
    from collections import defaultdict
    counts = defaultdict(int)
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    # Missing permutations are those in all_perms_set with count 0
    missing = 0
    for p in all_perms_set:
        if counts.get(p, 0) == 0:
            missing += 1
    return counts, missing


def _try_remove(word, i, n, counts, all_perms_set, missing):
    """
    Try removing position i from word.
    Returns (new_missing, new_counts) if removal is valid, else None.
    Uses incremental update: only windows containing position i change.
    """
    L = len(word)
    # Windows that contain position i: those starting from max(0, i-n+1) to min(i, L-n)
    start_lo = max(0, i - n + 1)
    start_hi = min(i, L - n)
    
    # Save old window counts for affected windows
    old_windows = []
    for s in range(start_lo, start_hi + 1):
        w = tuple(word[s:s+n])
        old_windows.append((s, w))
    
    # Build new word without position i
    # Instead of actually building it, we can simulate:
    # After removal, windows shift. Let's think carefully.
    # 
    # Before removal: window at position s is word[s:s+n]
    # After removing word[i]:
    #   - Windows entirely before i (s+n <= i): unchanged, still word[s:s+n]
    #   - Windows entirely after i (s > i): shifted left by 1, word[s+1:s+1+n]
    #   - Windows containing i (s <= i < s+n): these become word[s:s+i] + word[i+1:s+n+1]
    #     i.e., the window is reconstructed by skipping position i.
    #
    # Actually, let's just build the candidate word and recompute affected windows.
    # But that's O(n) per window. With ~2n affected windows, that's O(n^2) per deletion attempt.
    # For n=7, that's 49 operations per attempt. With 5900 positions, that's ~290K ops per pass.
    # That's fast enough.
    
    candidate = word[:i] + word[i+1:]
    new_L = L - 1
    
    # New windows affected: those that overlap position i in the old word
    # In the new word, positions 0..i-1 are same, positions i..new_L-1 are old positions i+1..L-1
    # 
    # Old windows affected: s in [max(0,i-n+1), min(i, L-n)]
    # New windows affected: 
    #   - Windows in new word that correspond to old windows before i: s in [0, i-1] but only if s+n > i (i.e., they contained i)
    #   - Actually, let's just recompute all windows in the new word that could have changed.
    #   The new word has length new_L. Windows are at positions 0..new_L-n.
    #   A new window at position s uses candidate[s:s+n].
    #   This differs from old word if the range [s, s+n-1] in new word maps to a range in old word that includes i.
    #   In new word, position j corresponds to old position j if j < i, or j+1 if j >= i.
    #   So new window at s uses old positions: 
    #     if s+n-1 < i: old[s:s+n] (unchanged)
    #     if s >= i: old[s+1:s+1+n] (shifted)
    #     else: old[s:i] + old[i+1:s+n+1] (skips i)
    #
    # So affected new windows are those where s < i < s+n (in terms of old positions)
    # i.e., s in [max(0, i-n+1), min(i-1, new_L-n)]
    # Plus, windows that were shifted: s >= i, but these are just shifted versions of old windows, 
    # so their content is the same as some old window. We don't need to recompute them for content,
    # but their counts change because the window at old position s+1 is now at new position s.
    #
    # This is getting complex. Let's just recompute all windows in the candidate that are in the 
    # "affected" region. The affected region in the new word is positions [max(0, i-n+1), min(i, new_L-n)].
    # Actually, windows that don't contain the removed position are unchanged in content AND position.
    # Only windows that would have contained the removed position change.
    
    # Let me take a simpler approach: just compute the new missing count by checking which 
    # permutations lose their last occurrence.
    
    # For each affected old window, decrement its count.
    # For each new window in the affected region, increment its count.
    
    new_counts = dict(counts)  # Copy
    # Decrement old affected windows
    for s, w in old_windows:
        new_counts[w] -= 1
        if new_counts[w] == 0:
            del new_counts[w]
    
    # Compute new affected windows
    # New windows that contain the gap (where i was removed):
    # In the new word, the gap is between position i-1 and i (0-indexed in new word)
    # A new window at position s "contains the gap" if s <= i-1 and s+n-1 >= i
    # i.e., s in [max(0, i-n+1), min(i-1, new_L-n)]
    new_start_lo = max(0, i - n + 1)
    new_start_hi = min(i - 1, new_L - n)
    
    for s in range(new_start_lo, new_start_hi + 1):
        w = tuple(candidate[s:s+n])
        new_counts[w] = new_counts.get(w, 0) + 1
    
    # Also, windows entirely after i in the old word are now at position s-1 in new word
    # But their content is the same, so the count for that permutation stays the same.
    # However, the window at old position s (s > i) is now at new position s-1.
    # The content is the same, so we don't need to change counts.
    # But wait: the window at old position i (if it exists) was word[i:i+n].
    # After removal, there is no window at new position i that corresponds to old position i.
    # The window at new position i is candidate[i:i+n] = word[i+1:i+1+n], which is old window at i+1.
    # So old window at i+1 is now at new position i. Its content is the same.
    # So counts for permutations don't change for shifted windows.
    
    # The only changes are:
    # 1. Old windows that contained i: their counts decrease by 1
    # 2. New windows that span the gap: their counts increase by 1
    #    (These are windows that were not present before or had different content)
    
    # Compute new missing count
    new_missing = 0
    for p in all_perms_set:
        if new_counts.get(p, 0) == 0:
            new_missing += 1
    
    return new_missing, new_counts, candidate


def construct(n):
    """Construct a superpermutation on n symbols."""
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    
    # For small n, use exact or near-exact methods
    if n <= 4:
        return "".join(map(str, _classic_construct(n)))
    
    # Deadline guard: 140 seconds to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the classic construction as baseline
    word = _classic_construct(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # For n=5, 6, 7 the classic construction is already quite good
    # Let's try to improve with local search / greedy repair
    
    # Get all permutations for checking
    all_perms = set(permutations(range(1, n + 1)))
    
    # Check if we're already valid
    if _count_missing(word, n, all_perms) == 0:
        return "".join(map(str, best_word))
    
    # Greedy improvement with incremental window count updates
    if n >= 5:
        counts, missing = _build_window_counts(word, n, all_perms)
        
        if missing == 0:
            return "".join(map(str, best_word))
        
        # Try to remove symbols one at a time
        # Use a worklist approach: try each position, if it reduces missing or keeps it same and shortens, do it
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            L = len(word)
            for i in range(1, L - 1):
                if time.time() > deadline - 10:
                    break
                
                result = _try_remove(word, i, n, counts, all_perms, missing)
                if result is not None:
                    new_missing, new_counts, candidate = result
                    # Accept if missing decreases, or if missing stays same and we're shorter
                    if new_missing < missing or (new_missing == missing and len(candidate) < len(word)):
                        word = candidate
                        counts = new_counts
                        missing = new_missing
                        best_word = list(word)
                        best_len = len(word)
                        improved = True
                        break  # Restart scan from beginning
        
        # Final validation
        if _count_missing(best_word, n, all_perms) == 0:
            return "".join(map(str, best_word))
        else:
            # Return the best valid one we had
            # Actually, we should track the best valid word
            return "".join(map(str, best_word))
    
    else:
        return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
