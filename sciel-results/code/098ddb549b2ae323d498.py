# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

# Precompute permutation index for n=7 (and smaller)
# Map each permutation tuple to a unique integer index
_PERM_INDEX = {}
_PERM_LIST = []
_PERM_INV = []  # inverse: index -> tuple

def _build_perm_tables(n):
    """Build permutation index tables for n symbols."""
    global _PERM_INDEX, _PERM_LIST, _PERM_INV
    _PERM_INDEX = {}
    _PERM_LIST = []
    _PERM_INV = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        _PERM_INV.append(p)
        idx += 1

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _perm_to_idx(window_tuple, n):
    """Convert a window tuple to its permutation index, or -1 if not a valid permutation."""
    if len(window_tuple) != n:
        return -1
    # Check if it's a permutation of 1..n
    if len(set(window_tuple)) != n:
        return -1
    idx = _PERM_INDEX.get(window_tuple, -1)
    return idx

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Fast deletion pass using permutation frequency counting.
    A character at position i can be deleted if no valid permutation window
    that contains position i is the ONLY occurrence of that permutation.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    # Build initial permutation frequency table
    # For each position i (window start), compute which permutation it is
    window_perm = [-1] * (len(word) - n + 1)
    perm_count = [0] * total_perms
    
    for i in range(len(word) - n + 1):
        window_tuple = tuple(word[i:i + n])
        if len(set(window_tuple)) == n:
            idx = _PERM_INDEX.get(window_tuple, -1)
            if idx >= 0:
                window_perm[i] = idx
                perm_count[idx] += 1
    
    # Count how many times each position is part of a valid permutation window
    # pos_in_valid_window[pos] = number of valid permutation windows containing pos
    # Also track: for each position, which permutations would be affected
    # Better approach: for each position, track the set of permutation indices
    # of valid windows that contain it. But that's expensive.
    
    # Alternative: for a deletion at position i, the affected windows are
    # windows [max(0, i-n+1) .. min(i, len(word)-n)] (window starts)
    # A window start s contains position i iff s <= i <= s+n-1
    # i.e., s in [i-n+1, i] intersected with [0, len(word)-n]
    
    # For speed, we'll do a greedy pass: try deleting each position,
    # check if any valid window containing it would become the last occurrence.
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try positions in random-ish order for diversity
        for i in range(len(word) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            # Find all valid permutation windows containing position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            critical = False
            for s in range(s_min, s_max + 1):
                if window_perm[s] >= 0:
                    if perm_count[window_perm[s]] == 1:
                        critical = True
                        break
            if not critical:
                # Delete position i
                # Update window_perm and perm_count
                # Windows that are destroyed: those with s in [s_min, s_max] that were valid
                # Windows that are created: none (deletion only)
                # Actually, windows that shift: windows with s < s_min are unaffected
                # Windows with s in [s_min, s_max]: the window at s now corresponds to
                #   word[s:s+n] after deletion, which is the old word[s:s+n+1] minus word[i]
                #   This is complex. Let me just rebuild the affected region.
                
                # Remove word[i]
                del word[i]
                
                # Rebuild window_perm for affected region
                # Windows with start s where s <= i <= s+n-1 are affected
                # After deletion, windows with start s < i are now word[s:s+n] (shifted)
                # Windows with start s >= i are now word[s:s+n] (unchanged content but shifted)
                
                # Simplest: rebuild all windows in the affected range
                # Affected window starts: those that overlapped position i before deletion
                # Before deletion: s in [i-n+1, i] (clamped)
                # After deletion, the word is shorter by 1
                # New word length = len(word)
                # New window starts range: [0, len(word)-n]
                
                # For correctness, rebuild window_perm for all s in [max(0,i-n+1), min(i, len(word)-n)]
                # But also, windows that were at s > i are now at s (same start, but the
                # content is word[s:s+n] which in the new word is the old word[s:s+n] since
                # we deleted before them... wait, we deleted at position i, so positions > i
                # shift left by 1. So old window at start s (s > i) becomes new window at
                # start s-1.
                
                # This is getting complex. Let me just rebuild the entire window_perm
                # for the affected region carefully.
                
                # New word length
                new_len = len(word)
                
                # For each window start s in the new word that could be affected:
                # A window start s in the new word corresponds to:
                #   - If s < i: old window start s (content is old word[s:s+n] with old word[i] removed)
                #     Actually no. Old word had position i. New word is old word with position i removed.
                #     New word[j] = old word[j] for j < i, old word[j+1] for j >= i.
                #     New window at start s (s < i): new word[s:s+n] = old word[s:s+n] (since all < i)
                #     Wait, if s+n-1 < i, then yes. If s+n-1 >= i, then the window spans the deletion.
                #     
                # Let me just rebuild all windows. For n=7 and word length ~5900,
                # that's ~5900 windows, each a tuple of 7 elements. This is fast enough.
                
                # Rebuild window_perm and perm_count from scratch
                # This is O(len(word) * n) per deletion, which is ~5900*7 = 41300 ops
                # With ~5900 deletions to try, that's ~243M ops. Too slow.
                
                # Better: only rebuild the affected windows.
                # Affected new window starts: those s where the new window [s, s+n-1]
                # includes a position that was shifted, i.e., s >= i-... 
                # Actually, positions j >= i in new word correspond to old positions j+1.
                # So new window [s, s+n-1] is affected if s+n-1 >= i (i.e., s >= i-n+1)
                # and s <= i (since positions < i are unchanged, but the window might
                # extend past i).
                # Also, new window starts s where s < i-n+1 are completely before the
                # deletion point and unchanged.
                # New window starts s where s > i: the window [s, s+n-1] in new word
                # corresponds to old [s+1, s+n] (all shifted). So it's a different window.
                
                # Let me define the affected range more carefully.
                # Old word length = L, new word length = L-1.
                # Deletion at old position i (0-indexed).
                # New position j corresponds to old position j (if j < i) or j+1 (if j >= i).
                # A new window at start s covers new positions [s, s+n-1].
                # This window is "new" (different from any old window) if it includes
                # at least one position j >= i (shifted) OR if it's a window that didn't
                # exist before (s > L-1-n).
                # 
                # For the frequency update, I need to:
                # 1. Remove the old windows that are no longer valid (or changed)
                # 2. Add the new windows
                #
                # Old windows that are affected: old window starts s_old where the old
                # window [s_old, s_old+n-1] included position i, i.e., s_old <= i <= s_old+n-1,
                # i.e., s_old in [i-n+1, i].
                # These old windows are gone (replaced by new windows).
                #
                # New windows that are affected: new window starts s_new where the new
                # window [s_new, s_new+n-1] includes a position that was shifted, i.e.,
                # s_new >= i-n+1 (since new position i-1 corresponds to old position i,
                # wait no. New position i-1 = old position i-1 (unchanged). New position i = old position i+1.
                # So new position j >= i corresponds to old position j+1.
                # New window [s_new, s_new+n-1] includes a shifted position if s_new+n-1 >= i,
                # i.e., s_new >= i-n+1.
                # Also, new window starts s_new > i-1 = i-1... hmm.
                # New window starts range: [0, L-1-n].
                # Old window starts range: [0, L-n].
                # New windows with s_new in [i-n+1, L-1-n] are affected.
                # Old windows with s_old in [i-n+1, i] are affected.
                
                # Let me just rebuild the affected range.
                # First, decrement counts for old affected windows
                old_s_min = max(0, i - n + 1)
                old_s_max = min(i, len(word) + 1 - n)  # old length was len(word)+1
                for s in range(old_s_min, old_s_max + 1):
                    if s < len(window_perm) and window_perm[s] >= 0:
                        perm_count[window_perm[s]] -= 1
                        window_perm[s] = -1  # mark as invalid for now
                
                # Now compute new windows for s_new in [max(0, i-n+1), min(i, len(word)-n)]
                # Wait, new word length is len(word) (after deletion).
                # New window starts: [0, len(word)-n]
                # Affected new starts: [max(0, i-n+1), min(len(word)-n, i)] 
                # Actually, new position i-1 is the last unshifted position before the gap.
                # New position i is the first shifted position.
                # A new window is affected if it includes new position i or later,
                # i.e., s_new + n - 1 >= i, i.e., s_new >= i - n + 1.
                # Also, new windows with s_new > i-1 are entirely in the shifted region.
                # But s_new can go up to len(word)-n.
                
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i, len(word) - n)
                # Also need to handle new windows with s_new > i that are shifted
                # But those were old windows with s_old = s_new + 1, which we already
                # accounted for if s_old was in the affected range.
                # Hmm, this is getting tangled. Let me just rebuild ALL windows.
                
                # For n=7, word length ~5900, rebuilding all windows is ~5900 * 7 = 41300
                # tuple operations. With ~1000 deletions, that's ~41M ops. Should be OK in 100s.
                
                window_perm = [-1] * (len(word) - n + 1)
                perm_count = [0] * total_perms
                for s in range(len(word) - n + 1):
                    window_tuple = tuple(word[s:s + n])
                    if len(set(window_tuple)) == n:
                        idx = _PERM_INDEX.get(window_tuple, -1)
                        if idx >= 0:
                            window_perm[s] = idx
                            perm_count[idx] += 1
                
                improved = True
                break  # Restart scan from beginning after a removal
    
    return word

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    total_perms = n * _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == total_perms

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters using fast deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
