"""Improved superpermutation constructor with efficient local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import permutations
from collections import defaultdict

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        # Collect all distinct (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _incremental_delete(word, n, time_limit):
    """
    Efficiently delete characters while maintaining window counts incrementally.
    A deletion at position i is safe iff every window that contains i has count >= 2
    in the current word, OR the window reappears at a new position after the shift.
    
    Key insight: after deleting position i, windows at positions s in [i-n+1, i] 
    disappear, but new windows appear at positions s' in [i-n+1, i-1] (shifted left).
    A window w at position s (s < i) has content word[s..s+n-1]. After deletion,
    the window at position s (if s+n-1 > i) becomes word[s..i-1] + word[i+1..s+n-1].
    The window at position i-1 (if i-1+n <= L-1) becomes word[i-1..i-2] + word[i..i+n-2] 
    shifted... actually it's simpler: new window at position s (for s in [max(0,i-n+1), min(i-1, L-n)])
    has content: word[s:i] + word[i+1:s+n] (the gap is filled).
    
    For efficiency, we'll use a two-pass approach:
    1. Try deletions from left to right
    2. For each position, check if all affected windows are "safe"
    """
    start_time = time.time()
    word = list(word)
    L = len(word)
    target_count = n * math.factorial(n - 1)  # n!
    
    # Build initial window counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try each position once per pass
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return word
            if L <= n:
                break
            
            # Determine which windows contain position i
            # Window at position s contains position i iff s <= i <= s+n-1
            # So s in [i-n+1, i] intersected with [0, L-n]
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            # Check if all windows containing position i have count >= 2
            # AND that no unique window is lost without being recreated
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s+n])
                if counts[w] == 1:
                    # This window is unique. After deletion, does it reappear?
                    # New windows appear at positions s_new in [max(0, i-n+1), min(i-1, L-n-1)]
                    # (positions that span the gap)
                    # Actually, after deleting position i, the word length becomes L-1.
                    # New windows at position p (0 <= p <= L-n-1) have content:
                    #   if p + n - 1 < i: word[p:p+n] (unchanged)
                    #   if p <= i-1 and p+n-1 >= i: word[p:i] + word[i+1:p+n] (spans gap)
                    #   if p >= i: word[p+1:p+1+n] (shifted)
                    # 
                    # For the unique window w to reappear, we need some new position p
                    # where the new window content equals w.
                    # 
                    # The windows that change are those spanning the gap:
                    # p in [max(0, i-n+1), min(i-1, L-n-1)]
                    # New content at such p: word[p:i] + word[i+1:p+n]
                    # 
                    # This is expensive to check for all p. Instead, let's use a 
                    # simpler heuristic: if the unique window w has its last 
                    # occurrence at position s and s >= i-n+1, it might be lost.
                    # 
                    # For correctness, we'll do a quick check: if w's only occurrence
                    # is at position s, and after deletion the window at position s
                    # (if it still exists, i.e., s+n-1 < i or s >= i) changes, 
                    # we need to verify.
                    # 
                    # Simplified safe check: if counts[w] == 1, we must verify
                    # that w still appears in the new word. We'll do a targeted search.
                    safe = False
                    break
            
            if not safe:
                continue
            
            # All windows containing position i have count >= 2, so deletion is safe
            # Perform the deletion
            word.pop(i)
            L = len(word)
            improved = True
            
            # Update counts incrementally
            # Windows that disappear: positions in [lo, hi] (old positions)
            # But after deletion, positions >= i shift left by 1.
            # 
            # Old windows at positions s in [0, L_old - n] where L_old = L+1:
            # - If s + n - 1 < i: unchanged, still at position s
            # - If s <= i-1 and s+n-1 >= i: these span the gap, their content changes
            # - If s >= i: shift to position s-1, content unchanged
            #
            # New windows at positions s in [0, L - n]:
            # - If s + n - 1 < i: same as old window at s
            # - If s <= i-1 and s+n-1 >= i: new content word[s:i] + word[i:s+n]
            #   (note: in the new word, position i is the old position i+1)
            #   Wait, after deletion, the new word has positions 0..L-1 where L = L_old - 1.
            #   New word[j] = old word[j] for j < i, old word[j+1] for j >= i.
            #   New window at position s: new_word[s:s+n]
            #   = old_word[s:i] + old_word[i+1:s+n] if s <= i-1 < s+n-1
            #   = old_word[s:s+n] if s+n-1 < i
            #   = old_word[s+1:s+1+n] if s >= i
            #
            # So the windows that change are:
            # 1. Old windows at positions s in [max(0,i-n+1), min(i, L_old-n)] are "removed"
            #    (their old content no longer appears at those positions in the same form)
            # 2. New windows at positions s in [max(0,i-n+1), min(i-1, L-n)] have new content
            #
            # Let's be more careful:
            # Old word has length L_old = L + 1.
            # Old windows: positions 0 to L_old - n = L - n + 1
            # New word has length L.
            # New windows: positions 0 to L - n
            #
            # For old window at position s:
            #   If s + n - 1 < i: new word has same window at position s
            #   If s <= i - 1 and s + n - 1 >= i: this old window is "destroyed" 
            #      (its content is no longer a window in the new word at the same position)
            #   If s >= i: new word has this window at position s - 1
            #
            # For new window at position s:
            #   If s + n - 1 < i: same as old window at s
            #   If s <= i - 1 and s + n - 1 >= i: NEW window with content 
            #      new_word[s:s+n] = old_word[s:i] + old_word[i+1:s+n]
            #   If s >= i: same as old window at s+1
            #
            # So:
            # - Old windows at positions s in [max(0, i-n+1), min(i-1, L-n+1)] are destroyed
            #   (they spanned the gap)
            #   Wait: s + n - 1 >= i means s >= i - n + 1
            #   And s <= i - 1
            #   So s in [max(0, i-n+1), min(i-1, L-n+1)]
            # - New windows at positions s in [max(0, i-n+1), min(i-1, L-n)] are created
            #   (they span the gap in the new word)
            #   s + n - 1 >= i means s >= i - n + 1
            #   And s <= i - 1
            #   And s <= L - n
            #   So s in [max(0, i-n+1), min(i-1, L-n)]
            #
            # Note: L-n+1 = (L+1)-n, and L-n = L-n.
            # The destroyed old windows: s in [max(0,i-n+1), min(i-1, L-n+1)]
            # The created new windows: s in [max(0,i-n+1), min(i-1, L-n)]
            #
            # Also, old window at position i (if i <= L-n+1) is destroyed? 
            # Old window at position i: positions i to i+n-1 in old word.
            # In new word, position i corresponds to old position i+1.
            # New window at position i: new_word[i:i+n] = old_word[i+1:i+1+n]
            # So old window at position i is NOT the same as new window at position i.
            # Old window at position i is "shifted" to position i-1 in the new word? 
            # No: new window at position i-1 = old_word[i-1:i] + old_word[i+1:i+n] 
            # (if i-1 <= i-1 < i-1+n-1 = i+n-2, which is true for n>=2)
            # Hmm, this is getting complicated. Let me just rebuild counts when L is small
            # or use a simpler but correct approach.
            
            # For correctness and simplicity, let's rebuild counts periodically
            # but use the incremental check for safety.
            # Actually, for n=7 and L~5900, rebuilding is O(5900) which is fine.
            # The bottleneck was doing it after EVERY deletion in a tight loop.
            # Let's rebuild every 100 deletions or when time allows.
            
            # Rebuild counts (this is O(L) but L is at most ~6000)
            counts = defaultdict(int)
            for j in range(L - n + 1):
                w = tuple(word[j:j+n])
                counts[w] += 1
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5:
        # Try incremental deletion
        word = _incremental_delete(word, n, remaining - 5)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
