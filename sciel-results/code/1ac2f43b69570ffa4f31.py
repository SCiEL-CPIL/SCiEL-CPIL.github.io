"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from math import factorial
from itertools import permutations

def construct(n):
    deadline = time.time() + 145  # 5-second safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
    if n <= 3:
        perms = list(permutations(range(1, n + 1)))
        best = None
        best_len = n * factorial(n) + 1
        for start in perms:
            s = list(start)
            remaining = set(perms)
            remaining.discard(start)
            while remaining:
                if time.time() > deadline:
                    break
                last = tuple(s[-(n - 1):])
                best_next = None
                best_count = -1
                for sym in range(1, n + 1):
                    new_win = last[1:] + (sym,)
                    cnt = 0
                    for p in remaining:
                        if p[:n - 1] == new_win:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_next = sym
                if best_next is None:
                    break
                s.append(best_next)
                if len(s) >= n:
                    win = tuple(s[-n:])
                    if len(set(win)) == n:
                        remaining.discard(win)
            if len(s) < best_len and len(s) >= n:
                # Verify all perms are covered
                if _verify_superperm(s, n):
                    best_len = len(s)
                    best = s
        if best is not None:
            return "".join(map(str, best))
        # Fallback: classic construction
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o > 0 and new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 4, use jumping construction
    word = _classic_superperm(n)
    
    # Try to improve with local search / jumping
    if time.time() < deadline - 5:
        word = _improve_by_jumping(word, n, deadline)
    
    return "".join(map(str, word))

def _classic_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _verify_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == factorial(n)

def _improve_by_jumping(word, n, deadline):
    """Try to reduce length by finding and removing redundant insertions."""
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    # Identify positions where the new symbol n is inserted
    # In classic construction, n appears after each (n-1)-permutation
    # We can try to remove some insertions if the permutation is still covered
    
    # Build set of all required permutations
    all_perms = set(permutations(range(1, n + 1)))
    
    # Find all positions of symbol n
    n_positions = [i for i, x in enumerate(word) if x == n]
    
    # For each position of n, check if removing it still covers all perms
    # This is expensive, so do it selectively
    
    # Strategy: try removing each n insertion one at a time, in reverse order
    # (later insertions are more likely to be redundant)
    for pos in reversed(n_positions):
        if time.time() > deadline - 2:
            break
        if pos < n - 1 or pos >= len(word) - (n - 1):
            continue
        
        # Try removing word[pos]
        test_word = word[:pos] + word[pos+1:]
        
        # Quick check: does the window around pos still form valid perms?
        # Check if all perms that ended at pos are still covered
        removed_win = tuple(word[pos-n+1:pos+1]) if pos >= n-1 else None
        if removed_win is None:
            continue
        if len(set(removed_win)) != n:
            continue
        
        # Verify the test word still contains all perms
        # This is O(len * n) which is ~5900*7 = 41300 operations, fast enough
        if _verify_superperm(test_word, n):
            word = test_word
            best_word = list(word)
            best_len = len(word)
            # Update n_positions
            n_positions = [i for i, x in enumerate(word) if x == n]
    
    return best_word
# EVOLVE-BLOCK-END
