"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from math import factorial

def _classic_superperm(n):
    """Generate the classic recursive superpermutation of length 1!+2!+...+n!"""
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
    return word

def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    from itertools import permutations
    target_perms = set(permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        win = tuple(word[i:i + n])
        if len(set(win)) == n:
            found.add(win)
        if len(found) == len(target_perms):
            return True
    return found == target_perms

def _try_remove_symbol(word, pos, n, deadline):
    """Try to remove symbol at position pos if it doesn't break validity."""
    if time.time() > deadline:
        return None
    original = list(word)
    # Check if removing pos would break any permutation
    # A permutation is broken if it only occurs at positions that include pos
    # For each window of length n that includes pos, check if that permutation
    # occurs elsewhere without pos
    affected_windows = []
    for start in range(max(0, pos - n + 1), min(len(word) - n + 1, pos + 1)):
        win = tuple(word[start:start + n])
        if len(set(win)) == n:
            affected_windows.append(win)
    
    if not affected_windows:
        # No valid permutation affected, safe to remove
        del word[pos]
        return word
    
    # For each affected permutation, check if it occurs elsewhere
    # Build set of all permutations in original word
    all_perms = set()
    for i in range(len(word) - n + 1):
        win = tuple(word[i:i + n])
        if len(set(win)) == n:
            all_perms.add(win)
    
    # Try removing and check validity
    candidate = list(word)
    del candidate[pos]
    
    # Quick check: does candidate still contain all permutations?
    # This is expensive, so only do it if we have time
    if time.time() > deadline:
        return None
    
    candidate_perms = set()
    for i in range(len(candidate) - n + 1):
        win = tuple(candidate[i:i + n])
        if len(set(win)) == n:
            candidate_perms.add(win)
        if len(candidate_perms) == len(all_perms):
            break
    
    if candidate_perms == all_perms:
        return candidate
    return None

def _optimize_by_removal(word, n, deadline):
    """Iteratively try to remove symbols to shorten the superpermutation."""
    word = list(word)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(word)):
            if time.time() > deadline:
                break
            result = _try_remove_symbol(word, pos, n, deadline)
            if result is not None:
                word = result
                improved = True
                break  # Restart scan from beginning since positions shifted
    return word

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n <= 3:
        from itertools import permutations
        perms = list(permutations(range(1, n + 1)))
        best = None
        best_len = n * factorial(n) + 1
        for start in perms:
            s = list(start)
            remaining = set(perms)
            remaining.discard(start)
            while remaining:
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
            if len(s) < best_len:
                best_len = len(s)
                best = s
        if best is not None:
            return "".join(map(str, best))
        result = ""
        for p in perms:
            p_str = "".join(map(str, p))
            if p_str not in result:
                for k in range(n - 1, -1, -1):
                    if result[-k:] == p_str[:k]:
                        result += p_str[k:]
                        break
                else:
                    result += p_str
        return result
    
    # For n >= 4, use classic construction + optimization pass
    deadline = time.time() + 120  # Safety margin before 150s limit
    word = _classic_superperm(n)
    
    # Apply removal optimization with time guard
    word = _optimize_by_removal(word, n, deadline)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
