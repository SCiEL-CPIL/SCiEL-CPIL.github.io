"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
from math import factorial
from itertools import permutations

def _classic_superperm(n):
    """Classic recursive construction giving length 1! + 2! + ... + n!."""
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
    """Check if word contains all permutations of 1..n as substrings."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == factorial(n)

def _get_suffix_map(n):
    """Map each (n-1)-tuple of distinct symbols to the set of permutations starting with it."""
    suffix_map = {}
    for perm in permutations(range(1, n + 1)):
        suffix = perm[:n-1]
        if suffix not in suffix_map:
            suffix_map[suffix] = set()
        suffix_map[suffix].add(perm)
    return suffix_map

def _greedy_extend_fast(current, remaining, n, deadline, suffix_map):
    """
    Fast greedy extension using precomputed suffix map.
    """
    if time.time() > deadline:
        return None
    
    word = list(current)
    remaining = set(remaining)
    
    while remaining:
        if time.time() > deadline:
            return None
        
        if len(word) < n - 1:
            # Not enough symbols yet, just append a new symbol
            best_next = 1
            for sym in range(1, n + 1):
                if sym not in word[-(n-1):]:
                    best_next = sym
                    break
            word.append(best_next)
            continue
        
        last = tuple(word[-(n - 1):])
        
        best_next = None
        best_score = -1
        
        for sym in range(1, n + 1):
            # Check if adding sym creates a valid window
            if len(word) >= n - 1:
                new_window = last[1:] + (sym,)
                if len(set(new_window)) != n:
                    # Invalid window (duplicate symbols), skip
                    continue
            
            score = 0
            
            if len(word) >= n - 1:
                new_window = last[1:] + (sym,)
                
                # Does this complete a permutation?
                if len(set(new_window)) == n:
                    if new_window in remaining:
                        score += 100
                        remaining.discard(new_window)
                    else:
                        score -= 50
                else:
                    # Lookahead: how many remaining perms have this as prefix
                    prefix_len = min(3, n - 1)
                    prefix = new_window[:prefix_len]
                    # Count remaining perms that start with this prefix
                    cnt = 0
                    for p in remaining:
                        if p[:prefix_len] == prefix:
                            cnt += 1
                    score += cnt * 0.1
            
            # Tie-break: prefer symbols that don't create immediate duplicates
            if len(word) >= 1 and sym == word[-1]:
                score -= 10
            
            if score > best_score:
                best_score = score
                best_next = sym
        
        if best_next is None:
            # Fallback: pick any symbol
            best_next = random.randint(1, n)
        
        word.append(best_next)
        
        # Remove any completed permutations
        if len(word) >= n:
            win = tuple(word[-n:])
            if len(set(win)) == n and win in remaining:
                remaining.discard(win)
    
    return word

def _local_search_fast(word, n, deadline, max_iters=500):
    """
    Fast local search to shorten the word by removing redundant symbols.
    """
    if time.time() > deadline:
        return word
    
    word = list(word)
    n_perms = factorial(n)
    
    # Precompute which permutations are covered at which positions
    perm_to_positions = {}
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            if w not in perm_to_positions:
                perm_to_positions[w] = set()
            perm_to_positions[w].add(i)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing symbols from the middle (not the first n-1 or last n-1)
        for j in range(len(word) - n + 1, n - 1, -1):
            if time.time() > deadline:
                break
            
            # Try removing position j
            new_word = word[:j] + word[j+1:]
            if len(new_word) < n:
                continue
            
            # Quick check: does new_word still contain all perms?
            # This is expensive, so do a partial check
            # Check a sample of permutations
            sample_size = min(100, n_perms)
            if random.random() < 0.1:  # Only do full check sometimes
                if _is_valid_superperm(new_word, n):
                    word = new_word
                    improved = True
                    break
            else:
                # Partial check: verify a random sample
                all_perms = list(perm_to_positions.keys())
                sample = random.sample(all_perms, sample_size) if len(all_perms) > sample_size else all_perms
                valid = True
                for p in sample:
                    # Check if p is in new_word
                    p_str = tuple(p)
                    found = False
                    for i in range(len(new_word) - n + 1):
                        if tuple(new_word[i:i+n]) == p_str:
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid:
                    word = new_word
                    improved = True
                    break
    
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # For small n, use exact or near-exact methods
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # Start with the classic construction
    word = _classic_superperm(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # Precompute suffix map for fast greedy extension
    suffix_map = _get_suffix_map(n)
    
    # Try fast local search to shorten
    if time.time() < deadline - 10:
        shortened = _local_search_fast(best_word, n, deadline - 5)
        if len(shortened) < best_len:
            if _is_valid_superperm(shortened, n):
                best_word = shortened
                best_len = len(best_word)
    
    # Try fast greedy construction for n <= 7
    if n <= 7 and time.time() < deadline - 20:
        all_perms = list(permutations(range(1, n + 1)))
        random.shuffle(all_perms)
        
        # Try a few starting permutations with fast greedy extension
        for start_perm in all_perms[:20]:
            if time.time() > deadline - 5:
                break
            
            remaining = set(all_perms)
            remaining.discard(start_perm)
            
            result = _greedy_extend_fast(list(start_perm), remaining, n, deadline - 5, suffix_map)
            if result is not None and len(result) < best_len:
                if _is_valid_superperm(result, n):
                    best_word = result
                    best_len = len(result)
    
    # Final verification
    if not _is_valid_superperm(best_word, n):
        best_word = _classic_superperm(n)
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
