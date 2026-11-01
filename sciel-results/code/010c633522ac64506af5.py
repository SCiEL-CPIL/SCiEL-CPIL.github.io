"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _generate_classic_superperm(n):
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
            overlap = next((o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o]), 0)
            new += block[overlap:]
        word = new
    return word

def _build_perm_map(n):
    perm_map = {}
    for idx, p in enumerate(permutations(range(1, n + 1))):
        perm_map[p] = idx
    return perm_map

def _get_window_id(word, pos, n, perm_map):
    if pos < 0 or pos + n > len(word):
        return -1
    window = tuple(word[pos:pos + n])
    return perm_map.get(window, -1)

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    start_time = time.time()
    deadline = start_time + 145
    
    word = _generate_classic_superperm(n)
    perm_map = _build_perm_map(n)
    n_fact = len(perm_map)
    
    counts = [0] * n_fact
    zero_count = n_fact
    for i in range(len(word) - n + 1):
        wid = _get_window_id(word, i, n, perm_map)
        if wid >= 0:
            if counts[wid] == 0:
                zero_count -= 1
            counts[wid] += 1
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(1, len(word) - 1):
            if (time.time() - start_time) >= deadline:
                break
            
            removed = word[i]
            old_affected = list(range(max(0, i - n + 1), min(i, len(word) - n) + 1))
            
            # Remove old windows
            saved_counts = []
            for k in old_affected:
                wid = _get_window_id(word, k, n, perm_map)
                if wid >= 0:
                    old_c = counts[wid]
                    counts[wid] -= 1
                    if old_c == 1:
                        zero_count += 1
                    saved_counts.append((k, wid, old_c))
            
            # Add new windows
            new_word_len = len(word) - 1
            new_affected = []
            for k in range(max(0, i - n + 1), min(i, new_word_len - n) + 1):
                if k + n <= i:
                    window = tuple(word[k:k+n])
                elif k >= i:
                    window = tuple(word[k+1:k+1+n])
                else:
                    left = word[k:i]
                    right = word[i+1:i+1+(n - len(left))]
                    window = tuple(left + right)
                wid = perm_map.get(window, -1)
                if wid >= 0:
                    if counts[wid] == 0:
                        zero_count -= 1
                    counts[wid] += 1
                    new_affected.append((k, wid))
            
            if zero_count == 0:
                del word[i]
                improved = True
                break
            else:
                # Restore
                for (k, wid) in new_affected:
                    counts[wid] -= 1
                    if counts[wid] == 0:
                        zero_count += 1
                for (k, wid, old_c) in saved_counts:
                    counts[wid] = old_c
                word.insert(i, removed)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
