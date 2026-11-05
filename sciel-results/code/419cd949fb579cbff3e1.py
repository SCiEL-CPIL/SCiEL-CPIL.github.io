import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1)-length windows with distinct symbols
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Generate m permutations by inserting m into p
            blocks = []
            for k in range(m):
                block = list(p[:k]) + [m] + list(p[k:])
                blocks.append(block)
            
            # Merge blocks
            if not new:
                new = blocks[0]
            else:
                for b in blocks[1:]:
                    max_check = min(len(new), len(b))
                    overlap = 0
                    for o in range(max_check, -1, -1):
                        if new[len(new) - o:] == b[:o]:
                            overlap = o
                            break
                    new += b[overlap:]
        
        word = new

    if n < 7:
        return "".join(map(str, word))

    # Optimization for n=7
    w = word
    L = len(w)
    
    all_perms = [tuple(p) for p in permutations(range(1, n+1))]
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    counts = [0] * len(all_perms)
    
    for i in range(L - n + 1):
        window = tuple(w[i:i+n])
        if window in perm_to_idx:
            counts[perm_to_idx[window]] += 1

    # Greedy deletion
    while time.time() < deadline:
        improved = False
        for i in range(len(w)):
            if time.time() > deadline:
                break
            
            valid_old_starts = []
            for s in range(max(0, i - n + 1), min(i, len(w) - n) + 1):
                valid_old_starts.append(s)
                
            valid_new_starts = []
            for s in range(max(0, i - n + 1), min(i - 1, len(w) - n) + 1):
                if s <= len(w) - n:
                    valid_new_starts.append(s)

            affected = {}
            
            for s in valid_old_starts:
                p = tuple(w[s:s+n])
                if p in perm_to_idx:
                    idx = perm_to_idx[p]
                    affected[idx] = affected.get(idx, 0) - 1
            
            for s in valid_new_starts:
                window = w[s:i] + w[i+1:s+n]
                p = tuple(window)
                if p in perm_to_idx:
                    idx = perm_to_idx[p]
                    affected[idx] = affected.get(idx, 0) + 1

            safe = True
            for idx, delta in affected.items():
                if counts[idx] + delta < 0:
                    safe = False
                    break
            
            if safe:
                for idx, delta in affected.items():
                    counts[idx] += delta
                w.pop(i)
                improved = True
                break
        
        if not improved:
            break

    return "".join(map(str, w))

