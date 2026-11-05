import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        perms, seen = [], set()
        # Extract unique (m-1)-length windows with distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Greedy nearest-neighbor ordering to maximize overlaps
        new = []
        remaining = list(perms)
        
        while remaining:
            if time.time() > deadline:
                break
                
            if not new:
                # Start with the first permutation
                best_idx = 0
            else:
                # Find the permutation that gives maximum overlap with current end
                best_overlap = -1
                best_idx = 0
                for idx, p in enumerate(remaining):
                    block = list(p) + [m] + list(p)
                    max_check = min(len(new), len(block))
                    for o in range(max_check, -1, -1):
                        if new[len(new) - o:] == block[:o]:
                            if o > best_overlap:
                                best_overlap = o
                                best_idx = idx
                            break
                    if o == max_check:  # Found full overlap, can't do better
                        break
                
                # Use best overlap found
                overlap = max(0, best_overlap)
            
            p = remaining.pop(best_idx)
            block = list(p) + [m] + list(p)
            
            if not new:
                new = list(block)
            else:
                # Recompute overlap for the chosen block
                max_check = min(len(new), len(block))
                overlap = 0
                for o in range(max_check, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
        
        word = new

    return "".join(map(str, word))
