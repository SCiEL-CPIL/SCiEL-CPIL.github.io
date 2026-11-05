import time
import sys
import itertools

def _overlap_len(a, b, max_overlap=None):
    """Find maximum overlap between end of list a and start of list b."""
    if not a or not b:
        return 0
    if max_overlap is None:
        max_overlap = min(len(a), len(b))
    for o in range(max_overlap, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _greedy_order(blocks, deadline):
    """Reorder blocks using greedy nearest-neighbor to maximize overlaps."""
    if len(blocks) <= 2:
        return blocks
    
    n_blocks = len(blocks)
    remaining = list(range(n_blocks))
    ordered = [0]
    remaining.remove(0)
    
    while remaining:
        last_block = blocks[ordered[-1]]
        best_idx = -1
        best_overlap = -1
        for i in remaining:
            overlap = _overlap_len(last_block, blocks[i])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
            if time.time() > deadline:
                break
        if best_idx == -1:
            # Time check: just append remaining in order
            ordered.extend(remaining)
            break
        ordered.append(best_idx)
        remaining.remove(best_idx)
    
    return [blocks[i] for i in ordered]

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Time check
        if time.time() > deadline:
            # If we time out, we need to complete the construction quickly
            # Use a fast path without optimization
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_check = min(len(new), len(block))
                overlap = 0
                for o in range(max_check, 0, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
            continue

        perms, seen = [], set()
        # Extract unique (m-1)-length windows with distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build blocks
        blocks = [list(p) + [m] + list(p) for p in perms]
        
        # For small m, use greedy ordering to find better overlaps
        if m <= 5 and len(blocks) > 2:
            ordered_blocks = _greedy_order(blocks, deadline)
        else:
            ordered_blocks = blocks
        
        new = []
        for block in ordered_blocks:
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    return "".join(map(str, word))
