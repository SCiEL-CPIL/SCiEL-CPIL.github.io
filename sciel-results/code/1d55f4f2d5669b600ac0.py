import time
from itertools import permutations

def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1)-permutations in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For n=7, use local search to improve ordering
        if m == 7:
            result = _optimize_ordering(perms, m, word)
            word = result
        else:
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    return "".join(map(str, word))

def _optimize_ordering(perms, m, base_word):
    """
    For m=7, try to find a better ordering of the 720 permutations
    to maximize overlap when building blocks P, 7, P.
    """
    deadline = time.time() + 115  # 15s safety margin
    
    # Precompute all blocks
    blocks = [list(p) + [m] + list(p) for p in perms]
    n_blocks = len(blocks)
    
    # Compute overlap matrix O[i][j] = max overlap between block i ending and block j starting
    # O[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # For memory and speed, only store useful overlaps
    overlap = [[0] * n_blocks for _ in range(n_blocks)]
    for i in range(n_blocks):
        bi = blocks[i]
        # The tail of bi is the permutation p_i
        # We can try overlaps up to len(p_i)
        for j in range(n_blocks):
            if i == j:
                continue
            bj = blocks[j]
            # Maximum possible overlap is min(len(bi), len(bj)) but realistically
            # the tail of bi is p_i (length m-1) and head of bj is p_j (length m-1)
            # So max overlap is m-1 = 6
            for k in range(min(len(bi), len(bj), m), 0, -1):
                if bi[-k:] == bj[:k]:
                    overlap[i][j] = k
                    break
    
    # Standard ordering: 0, 1, 2, ..., 719
    order = list(range(n_blocks))
    
    # Compute total length for a given order
    def compute_length(ord):
        if not ord:
            return 0
        total = len(blocks[ord[0]])
        for idx in range(1, len(ord)):
            i = ord[idx - 1]
            j = ord[idx]
            total += len(blocks[j]) - overlap[i][j]
        return total
    
    # Also need to account for overlap with base_word end
    # The new word is base_word + first block (with overlap) + subsequent blocks
    # Let's be more precise:
    def compute_total_length(ord):
        if not ord:
            return len(base_word)
        # Overlap between base_word and first block
        first_block = blocks[ord[0]]
        base_tail = base_word
        max_ov = min(len(base_tail), len(first_block))
        ov = 0
        for k in range(max_ov, 0, -1):
            if base_tail[-k:] == first_block[:k]:
                ov = k
                break
        
        total = len(base_word) + len(first_block) - ov
        for idx in range(1, len(ord)):
            i = ord[idx - 1]
            j = ord[idx]
            total += len(blocks[j]) - overlap[i][j]
        return total
    
    current_len = compute_total_length(order)
    best_order = order[:]
    best_len = current_len
    
    # Local search: 2-opt style swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_blocks - 1):
            if time.time() > deadline:
                break
            # Try swapping i and i+1
            if order[i] == order[i+1]:
                continue
            order[i], order[i+1] = order[i+1], order[i]
            new_len = compute_total_length(order)
            if new_len < current_len:
                current_len = new_len
                improved = True
                if current_len < best_len:
                    best_len = current_len
                    best_order = order[:]
            else:
                order[i], order[i+1] = order[i+1], order[i]
        
        # Try 2-opt: reverse segment
        if time.time() < deadline:
            for i in range(n_blocks - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 2, min(i + 10, n_blocks)):
                    if order[i] == order[j]:
                        continue
                    # Reverse segment i..j
                    segment = order[i:j+1]
                    order[i:j+1] = segment[::-1]
                    new_len = compute_total_length(order)
                    if new_len < current_len:
                        current_len = new_len
                        improved = True
                        if current_len < best_len:
                            best_len = current_len
                            best_order = order[:]
                    else:
                        order[i:j+1] = segment[::-1]
    
    order = best_order
    
    # Build the final word
    new = list(base_word)
    for idx, block_idx in enumerate(order):
        block = blocks[block_idx]
        # Find overlap with current new
        max_ov = min(len(new), len(block))
        ov = 0
        for k in range(max_ov, 0, -1):
            if new[-k:] == block[:k]:
                ov = k
                break
        new += block[ov:]
    
    return new
