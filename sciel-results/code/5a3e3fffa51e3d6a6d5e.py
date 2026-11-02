import itertools
import time
import random

def construct(n):
    if n < 7:
        # Use classic construction for small n
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
        return "".join(map(str, word))
    
    # For n=7, try to optimize the order of 6-permutations
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Generate standard word to have a baseline
    # We can generate it quickly
    word_std = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word_std) - m + 2):
            window = tuple(word_std[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_std = new
    std_len = len(word_std)
    best_word = word_std
    best_len = std_len
    
    # 2. Optimize for n=7
    # Permutations of 1..6
    perms_6 = list(itertools.permutations(range(1, 7)))
    N = len(perms_6)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms_6)}
    
    # Precompute overlap matrix
    # ov[i][j] = overlap(perms_6[i], perms_6[j])
    ov = [[0]*N for _ in range(N)]
    for i, p in enumerate(perms_6):
        for j, q in enumerate(perms_6):
            # Calculate overlap
            # Max overlap is 5
            for k in range(5, -1, -1):
                if p[6-k:] == q[:k]:
                    ov[i][j] = k
                    break
    
    # Greedy path
    path = [0]
    remaining = set(range(1, N))
    current = 0
    while remaining:
        best_next = -1
        best_ov = -1
        for q in remaining:
            if ov[current][q] > best_ov:
                best_ov = ov[current][q]
                best_next = q
        path.append(best_next)
        remaining.remove(best_next)
        current = best_next
    
    # 2-Opt improvement
    # Calculate initial weight
    total_weight = sum(ov[path[i]][path[i+1]] for i in range(N-1))
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N-1):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, N-1):
                # Reverse segment path[i+1] to path[j]
                # Old edges: (path[i], path[i+1]) and (path[j], path[j+1])
                # New edges: (path[i], path[j]) and (path[i+1], path[j+1])
                old_w = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                new_w = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                if new_w > old_w:
                    total_weight += new_w - old_w
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
    
    # Construct word from optimized path
    word_opt = []
    for idx, p_idx in enumerate(path):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word_opt = block
        else:
            # Find overlap with previous block
            # Previous block ended with p_prev
            # Current block starts with p
            # Overlap is suffix of prev_block matching prefix of block
            # But we know the structure: prev_block is p_prev, 7, p_prev
            # So suffix is p_prev. Prefix is p.
            # Overlap is L(p_prev, p)
            prev_p_idx = path[idx-1]
            ov_len = ov[prev_p_idx][p_idx]
            word_opt += block[ov_len:]
    
    opt_len = len(word_opt)
    
    if opt_len < best_len:
        best_word = word_opt
        best_len = opt_len
    
    return "".join(map(str, best_word))
