import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # 1. Generate all words and compute balls
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    balls = [0] * total_words
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        # Fallback for general radius (slower, but n is small)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            
    # 2. Compute coverers: coverers[v] is bitmask of codewords covering v
    coverers = [0] * total_words
    for i in range(total_words):
        b = balls[i]
        while b:
            lsb = b & -b
            v = lsb.bit_length() - 1
            coverers[v] |= (1 << i)
            b ^= lsb

    # 3. Greedy Heuristic to get initial upper bound
    # This mimics the current program's logic but optimized
    def greedy_cover():
        uncovered = (1 << total_words) - 1
        selected = []
        while uncovered:
            best_idx = -1
            max_cover = 0
            # To speed up, we can randomize or use a heuristic
            # For determinism and speed in initial bound, use simple max
            for i in range(total_words):
                c = (balls[i] & uncovered).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1: break
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Removal phase
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                if (union_rest & ((1 << total_words) - 1)) == ((1 << total_words) - 1):
                    selected.pop(i)
                    changed = True
                    break
        return selected

    init_selected = greedy_cover()
    best_size = len(init_selected)
    best_indices = init_selected
    
    # 4. Branch and Bound
    start_time = time.time()
    time_limit = 110.0
    
    # Precompute max ball size for LB
    max_ball_size = max(b.bit_count() for b in balls)
    
    # Global state for B&B
    # Using nonlocal variables in a nested function or a class. 
    # Here we use a list to hold mutable state for speed/clarity in this context.
    state = {
        'best_size': best_size,
        'best_indices': best_indices,
        'timeout': False
    }

    def bb_solve(uncovered, selected_mask, count):
        if time.time() - start_time > time_limit:
            state['timeout'] = True
            return
            
        if uncovered == 0:
            if count < state['best_size']:
                state['best_size'] = count
                state['best_indices'] = [i for i in range(total_words) if (selected_mask >> i) & 1]
            return
        
        # Lower Bound
        lb = (uncovered.bit_count() + max_ball_size - 1) // max_ball_size
        if count + lb >= state['best_size']:
            return
            
        # Find pivot vertex with min candidates
        min_count = 1000
        pivot_v = -1
        candidates_mask = 0
        
        u = uncovered
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            u ^= lsb
            
            avail = coverers[v] & ~selected_mask
            cnt = avail.bit_count()
            
            if cnt == 0:
                return # Uncoverable
            
            if cnt < min_count:
                min_count = cnt
                pivot_v = v
                candidates_mask = avail
                if cnt == 1:
                    break
        
        if min_count == 1:
            # Force pick
            c_bit = candidates_mask
            c_idx = c_bit.bit_length() - 1
            new_uncovered = uncovered & ~balls[c_idx]
            new_selected = selected_mask | c_bit
            bb_solve(new_uncovered, new_selected, count + 1)
        else:
            # Branch
            cands = []
            c = candidates_mask
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                c ^= lsb
                cands.append(idx)
            
            # Sort by coverage (descending) to find good solutions early
            # This can be expensive, so only do it if list is small? 
            # List size is <= 15 for R=1. So it's fast.
            cands.sort(key=lambda i: (balls[i] & uncovered).bit_count(), reverse=True)
            
            for idx in cands:
                c_bit = 1 << idx
                new_uncovered = uncovered & ~balls[idx]
                new_selected = selected_mask | c_bit
                bb_solve(new_uncovered, new_selected, count + 1)
                
                if state['timeout']:
                    return

    # Start B&B
    # Initial uncovered is all ones
    initial_uncovered = (1 << total_words) - 1
    bb_solve(initial_uncovered, 0, 0)
    
    if state['best_size'] < best_size:
        return [all_words[i] for i in state['best_indices']]
    else:
        return [all_words[i] for i in best_indices]
