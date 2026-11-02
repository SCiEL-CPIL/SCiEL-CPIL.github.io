import time
import random
import numpy as np
from itertools import product
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute neighbor lists for fast updates
    # neighbors[i] = list of indices covered by word i (including i)
    neighbors = []
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
    
    # Convert neighbors to numpy arrays for faster indexing and min operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]

    # 1. Initial Construction: Greedy Set Cover
    covered = np.zeros(N, dtype=bool)
    codewords = []
    remaining = set(range(N))
    
    # Precompute bitsets for each ball
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors_np[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    is_chosen = np.zeros(N, dtype=bool)
    
    # Greedy Set Cover
    while current_cover_mask != all_mask:
        if time.time() > deadline - 60:
            break # Safety break
            
        best_i = -1
        best_count = 0
        
        # Scan all candidates
        for i in range(N):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
                    
        if best_i == -1:
            break # Should not happen if mask != all_mask
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure we have a valid cover. If we broke early, finish it.
    if current_cover_mask != all_mask:
        # Fallback: add random words until covered
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            current_cover_mask |= ball_bitsets[i]
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            if time.time() > deadline - 10:
                break

    # 2. Local Search
    # We have a valid cover. We want to minimize len(codewords).
    # State: codewords (list), cover_count (np array of int8/int16)
    
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors_np[cw]] += 1

    def try_remove(cw_idx):
        b = neighbors_np[cw_idx]
        if np.min(cover_count[b]) > 1:
            cover_count[b] -= 1
            return True
        return False

    # Redundancy Removal Pass
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 30:
            break

    # Local Search Loop: Systematic Swap/Remove
    random.seed(42)
    all_indices = list(range(N))
    
    # Precompute a list of indices for each ball? No, bitsets are enough.
    
    while time.time() < deadline - 2:
        improved = False
        # Shuffle codewords to avoid bias
        random.shuffle(codewords)
        
        # Create a shuffled list of all candidates for swap search
        # We do this once per outer loop to reduce overhead
        shuffled_candidates = all_indices[:]
        random.shuffle(shuffled_candidates)
        
        for idx, cw in enumerate(codewords):
            if time.time() > deadline - 2:
                break
            
            # 1. Check for removal
            b = neighbors_np[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(idx)
                improved = True
                break # Restart scan as indices changed
            
            # 2. Check for swap
            # Find vulnerable points: points in b with cover_count == 1
            counts = cover_count[b]
            vuln_mask = 0
            if np.any(counts == 1):
                vuln_positions = np.where(counts == 1)[0]
                vuln_ids = b[vuln_positions] # b is now numpy array, this works
                for vid in vuln_ids:
                    vuln_mask |= (1 << vid)
            
            if vuln_mask != 0:
                # Try to find a cw_new that covers vuln_mask
                found_swap = False
                for cw_new in shuffled_candidates:
                    if cw_new == cw:
                        continue
                    # Check if ball_bitsets[cw_new] covers vuln_mask
                    if (vuln_mask & ~ball_bitsets[cw_new]) == 0:
                        # Valid swap
                        # Update cover_count
                        cover_count[b] -= 1
                        cover_count[neighbors_np[cw_new]] += 1
                        codewords[idx] = cw_new
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart scan
    
    # Final redundancy removal pass
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1

    # Final redundancy removal pass
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
    
    # Final Result
    result = [list(all_words[cw]) for cw in codewords]
    return result
