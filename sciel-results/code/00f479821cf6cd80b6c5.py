import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Split into 8-bit low part and 9-bit high part
    LOW_BITS = 8
    HIGH_BITS = 9
    LOW_SIZE = 1 << LOW_BITS   # 256
    HIGH_SIZE = 1 << HIGH_BITS  # 512
    
    # Precompute popcount for 8-bit and 9-bit values
    pop8 = np.array([i.bit_count() for i in range(LOW_SIZE)], dtype=np.uint8)
    pop9 = np.array([i.bit_count() for i in range(HIGH_SIZE)], dtype=np.uint8)
    
    # Precompute XOR popcount matrices for low and high halves
    # dist_low[i][j] = popcount(i ^ j) for 8-bit values
    # dist_high[i][j] = popcount(i ^ j) for 9-bit values
    low_vals = np.arange(LOW_SIZE, dtype=np.uint16)
    high_vals = np.arange(HIGH_SIZE, dtype=np.uint16)
    
    # Compute distance matrices using broadcasting
    dist_low = np.zeros((LOW_SIZE, LOW_SIZE), dtype=np.uint8)
    for i in range(LOW_SIZE):
        dist_low[i] = pop8[low_vals ^ i]
    
    dist_high = np.zeros((HIGH_SIZE, HIGH_SIZE), dtype=np.uint8)
    for i in range(HIGH_SIZE):
        dist_high[i] = pop9[high_vals ^ i]
    
    # A full 17-bit word is (high, low) where high is 9 bits, low is 8 bits
    # Full distance = dist_high[hi1][hi2] + dist_low[lo1][lo2] >= MIN_DIST
    
    # Seed: First-order Reed-Muller code on 16 coordinates, 17th bit = 0
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    code = set(words)
    code_list = list(code)
    
    # For each codeword, extract high and low parts
    def split_word(w):
        low = w & (LOW_SIZE - 1)
        high = (w >> LOW_BITS) & (HIGH_SIZE - 1)
        return high, low
    
    # Track which (high, low) pairs are in the code
    used_pairs = set()
    for w in code_list:
        h, l = split_word(w)
        used_pairs.add((h, l))
    
    # Precompute for each high value, the list of low values that are compatible
    # with each existing codeword given the distance constraint
    
    # For a new word (h_new, l_new) to be compatible with existing (h_old, l_old):
    # dist_high[h_new][h_old] + dist_low[l_new][l_old] >= MIN_DIST
    
    # Strategy: For each candidate (h_new, l_new), check compatibility with all existing words.
    # But we need to do this efficiently.
    
    # Alternative: Precompute for each pair of high values (h1, h2), the minimum
    # dist_low required: need dist_low[l1][l2] >= MIN_DIST - dist_high[h1][h2]
    
    # If dist_high[h1][h2] >= MIN_DIST, then any low pair works.
    # If dist_high[h1][h2] < MIN_DIST, then we need dist_low[l1][l2] >= MIN_DIST - dist_high[h1][h2]
    
    # Build a compatibility table:
    # For each (h1, h2) pair, store the threshold for dist_low
    # threshold[h1][h2] = max(0, MIN_DIST - dist_high[h1][h2])
    
    threshold = np.zeros((HIGH_SIZE, HIGH_SIZE), dtype=np.uint8)
    for i in range(HIGH_SIZE):
        for j in range(HIGH_SIZE):
            dh = int(dist_high[i][j])
            threshold[i][j] = max(0, MIN_DIST - dh)
    
    # For each low pair (l1, l2), we know dist_low[l1][l2]
    # A pair (h1, l1) and (h2, l2) is compatible iff dist_low[l1][l2] >= threshold[h1][h2]
    
    # Precompute for each (h1, h2) the set of low pairs that are compatible
    # Actually, let's precompute: for each high pair (h1, h2), which low values l2 are compatible
    # with a given l1? That's too much.
    
    # Better approach: For each candidate word (h, l), compute its distance to all existing words
    # using the precomputed matrices.
    
    # Let's use a different strategy: maintain for each (h, l) position whether it's used,
    # and for each candidate, quickly compute min distance to all used words.
    
    # For efficiency, maintain:
    # - used_highs: for each high value, list of low values used with that high
    # - For a new candidate (h_new, l_new), for each existing high h_old:
    #   min over l_old in used_highs[h_old] of (dist_high[h_new][h_old] + dist_low[l_new][l_old])
    
    # Let's maintain for each high value h, an array of used low values
    used_lows_per_high = {h: [] for h in range(HIGH_SIZE)}
    for w in code_list:
        h, l = split_word(w)
        used_lows_per_high[h].append(l)
    
    # For faster lookup, also maintain a boolean array for each high
    used_low_mask = np.zeros((HIGH_SIZE, LOW_SIZE), dtype=bool)
    for h in range(HIGH_SIZE):
        for l in used_lows_per_high[h]:
            used_low_mask[h][l] = True
    
    # For a candidate (h_new, l_new), compute min distance to all used words:
    # min over all (h_old, l_old) in used of dist_high[h_new][h_old] + dist_low[l_new][l_old]
    
    # To speed this up, precompute for each (h_new, l_new) the min distance to all used words.
    # But that's 131072 * |code| which is too slow.
    
    # Instead, use a smarter approach: for each high value h_old, precompute the min
    # dist_low[l_new][l_old] over all used l_old for that h_old.
    
    # min_dist_low_for_high[h_old][l_new] = min over l_old in used_lows_per_high[h_old] of dist_low[l_new][l_old]
    
    min_dist_low_for_high = np.full((HIGH_SIZE, LOW_SIZE), 100, dtype=np.uint8)
    for h_old in range(HIGH_SIZE):
        if used_lows_per_high[h_old]:
            l_old_arr = np.array(used_lows_per_high[h_old], dtype=np.uint16)
            # dist_low is (LOW_SIZE, LOW_SIZE), so dist_low[:, l_old] gives dist for all l_new to l_old
            # We want min over l_old of dist_low[l_new, l_old]
            # dist_low[l_new, l_old] for all l_old: take column l_old
            cols = dist_low[:, l_old_arr]  # shape (LOW_SIZE, len(l_old_arr))
            min_dist_low_for_high[h_old] = np.min(cols, axis=1)
    
    # Now for candidate (h_new, l_new):
    # min_dist = min over h_old of (dist_high[h_new][h_old] + min_dist_low_for_high[h_old][l_new])
    
    # Precompute this for all candidates
    # min_dist_to_code[h_new][l_new] = min over h_old of (dist_high[h_new][h_old] + min_dist_low_for_high[h_old][l_new])
    
    min_dist_to_code = np.full((HIGH_SIZE, LOW_SIZE), 100, dtype=np.uint8)
    for h_new in range(HIGH_SIZE):
        for l_new in range(LOW_SIZE):
            min_d = 100
            for h_old in range(HIGH_SIZE):
                if not used_lows_per_high[h_old]:
                    continue
                d = int(dist_high[h_new][h_old]) + int(min_dist_low_for_high[h_old][l_new])
                if d < min_d:
                    min_d = d
            min_dist_to_code[h_new][l_new] = min_d
    
    # Now we can greedily add words by finding the (h, l) with max min_dist_to_code >= MIN_DIST
    # and not already used.
    
    # Create a flat array for easier indexing
    # index = h * LOW_SIZE + l
    min_dist_flat = min_dist_to_code.flatten()
    used_flat = used_low_mask.flatten()
    
    # Greedy loop
    while time.time() < deadline:
        # Find valid candidates: min_dist >= MIN_DIST and not used
        valid_mask = (min_dist_flat >= MIN_DIST) & (~used_flat)
        if not np.any(valid_mask):
            break
        
        valid_indices = np.where(valid_mask)[0]
        valid_dists = min_dist_flat[valid_indices]
        best_local = np.argmax(valid_dists)
        best_idx = int(valid_indices[best_local])
        best_d = int(valid_dists[best_local])
        
        if best_d < MIN_DIST:
            break
        
        # Convert flat index to (h, l)
        h_new = best_idx // LOW_SIZE
        l_new = best_idx % LOW_SIZE
        new_word = (h_new << LOW_BITS) | l_new
        
        code.add(new_word)
        code_list.append(new_word)
        used_pairs.add((h_new, l_new))
        used_low_mask[h_new][l_new] = True
        used_flat[best_idx] = True
        used_lows_per_high[h_new].append(l_new)
        
        # Update min_dist_to_code for all candidates
        # For each (h_old, l_old) that is now used, update:
        # min_dist_to_code[h_new][l_new] = min(min_dist_to_code[h_new][l_new], dist_high[h_new][h_old] + dist_low[l_new][l_old])
        
        # But we only added one word, so we need to update based on that one word.
        # Actually, min_dist_to_code already accounts for all used words.
        # When we add a new word, we need to recompute min_dist_to_code to include it.
        
        # The new word is (h_new, l_new). For any candidate (h_c, l_c):
        # new_min = min(old_min, dist_high[h_c][h_new] + dist_low[l_c][l_new])
        
        # Update min_dist_to_code
        for h_c in range(HIGH_SIZE):
            dh = dist_high[h_c, h_new]  # shape (LOW_SIZE,)
            dl = dist_low[:, l_new]  # shape (LOW_SIZE,)
            new_dists = dh + dl  # shape (LOW_SIZE,)
            min_dist_to_code[h_c] = np.minimum(min_dist_to_code[h_c], new_dists)
        
        min_dist_flat = min_dist_to_code.flatten()
    
    return list(code)
