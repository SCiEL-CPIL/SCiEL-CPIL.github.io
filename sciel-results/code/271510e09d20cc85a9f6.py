import time
import itertools

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 12
    m = 40
    
    # Precompute all 2^12 masks
    all_masks = list(range(1 << n))
    
    # Subset check: a is strict subset of b iff (a & b) == a and a != b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    # Start with a full chain: C_i = {0, 1, ..., i-1} for i in 0..12
    # Chain masks: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    chain_masks = [(1 << i) - 1 for i in range(n + 1)]  # 13 sets
    sets = list(chain_masks)
    
    # Count comparable pairs in current set
    def count_pairs(ms):
        cnt = 0
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if is_strict_subset(ms[i], ms[j]):
                    cnt += 1
                elif is_strict_subset(ms[j], ms[i]):
                    cnt += 1
        return cnt
    
    # Greedily add m - len(sets) more masks
    remaining = m - len(sets)
    
    for _ in range(remaining):
        if time.time() > deadline - 2:
            break
        best_mask = -1
        best_gain = -1
        current_count = count_pairs(sets)
        
        # Try each unused mask
        used = set(sets)
        for mask in all_masks:
            if mask in used:
                continue
            # Count how many new comparable pairs this mask would add
            gain = 0
            for s in sets:
                if is_strict_subset(mask, s) or is_strict_subset(s, mask):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask >= 0:
            sets.append(best_mask)
    
    # Local search: try swapping sets to improve
    if time.time() < deadline - 5:
        current_count = count_pairs(sets)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(sets)):
                if time.time() > deadline - 3:
                    break
                for mask in all_masks:
                    if mask in sets:
                        continue
                    old = sets[i]
                    sets[i] = mask
                    new_count = count_pairs(sets)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                        break
                    sets[i] = old
                if improved:
                    break
    
    # Convert to lists of distinct integers
    result = []
    for mask in sets:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
