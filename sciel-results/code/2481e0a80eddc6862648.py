"""Optimized construction: max 3-AP, no 4-AP subset of [1,256].
Uses greedy initialization + local search + simulated annealing with
vectorized numpy operations for speed.
"""
import numpy as np
import time
import random

N = 256
DEADLINE = 85  # seconds, safety margin before 90s kill

def _precompute():
    """Precompute all 3-APs and 4-APs as index pairs/sets."""
    # All 3-APs: (a, b, c) where a < b < c and a+c = 2b
    # Represent as (a-1, c-1) zero-indexed pair, with b determined
    three_aps = []
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1):
            if (a + c) % 2 == 0:
                b = (a + c) // 2
                if a < b < c:
                    three_aps.append((a - 1, c - 1))  # zero-indexed endpoints
    # All 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            a4 = a + 3 * d
            if a4 <= N:
                four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return three_aps, four_aps

def _count_3aps(mask, three_aps):
    """Count 3-APs in a boolean mask (zero-indexed positions 0..255)."""
    # mask[i] is True if element (i+1) is in the set
    # For each 3-AP (a, c) zero-indexed, b = (a+c)//2
    # Need mask[a] and mask[c] both True (b is determined)
    a_idx = np.array([p[0] for p in three_aps], dtype=np.int32)
    c_idx = np.array([p[1] for p in three_aps], dtype=np.int32)
    return int(np.sum(mask[a_idx] & mask[c_idx]))

def _has_4ap(mask, four_aps):
    """Check if mask contains any 4-AP."""
    for ap in four_aps:
        if all(mask[i] for i in ap):
            return True
    return False

def _delta_3aps_add(mask, idx, three_aps, ap_index):
    """Count how many 3-APs would be formed by adding element at zero-index idx."""
    # A 3-AP (a, c) is formed if idx == a or idx == c, and the other endpoint is in mask
    # b is automatically (a+c)//2, no constraint on b being in set
    # Actually for 3-AP we need all three in the set: a, b, c
    # So adding idx helps if idx is a, b, or c of a 3-AP where the other two are present
    count = 0
    for (a, c) in three_aps:
        b = (a + c) // 2
        if idx == a:
            if mask[b] and mask[c]:
                count += 1
        elif idx == c:
            if mask[a] and mask[b]:
                count += 1
        elif idx == b:
            if mask[a] and mask[c]:
                count += 1
    return count

def _creates_4ap(mask, idx, four_aps):
    """Check if adding element at zero-index idx creates a 4-AP."""
    for ap in four_aps:
        if idx in ap:
            if all(mask[i] for i in ap if i != idx):
                return True
    return False

def _greedy_construct(three_aps, four_aps):
    """Greedy: add elements in order of maximum 3-AP contribution, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        if not _creates_4ap(mask, idx, four_aps):
            mask[idx] = True
    return mask

def _greedy_smart(three_aps, four_aps):
    """Greedy with lookahead: at each step, pick the element that adds the most 3-APs."""
    mask = np.zeros(N, dtype=bool)
    available = list(range(N))
    while available:
        best_idx = -1
        best_gain = -1
        for idx in available:
            if _creates_4ap(mask, idx, four_aps):
                continue
            gain = _delta_3aps_add(mask, idx, three_aps, None)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1:
            break
        mask[best_idx] = True
        available.remove(best_idx)
    return mask

def _local_search(mask, three_aps, four_aps, max_iters=5000):
    """Local search: try single add/remove/swap moves."""
    best_score = _count_3aps(mask, three_aps)
    current_score = best_score
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try adding elements
        for idx in range(N):
            if mask[idx]:
                continue
            if _creates_4ap(mask, idx, four_aps):
                continue
            new_mask = mask.copy()
            new_mask[idx] = True
            new_score = _count_3aps(new_mask, three_aps)
            if new_score > current_score:
                mask = new_mask
                current_score = new_score
                best_score = max(best_score, new_score)
                improved = True
                break
        if improved:
            continue
        # Try removing elements
        for idx in range(N):
            if not mask[idx]:
                continue
            new_mask = mask.copy()
            new_mask[idx] = False
            new_score = _count_3aps(new_mask, three_aps)
            # Only remove if it allows future additions (for now, skip pure removals)
            # Actually removals decrease score, so skip unless we do swaps
        # Try swaps: remove one, add another
        for remove_idx in range(N):
            if not mask[remove_idx]:
                continue
            for add_idx in range(N):
                if mask[add_idx] or add_idx == remove_idx:
                    continue
                new_mask = mask.copy()
                new_mask[remove_idx] = False
                if _creates_4ap(new_mask, add_idx, four_aps):
                    continue
                new_mask[add_idx] = True
                new_score = _count_3aps(new_mask, three_aps)
                if new_score > current_score:
                    mask = new_mask
                    current_score = new_score
                    best_score = max(best_score, new_score)
                    improved = True
                    break
            if improved:
                break
    return mask, best_score

def _simulated_annealing(mask, three_aps, four_aps, start_temp=10.0, cool_rate=0.999,
                          max_iters=200000, time_limit=None):
    """Simulated annealing with add/remove/swap moves."""
    start_time = time.time()
    current_score = _count_3aps(mask, three_aps)
    best_mask = mask.copy()
    best_score = current_score
    temp = start_temp
    
    for it in range(max_iters):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        
        # Choose a random move
        move_type = random.random()
        new_mask = mask.copy()
        
        if move_type < 0.4:
            # Add an element
            empty = [i for i in range(N) if not mask[i]]
            if not empty:
                continue
            idx = random.choice(empty)
            if _creates_4ap(mask, idx, four_aps):
                continue
            new_mask[idx] = True
        elif move_type < 0.7:
            # Remove an element
            filled = [i for i in range(N) if mask[i]]
            if not filled:
                continue
            idx = random.choice(filled)
            new_mask[idx] = False
        else:
            # Swap
            filled = [i for i in range(N) if mask[i]]
            empty = [i for i in range(N) if not mask[i]]
            if not filled or not empty:
                continue
            remove_idx = random.choice(filled)
            add_idx = random.choice(empty)
            new_mask[remove_idx] = False
            if _creates_4ap(new_mask, add_idx, four_aps):
                continue
            new_mask[add_idx] = True
        
        new_score = _count_3aps(new_mask, three_aps)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < np.exp(delta / max(temp, 1e-9)):
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask.copy()
        
        temp *= cool_rate
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    three_aps, four_aps = _precompute()
    
    # Precompute numpy arrays for fast 3-AP counting
    a_idx = np.array([p[0] for p in three_aps], dtype=np.int32)
    c_idx = np.array([p[1] for p in three_aps], dtype=np.int32)
    
    def count_3aps_fast(mask):
        return int(np.sum(mask[a_idx] & mask[c_idx]))
    
    # Phase 1: Greedy initialization
    mask = _greedy_construct(three_aps, four_aps)
    best_mask = mask.copy()
    best_score = count_3aps_fast(mask)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return sorted(i + 1 for i in range(N) if best_mask[i])
    
    # Phase 2: Local search
    mask, score = _local_search(mask, three_aps, four_aps, max_iters=2000)
    if score > best_score:
        best_score = score
        best_mask = mask.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return sorted(i + 1 for i in range(N) if best_mask[i])
    
    # Phase 3: Simulated annealing with multiple restarts
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        # Try SA from current best
        sa_time = min(remaining * 0.6, 40)
        mask, score = _simulated_annealing(
            mask, three_aps, four_aps,
            start_temp=5.0, cool_rate=0.9995,
            max_iters=1000000, time_limit=sa_time
        )
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
        
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 3
        if remaining > 5:
            # Try a few random restarts
            for restart in range(3):
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Random mask with ~40% density
                rand_mask = np.random.random(N) < 0.4
                # Remove 4-APs
                for ap in four_aps:
                    if all(rand_mask[i] for i in ap):
                        rand_mask[random.choice(ap)] = False
                # Local search
                rand_mask, rs = _local_search(rand_mask, three_aps, four_aps, max_iters=500)
                if rs > best_score:
                    best_score = rs
                    best_mask = rand_mask.copy()
    
    # Final validation
    final_mask = best_mask.copy()
    # Ensure no 4-APs
    for ap in four_aps:
        if all(final_mask[i] for i in ap):
            final_mask[ap[0]] = False
    
    result = sorted(i + 1 for i in range(N) if final_mask[i])
    if not result:
        result = [1]
    return result

# Precompute at module level for speed
_three_aps, _four_aps = _precompute()
_a_idx = np.array([p[0] for p in _three_aps], dtype=np.int32)
_c_idx = np.array([p[1] for p in _three_aps], dtype=np.int32)

def _fast_count_3aps(mask):
    return int(np.sum(mask[_a_idx] & mask[_c_idx]))

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy initialization (fast)
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        if not _creates_4ap(mask, idx, _four_aps):
            mask[idx] = True
    best_mask = mask.copy()
    best_score = _fast_count_3aps(mask)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        return sorted(i + 1 for i in range(N) if best_mask[i])
    
    # Phase 2: Local search (add moves)
    improved = True
    max_ls_iters = 3000
    iters = 0
    while improved and iters < max_ls_iters and (time.time() - start_time) < DEADLINE - 20:
        improved = False
        iters += 1
        for idx in range(N):
            if mask[idx]:
                continue
            if _creates_4ap(mask, idx, _four_aps):
                continue
            new_mask = mask.copy()
            new_mask[idx] = True
            new_score = _fast_count_3aps(new_mask)
            if new_score > _fast_count_3aps(mask):
                mask = new_mask
                improved = True
                break
    if _fast_count_3aps(mask) > best_score:
        best_score = _fast_count_3aps(mask)
        best_mask = mask.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 25:
        return sorted(i + 1 for i in range(N) if best_mask[i])
    
    # Phase 3: Simulated annealing
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        sa_mask, sa_score = _simulated_annealing(
            mask, _three_aps, _four_aps,
            start_temp=5.0, cool_rate=0.9995,
            max_iters=1000000, time_limit=remaining
        )
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask.copy()
    
    # Final validation
    final_mask = best_mask.copy()
    for ap in _four_aps:
        if all(final_mask[i] for i in ap):
            final_mask[ap[0]] = False
    
    result = sorted(i + 1 for i in range(N) if final_mask[i])
    if not result:
        result = [1]
    return result
