"""Combined program: many randomized greedy restarts (proven mechanism) + fast swap local search."""
import time
import random
import numpy as np

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for d in _pos_last[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x - 2 * d)) & 1 and (bits >> (x - 3 * d)) & 1:
            return False
    for d in _pos_first[x]:
        if (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1 and (bits >> (x + 3 * d)) & 1:
            return False
    for d in _pos_second[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1:
            return False
    for d in _pos_third[x]:
        if (bits >> (x - 2 * d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1:
            return False
    return True


def _is_4ap_free_np(selected):
    """Vectorized check using NumPy."""
    if not selected:
        return True
    arr = np.zeros(MAX + 1, dtype=bool)
    arr[list(selected)] = True
    s = np.nonzero(arr)[0]
    if len(s) == 0:
        return True
    a_arr = s.astype(np.int64)
    for a in a_arr:
        dm = (N - a) // 3
        if dm < 1:
            continue
        d = np.arange(1, dm + 1)
        t1 = a + d
        t2 = a + 2 * d
        t3 = a + 3 * d
        if np.any(arr[t1] & arr[t2] & arr[t3]):
            return False
    return True


def _greedy_forward_bitset():
    """Fast forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random_order_bitset(rng):
    """Random-order greedy using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}


def _local_improve_bitset(bits, deadline):
    """Fast local search using bitset."""
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    # Precompute list of set bits for faster iteration? 
    # Or just iterate 1..N and check bit.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current_bits >> x) & 1) and _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
        
        # Pass 2: Try swaps (remove b, add x)
        # We look for x not in set, and b in set such that removing b allows x.
        # To be fast, we only try if x is blocked by exactly one element b? 
        # Or just try all pairs? That's too slow.
        # Strategy: For each x not in set, find blockers. If 1 blocker, try swap.
        
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers
            blockers = []
            for d in _pos_last[x]:
                a, b, c = x - d, x - 2 * d, x - 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_first[x]:
                a, b, c = x + d, x + 2 * d, x + 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_second[x]:
                a, b, c = x - d, x + d, x + 2 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_third[x]:
                a, b, c = x - 2 * d, x - d, x + d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            
            if not blockers:
                continue # Should have been caught in Pass 1, but just in case
                
            # Only try if number of unique blockers is small (e.g., <= 3)
            # If many blockers, swap is unlikely to help or too expensive.
            unique_blockers = set(blockers)
            if len(unique_blockers) > 3:
                continue
                
            for b in unique_blockers:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Quick re-fill: try to add back some elements? 
                    # Or just check if size increased? 
                    # Removing 1, adding 1 -> size same. 
                    # We need re-fill to gain.
                    
                    # Greedy re-fill on candidate
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, N + 1):
                            if time.time() > deadline - 2:
                                break
                            if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                candidate_bits |= (1 << y)
                                changed = True
                                
                    cand_size = bin(candidate_bits).count('1')
                    if cand_size > current_size:
                        current_bits = candidate_bits
                        current_size = cand_size
                        improved = True
                        break # Restart pass 2 for this x? Or just continue. 
                        # Breaking inner loop, continue outer.
    
    return current_bits


def construct():
    deadline = time.time() + 50  # Safety margin within 60s limit

    # 1) Deterministic forward greedy (fast)
    best_bits = _greedy_forward_bitset()
    best_size = bin(best_bits).count('1')

    # 2) Many randomized restarts
    rng = random.Random(42)
    while time.time() < deadline - 10:
        bits = _greedy_random_order_bitset(rng)
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size

    # 3) Local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size

    # 4) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 5) Final verification; fallback if invalid
    if not _is_4ap_free_np(best_set):
        best_set = _bits_to_set(_greedy_forward_bitset())

    return sorted(best_set)
