"""Combined program: Cantor set seed + many randomized greedy restarts + fast swap local search with bitset verification."""
import time
import random

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


def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for d in _pos_last[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - 3 * d)) & 1):
                return False
        for d in _pos_first[x]:
            if ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1) and ((bits >> (x + 3 * d)) & 1):
                return False
        for d in _pos_second[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1):
                return False
        for d in _pos_third[x]:
            if ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1):
                return False
    return True


def _greedy_forward_bitset():
    """Fast forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_reverse_order_bitset():
    """Fast reverse-order greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
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


def _greedy_random_reverse_bitset(rng):
    """Random-order greedy with reverse bias using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    # Bias towards larger numbers first
    order.sort(key=lambda x: -x if rng.random() < 0.7 else x)
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


def _cantor_set_bits():
    """
    Generate a 4-AP-free set using the base-3 Cantor set construction.
    Numbers in [1, N] whose base-3 representation contains only 0s and 1s
    (no digit 2) form a 4-AP-free set.
    """
    bits = 0
    for x in range(1, N + 1):
        # Check if base-3 representation of x contains only 0s and 1s
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits


def _cantor_local_improve(bits, deadline):
    """
    Local improvement starting from a Cantor-set seed.
    Uses a more aggressive add-then-swap strategy.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
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
        
        # Pass 2: Try swaps (remove b, add x) with re-fill
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers
            blockers = set()
            for d in _pos_last[x]:
                a, b, c = x - d, x - 2 * d, x - 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_first[x]:
                a, b, c = x + d, x + 2 * d, x + 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_second[x]:
                a, b, c = x - d, x + d, x + 2 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_third[x]:
                a, b, c = x - 2 * d, x - d, x + d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            
            if not blockers:
                continue
                
            # Only try if number of unique blockers is small
            if len(blockers) > 4:
                continue
                
            for b in blockers:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
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
                        break
    
    return current_bits


def construct():
    deadline = time.time() + 55  # Safety margin within 60s limit

    # 1) Start with Cantor set (strong structured seed ~200 elements)
    cantor_bits = _cantor_set_bits()
    cantor_size = bin(cantor_bits).count('1')
    best_bits = cantor_bits
    best_size = cantor_size

    # 1b) Also try deterministic forward greedy
    fwd_bits = _greedy_forward_bitset()
    fwd_size = bin(fwd_bits).count('1')
    if fwd_size > best_size:
        best_bits = fwd_bits
        best_size = fwd_size

    # 1c) Deterministic reverse greedy
    rev_bits = _greedy_reverse_order_bitset()
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_bits = rev_bits
        best_size = rev_size

    # 2) Local improvement on Cantor seed (this is the key phase)
    if time.time() < deadline - 15:
        cantor_improved = _cantor_local_improve(cantor_bits, deadline)
        cantor_imp_size = bin(cantor_improved).count('1')
        if cantor_imp_size > best_size:
            best_bits = cantor_improved
            best_size = cantor_imp_size

    # 3) Many randomized restarts with diverse strategies
    rng = random.Random(42)
    while time.time() < deadline - 10:
        if rng.random() < 0.5:
            bits = _greedy_random_order_bitset(rng)
        else:
            bits = _greedy_random_reverse_bitset(rng)
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size

    # 4) Final local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size

    # 5) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 6) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        # Fallback to Cantor set (always valid)
        best_set = _bits_to_set(_cantor_set_bits())

    return sorted(best_set)
