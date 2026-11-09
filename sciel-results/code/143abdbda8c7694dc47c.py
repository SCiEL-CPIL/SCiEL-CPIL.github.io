"""Reciprocal weight maximization for 3-AP-free subsets of [1,256]."""
import time
import random
from fractions import Fraction

N = 256
DEADLINE_MARGIN = 10  # seconds to leave before the 90s kill

def _has_3ap_bitset(bits):
    """Check if bitset (int, bit i set means i+1 in set) contains a 3-AP."""
    for d in range(1, N // 2 + 1):
        if (bits & (bits >> d) & (bits >> 2 * d)) != 0:
            return True
    return False

def _can_add(bits, x):
    """Check if adding element x (1-indexed) to set represented by bits keeps it 3-AP-free."""
    # x as first element: x, x+d, x+2d for d>=1, x+2d <= N
    for d in range(1, (N - x + 1) // 2 + 1):
        if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
            return False
    # x as middle: x-d, x, x+d for d>=1, x-d >= 1
    for d in range(1, x):
        if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
            return False
    # x as last: x-2d, x-d, x for d>=1, x-2d >= 1
    for d in range(1, (x + 1) // 2):
        if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
            return False
    return True

def _swap_valid(bits, x_remove, x_add):
    """Check if swapping x_remove with x_add keeps the set 3-AP-free.
    Only need to check 3-APs that involve x_add (since x_remove was already valid).
    """
    # After removing x_remove and adding x_add, check 3-APs involving x_add
    # x_add as first: x_add, x_add+d, x_add+2d
    for d in range(1, (N - x_add + 1) // 2 + 1):
        a = x_add + d
        c = x_add + 2 * d
        if a == x_remove or c == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
            return False
    # x_add as middle: x_add-d, x_add, x_add+d
    for d in range(1, x_add):
        a = x_add - d
        c = x_add + d
        if a == x_remove or c == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
            return False
    # x_add as last: x_add-2d, x_add-d, x_add
    for d in range(1, (x_add + 1) // 2):
        a = x_add - 2 * d
        b = x_add - d
        if a == x_remove or b == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (b - 1)) & 1:
            return False
    return True

def _reciprocal_sum_float(bits):
    """Compute floating sum of reciprocals."""
    total = 0.0
    for i in range(N):
        if (bits >> i) & 1:
            total += 1.0 / (i + 1)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Phase 1: Greedy construction prioritizing high reciprocal weight
    best_bits = 0
    best_score = 0.0
    
    # Strategy 1: Greedy by 1/x descending (smallest x first)
    bits = 0
    for x in range(1, N + 1):
        if _can_add(bits, x):
            bits |= (1 << (x - 1))
    score = _reciprocal_sum_float(bits)
    if score > best_score:
        best_score = score
        best_bits = bits
    
    # Phase 2: Simulated annealing with fast swap validity check
    if time.time() > deadline:
        return _bits_to_list(best_bits)
    
    rng = random.Random(42)
    
    current_bits = best_bits
    current_score = best_score
    best_bits_sa = best_bits
    best_score_sa = best_score
    
    def bits_to_list(b):
        return [i + 1 for i in range(N) if (b >> i) & 1]
    
    # SA parameters
    T_start = 0.01
    T_end = 0.0001
    alpha = 0.995  # cooling rate
    
    current_T = T_start
    while current_T > T_end:
        if time.time() > deadline:
            break
        
        # Generate a swap move: remove one element, add another
        cur_list = bits_to_list(current_bits)
        if len(cur_list) < 2:
            break
        
        # Remove a random element
        idx_remove = rng.randint(0, len(cur_list) - 1)
        x_remove = cur_list[idx_remove]
        
        # Try to add a random element not in set
        if rng.random() < 0.7:
            # Pick from small numbers not in set, weighted by 1/x
            candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
            if not candidates:
                continue
            weights = [1.0 / x for x in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            x_add = candidates[-1]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    x_add = c
                    break
        else:
            candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
            if not candidates:
                continue
            x_add = candidates[rng.randint(0, len(candidates) - 1)]
        
        if x_remove == x_add:
            continue
        
        # Fast validity check: only check 3-APs involving x_add
        if not _swap_valid(current_bits, x_remove, x_add):
            continue
        
        new_bits = (current_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
        new_score = current_score - 1.0 / x_remove + 1.0 / x_add
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < pow(2.718281828, delta / current_T):
            current_bits = new_bits
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_bits_sa = current_bits
        
        current_T *= alpha
    
    if best_score_sa > best_score:
        best_bits = best_bits_sa
        best_score = best_score_sa
    
    # Phase 3: Local search - try all single swaps to find improvements
    if time.time() > deadline:
        return _bits_to_list(best_bits)
    
    improved = True
    while improved:
        improved = False
        cur_list = bits_to_list(best_bits)
        for x_remove in cur_list:
            if time.time() > deadline:
                break
            for x_add in range(1, N + 1):
                if x_add == x_remove or ((best_bits >> (x_add - 1)) & 1):
                    continue
                if not _swap_valid(best_bits, x_remove, x_add):
                    continue
                new_score = best_score - 1.0 / x_remove + 1.0 / x_add
                if new_score > best_score:
                    best_bits = (best_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                    best_score = new_score
                    improved = True
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return _bits_to_list(best_bits)

def _bits_to_list(bits):
    return [i + 1 for i in range(N) if (bits >> i) & 1]
