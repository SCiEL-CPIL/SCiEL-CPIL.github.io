# EVOLVE-BLOCK-START
import time
import random

def _can_add(S, cand, bitset):
    """
    Check if cand can be added to S without violating the mean condition.
    S is a list of existing elements (not including cand).
    bitset is a list of bitsets for subset sums of S (already computed).
    Returns True if cand can be safely added.
    """
    n = len(S)
    if n < 1:
        return True
    
    max_sum = n * 512
    
    # Check 1: Is cand the mean of any subset of S with size >= 2?
    for k in range(2, n + 1):
        target = k * cand
        if target <= max_sum and (bitset[k] >> target) & 1:
            return False
    
    # Check 2: For each a in S, is a the mean of a subset including cand?
    # Subset = T ∪ {cand}, T ⊆ S\{a}, |T| >= 1
    # sum(T) + cand = a * (|T| + 1) => sum(T) = a*(|T|+1) - cand
    # Approximate using S bitset (may reject valid adds but safe)
    for a in S:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if 0 <= target <= max_sum and (bitset[k] >> target) & 1:
                return False
    
    return True

def _update_bitset(bitset, S, cand):
    """
    Update bitset to include cand in S.
    Returns new bitset list.
    """
    n = len(S) + 1
    new_bitset = [0] * (n + 1)
    new_bitset[0] = 1
    
    # Copy existing bitsets
    for k in range(len(bitset)):
        new_bitset[k] = bitset[k]
    
    # Add contributions from subsets including cand
    for k in range(n, 0, -1):
        if k - 1 < len(bitset):
            new_bitset[k] |= (bitset[k - 1] << cand)
    
    return new_bitset

def _conflict_score(x, all_numbers):
    """
    Refined conflict score: count 2-element and 3-element AP conflicts.
    """
    count_2 = 0
    count_3 = 0
    for a in all_numbers:
        if a == x:
            continue
        b = 2 * x - a
        if b != x and b != a and b in all_numbers and b > x:
            count_2 += 1
        if (a + x) % 2 == 0:
            m = (a + x) // 2
            if m != a and m != x and m in all_numbers:
                count_3 += 1
    
    centrality = 512 - abs(x - 256)
    score = count_2 * 5 + count_3 * 3 + centrality // 20
    
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute conflict scores for all numbers 1..512
    all_numbers = set(range(1, 513))
    conflict_scores = {}
    for x in range(1, 513):
        conflict_scores[x] = _conflict_score(x, all_numbers)
    
    # Sort candidates by conflict score (fewer conflicts first)
    candidates_sorted = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    
    max_restarts = 120
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Choose candidate ordering
        if restart == 0:
            candidates = list(candidates_sorted)
        elif restart == 1:
            candidates = list(reversed(candidates_sorted))
        elif restart % 3 == 0:
            candidates = list(candidates_sorted)
            random.shuffle(candidates)
        else:
            # Weighted random selection
            candidates = list(range(1, 513))
            weights = [1.0 / (1.0 + conflict_scores[x] * 0.05) for x in candidates]
            selected = []
            remaining = list(range(512))
            remaining_weights = weights[:]
            for _ in range(512):
                if not remaining:
                    break
                r = random.random() * sum(remaining_weights)
                cum = 0
                for i, w in enumerate(remaining_weights):
                    cum += w
                    if r <= cum:
                        selected.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break
            candidates = selected
        
        current_set = []
        current_set_set = set()
        bitset = [1]  # dp[0] = 1, no elements, sum 0
        
        for cand in candidates:
            if time.time() > deadline - 10.0:
                break
            
            if cand in current_set_set:
                continue
            
            if _can_add(current_set, cand, bitset):
                current_set.append(cand)
                current_set_set.add(cand)
                bitset = _update_bitset(bitset, current_set[:-1], cand)
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            
            if best_len >= 48:
                return best_set
    
    # Try to add any remaining numbers to best_set
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: conflict_scores[x])
        
        # Recompute bitset for best_set
        bitset = [1]
        for i, x in enumerate(best_set):
            bitset = _update_bitset(bitset, best_set[:i], x)
        
        for cand in remaining:
            if time.time() > deadline - 2.0:
                break
            if _can_add(best_set, cand, bitset):
                best_set.append(cand)
                bitset = _update_bitset(bitset, best_set[:-1], cand)
    
    return best_set

# EVOLVE-BLOCK-END
