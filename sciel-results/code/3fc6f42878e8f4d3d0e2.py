"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

N = 512
MAX_A = 511

def _score_fraction(chosen):
    """Compute exact rational score sum 1/(512-a) for a list of chosen integers."""
    if not chosen:
        return Fraction(0, 1)
    s = Fraction(0, 1)
    for a in chosen:
        s += Fraction(1, N - a)
    return s

def _score_float(chosen):
    """Compute float score for quick comparison."""
    return sum(1.0 / (N - a) for a in chosen)

def _build_coprime_bitmasks():
    """For each a in 1..511, build a bitmask of all b in 1..511 that are coprime to a."""
    coprime_masks = [0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        mask = 0
        for b in range(1, MAX_A + 1):
            if gcd(a, b) == 1:
                mask |= (1 << b)
        coprime_masks[a] = mask
    return coprime_masks

def _is_coprime_to_all_bitmask(a, chosen_mask, coprime_masks):
    """Check if a is coprime to all elements in chosen using bitmask."""
    return (coprime_masks[a] & chosen_mask) == chosen_mask

def _greedy_construct(order, deadline, coprime_masks):
    """Greedy construction following a given order."""
    chosen = []
    chosen_mask = 0
    for a in order:
        if time.time() > deadline:
            break
        if _is_coprime_to_all_bitmask(a, chosen_mask, coprime_masks):
            chosen.append(a)
            chosen_mask |= (1 << a)
    return chosen

def _local_search(chosen, deadline, coprime_masks):
    """Fast local search using bitmask coprimality checks."""
    chosen_set = set(chosen)
    chosen_mask = 0
    for a in chosen:
        chosen_mask |= (1 << a)
    
    best_score = _score_float(chosen)
    best_chosen = list(chosen)
    best_mask = chosen_mask
    best_set = chosen_set
    
    # Precompute score contributions
    score_vals = {a: 1.0 / (N - a) for a in range(1, MAX_A + 1)}
    
    def _update_best(candidate, new_score):
        nonlocal best_score, best_chosen, best_mask, best_set
        best_score = new_score
        best_chosen = candidate
        best_mask = 0
        for a in candidate:
            best_mask |= (1 << a)
        best_set = set(candidate)
    
    # Phase 1: Try all single additions
    while True:
        if time.time() > deadline:
            break
        improved = False
        for a in range(MAX_A, 0, -1):
            if a in best_set:
                continue
            if _is_coprime_to_all_bitmask(a, best_mask, coprime_masks):
                new_score = best_score + score_vals[a]
                if new_score > best_score + 1e-15:
                    _update_best(best_chosen + [a], new_score)
                    improved = True
                    break
        if not improved:
            break
    
    # Phase 2: Try all single swaps (remove one, add one)
    while True:
        if time.time() > deadline:
            break
        improved = False
        for i, a_out in enumerate(best_chosen):
            if time.time() > deadline:
                break
            # Compute mask without a_out
            mask_without = best_mask & ~(1 << a_out)
            base_score = best_score - score_vals[a_out]
            for a_in in range(MAX_A, 0, -1):
                if a_in == a_out or a_in in best_set:
                    continue
                if _is_coprime_to_all_bitmask(a_in, mask_without, coprime_masks):
                    new_score = base_score + score_vals[a_in]
                    if new_score > best_score + 1e-15:
                        new_chosen = [x for x in best_chosen if x != a_out] + [a_in]
                        _update_best(new_chosen, new_score)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Phase 3: Try 2-for-1 swaps (remove two, add two)
    while True:
        if time.time() > deadline:
            break
        found = False
        n = len(best_chosen)
        for i in range(n):
            if time.time() > deadline:
                break
            a_out1 = best_chosen[i]
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                a_out2 = best_chosen[j]
                mask_without = best_mask & ~(1 << a_out1) & ~(1 << a_out2)
                base_score = best_score - score_vals[a_out1] - score_vals[a_out2]
                for a_in1 in range(MAX_A, 0, -1):
                    if time.time() > deadline:
                        break
                    if a_in1 in best_set or a_in1 == a_out1 or a_in1 == a_out2:
                        continue
                    if not _is_coprime_to_all_bitmask(a_in1, mask_without, coprime_masks):
                        continue
                    mask_with_in1 = mask_without | (1 << a_in1)
                    for a_in2 in range(MAX_A, 0, -1):
                        if a_in2 in best_set or a_in2 == a_out1 or a_in2 == a_out2 or a_in2 == a_in1:
                            continue
                        if not _is_coprime_to_all_bitmask(a_in2, mask_with_in1, coprime_masks):
                            continue
                        new_score = base_score + score_vals[a_in1] + score_vals[a_in2]
                        if new_score > best_score + 1e-15:
                            new_chosen = [x for x in best_chosen if x != a_out1 and x != a_out2] + [a_in1, a_in2]
                            _update_best(new_chosen, new_score)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    return best_chosen

def construct():
    start_time = time.time()
    # Use 140s as deadline to stay well within 150s limit
    deadline = start_time + 140.0
    
    # Build coprime bitmasks
    coprime_masks = _build_coprime_bitmasks()
    
    # Multiple greedy initializations with different orderings
    candidates = []
    
    # Order 1: descending (largest first)
    order1 = list(range(MAX_A, 0, -1))
    c1 = _greedy_construct(order1, deadline, coprime_masks)
    candidates.append(c1)
    
    # Order 2: ascending (smallest first)
    if time.time() < deadline - 10:
        order2 = list(range(1, MAX_A + 1))
        c2 = _greedy_construct(order2, deadline, coprime_masks)
        candidates.append(c2)
    
    # Order 3: random with seed
    if time.time() < deadline - 10:
        rng = random.Random(42)
        order3 = list(range(1, MAX_A + 1))
        rng.shuffle(order3)
        c3 = _greedy_construct(order3, deadline, coprime_masks)
        candidates.append(c3)
    
    # Order 4: another random
    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        order4 = list(range(1, MAX_A + 1))
        rng2.shuffle(order4)
        c4 = _greedy_construct(order4, deadline, coprime_masks)
        candidates.append(c4)
    
    # Order 5: prioritize high score values (large a) but with some randomness
    if time.time() < deadline - 10:
        rng3 = random.Random(999)
        # Weight by 1/(512-a) to favor large a
        weights = [1.0 / (N - a) for a in range(1, MAX_A + 1)]
        total_w = sum(weights)
        order5 = list(range(1, MAX_A + 1))
        # Sort by weight descending with slight randomization
        scored = [(weights[a-1] + rng3.random() * 0.01, a) for a in range(1, MAX_A + 1)]
        scored.sort(reverse=True)
        order5 = [a for _, a in scored]
        c5 = _greedy_construct(order5, deadline, coprime_masks)
        candidates.append(c5)
    
    # Pick the best initial candidate
    best_initial = max(candidates, key=_score_float)
    
    if time.time() > deadline:
        return best_initial
    
    # Local search from best initial
    result = _local_search(best_initial, deadline, coprime_masks)
    
    # If we have time, try local search from other good candidates too
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if _score_float(cand) >= _score_float(result) - 0.01:
                # Only search if close to best
                alt_result = _local_search(cand, deadline, coprime_masks)
                if _score_float(alt_result) > _score_float(result):
                    result = alt_result
    
    return result
# EVOLVE-BLOCK-END
