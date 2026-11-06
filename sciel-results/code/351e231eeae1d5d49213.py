"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

N = 512
MAX_A = N - 1  # 511

# Precompute weights: weight[a] = 1/(512-a) for a in 1..511
WEIGHTS = [0.0] + [1.0 / (N - a) for a in range(1, MAX_A + 1)]

# Precompute conflict bitmask for each number
# conflict_mask[a] has bit b set if gcd(a, b) > 1
def _build_conflict_masks():
    masks = [0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        mask = 0
        for b in range(a + 1, MAX_A + 1):
            if gcd(a, b) > 1:
                mask |= (1 << b)
        masks[a] = mask
    return masks

CONFLICT_MASKS = _build_conflict_masks()

def _score(chosen):
    """Compute the score sum 1/(512-a) for chosen numbers."""
    total = 0.0
    for a in chosen:
        total += WEIGHTS[a]
    return total

def _greedy_independent_set(candidates, time_limit, start_time):
    """Greedy: pick highest weight first, then next compatible, etc."""
    chosen = []
    chosen_mask = 0  # bitmask of chosen numbers
    for a in candidates:
        if time.time() - start_time > time_limit:
            break
        # Check if a is coprime with all chosen
        if (CONFLICT_MASKS[a] & chosen_mask) == 0:
            chosen.append(a)
            chosen_mask |= (1 << a)
    return chosen

def _branch_and_bound(candidates, best_score, best_set, time_limit, start_time, depth=0):
    """
    Branch and bound for maximum weight independent set.
    candidates: list of candidate numbers to consider
    best_score: current best score found
    best_set: current best set
    """
    if time.time() - start_time > time_limit:
        return best_score, best_set
    
    if not candidates:
        return best_score, best_set
    
    # Upper bound: sum of weights of all candidates (loose but fast)
    upper_bound = sum(WEIGHTS[a] for a in candidates)
    
    # If even taking all candidates can't beat best, prune
    if upper_bound <= best_score:
        return best_score, best_set
    
    # Sort candidates by weight descending for better pruning
    candidates_sorted = sorted(candidates, key=lambda x: WEIGHTS[x], reverse=True)
    
    # Try including the first candidate
    first = candidates_sorted[0]
    remaining = candidates_sorted[1:]
    
    # Filter remaining to those compatible with first
    compatible_with_first = [a for a in remaining if (CONFLICT_MASKS[first] >> a) & 1 == 0]
    
    # Recurse with first included
    new_score = best_score + WEIGHTS[first]
    new_set = best_set + [first] if best_set else [first]
    
    # Check if this alone beats current best
    if new_score > best_score:
        best_score = new_score
        best_set = new_set
    
    # Recurse on compatible remaining
    if compatible_with_first:
        best_score, best_set = _branch_and_bound(
            compatible_with_first, best_score, best_set, time_limit, start_time, depth + 1
        )
    
    # Try excluding the first candidate
    if remaining:
        best_score, best_set = _branch_and_bound(
            remaining, best_score, best_set, time_limit, start_time, depth + 1
        )
    
    return best_score, best_set

def _local_search(chosen_list, time_limit, start_time):
    """Local search: try swapping elements for better compatible ones."""
    chosen_set = set(chosen_list)
    best_score = _score(chosen_list)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        for i, a in enumerate(chosen_list):
            # Try replacing a with a larger weight number
            for new_a in range(MAX_A, 0, -1):
                if time.time() - start_time > time_limit:
                    break
                if new_a in chosen_set:
                    continue
                if WEIGHTS[new_a] <= WEIGHTS[a]:
                    continue
                # Check if new_a is coprime with all others (excluding a)
                others = [b for j, b in enumerate(chosen_list) if j != i]
                if all(gcd(new_a, b) == 1 for b in others):
                    # Do the swap
                    chosen_list[i] = new_a
                    chosen_set.remove(a)
                    chosen_set.add(new_a)
                    improved = True
                    break
    
    # Try adding more numbers
    for a in range(MAX_A, 0, -1):
        if time.time() - start_time > time_limit:
            break
        if a not in chosen_set and all(gcd(a, b) == 1 for b in chosen_list):
            chosen_list.append(a)
            chosen_set.add(a)
    
    return chosen_list

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s
    
    # Phase 1: Greedy from 511 down (highest weight first)
    candidates = list(range(MAX_A, 0, -1))
    chosen = _greedy_independent_set(candidates, time_limit, start_time)
    
    # Phase 2: Local search to improve
    if time.time() - start_time < time_limit:
        chosen = _local_search(chosen, time_limit, start_time)
    
    # Phase 3: Branch and bound on a reduced candidate set
    # Focus on high-weight numbers that could potentially improve
    current_score = _score(chosen)
    chosen_set = set(chosen)
    
    # Candidates: numbers not in chosen that have higher weight than some chosen number
    # and are coprime with most of the chosen set
    bnb_candidates = []
    for a in range(MAX_A, 0, -1):
        if time.time() - start_time > time_limit:
            break
        if a in chosen_set:
            continue
        # Only consider if weight is high enough
        if WEIGHTS[a] < 0.005:  # 1/(512-a) < 0.005 means a > 312
            continue
        # Check compatibility with chosen set
        compatible_count = sum(1 for b in chosen if gcd(a, b) == 1)
        if compatible_count >= len(chosen) - 2:  # Compatible with most
            bnb_candidates.append(a)
    
    if bnb_candidates and time.time() - start_time < time_limit - 5:
        # Use branch and bound on the reduced set
        # We'll try to find a better subset from bnb_candidates + some chosen
        # For simplicity, we do B&B on bnb_candidates alone and merge if better
        bnb_start = time.time()
        bnb_limit = min(time_limit - (time.time() - start_time), 30.0)
        
        if bnb_limit > 5:
            bnb_score, bnb_set = _branch_and_bound(
                bnb_candidates, 0.0, [], bnb_limit, bnb_start
            )
            
            # If B&B found a good set, try combining with existing
            if bnb_set and bnb_score > 0:
                # Check if we can merge
                merged = list(chosen)
                for a in bnb_set:
                    if a not in set(merged) and all(gcd(a, b) == 1 for b in merged):
                        merged.append(a)
                if _score(merged) > current_score:
                    chosen = merged
                    current_score = _score(chosen)
    
    # Final local search pass
    if time.time() - start_time < time_limit - 2:
        chosen = _local_search(chosen, time_limit, start_time)
    
    return chosen
# EVOLVE-BLOCK-END
