"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECS = [1.0 / d for d in DENOMS]
REC_FRACS = [Fraction(1, d) for d in DENOMS]

def _local_swap_search(selected, deadline):
    """Try to improve a selected set via single swaps (remove one, add one)."""
    selected_set = set(selected)
    current_sum = sum(RECS[i] for i in selected)
    best_sum = current_sum
    best_set = set(selected)
    
    # Precompute: for each i not in selected, and j in selected,
    # check if removing j and adding i keeps sum < 1 and improves deficit
    for _ in range(50):  # limit iterations for speed
        if time.time() > deadline:
            break
        improved = False
        # Try all removals
        for j in list(selected_set):
            if time.time() > deadline:
                break
            sum_without_j = current_sum - RECS[j]
            # Try all additions
            for i in range(N):
                if i in selected_set:
                    continue
                new_sum = sum_without_j + RECS[i]
                if new_sum < 1.0 and new_sum > best_sum:
                    # Verify with Fraction
                    exact = Fraction(0, 1)
                    for k in selected_set:
                        if k == j:
                            continue
                        exact += REC_FRACS[k]
                    exact += REC_FRACS[i]
                    if exact < 1:
                        best_sum = new_sum
                        best_set = (selected_set - {j}) | {i}
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
        # Update state
        selected_set = best_set
        current_sum = best_sum
    
    return list(best_set), best_sum

def _greedy_construct(deadline):
    """Greedy: start empty, repeatedly add the largest reciprocal that keeps sum < 1."""
    selected = set()
    current_sum = 0.0
    
    # Sort by reciprocal descending
    order = sorted(range(N), key=lambda i: -RECS[i])
    
    for i in order:
        if time.time() > deadline:
            break
        if i in selected:
            continue
        new_sum = current_sum + RECS[i]
        if new_sum < 1.0:
            # Verify with Fraction
            exact = Fraction(0, 1)
            for k in selected:
                exact += REC_FRACS[k]
            exact += REC_FRACS[i]
            if exact < 1:
                selected.add(i)
                current_sum = new_sum
    
    if not selected:
        selected = {0}
        current_sum = RECS[0]
    
    return list(selected), current_sum

def _randomized_greedy(deadline):
    """Randomized greedy: shuffle order, then greedily add."""
    selected = set()
    current_sum = 0.0
    
    order = list(range(N))
    random.shuffle(order)
    
    for i in order:
        if time.time() > deadline:
            break
        if i in selected:
            continue
        new_sum = current_sum + RECS[i]
        if new_sum < 1.0:
            exact = Fraction(0, 1)
            for k in selected:
                exact += REC_FRACS[k]
            exact += REC_FRACS[i]
            if exact < 1:
                selected.add(i)
                current_sum = new_sum
    
    if not selected:
        selected = {0}
        current_sum = RECS[0]
    
    return list(selected), current_sum

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_indices = [0]
    best_deficit = Fraction(1, 1) - Fraction(1, 2)  # deficit for [0] is 1/2
    best_sum_float = RECS[0]
    
    # Phase 1: Greedy construct
    inds, s = _greedy_construct(deadline)
    exact = Fraction(0, 1)
    for i in inds:
        exact += REC_FRACS[i]
    if exact < 1:
        deficit = Fraction(1, 1) - exact
        if deficit < best_deficit:
            best_deficit = deficit
            best_indices = inds
            best_sum_float = s
    
    # Phase 2: Local swap search from greedy result
    if time.time() < deadline:
        inds2, s2 = _local_swap_search(inds, deadline)
        exact2 = Fraction(0, 1)
        for i in inds2:
            exact2 += REC_FRACS[i]
        if exact2 < 1:
            deficit2 = Fraction(1, 1) - exact2
            if deficit2 < best_deficit:
                best_deficit = deficit2
                best_indices = inds2
                best_sum_float = s2
    
    # Phase 3: Multiple randomized greedy restarts + local swap
    while time.time() < deadline:
        inds3, s3 = _randomized_greedy(deadline)
        exact3 = Fraction(0, 1)
        for i in inds3:
            exact3 += REC_FRACS[i]
        if exact3 < 1:
            deficit3 = Fraction(1, 1) - exact3
            if deficit3 < best_deficit:
                best_deficit = deficit3
                best_indices = inds3
                best_sum_float = s3
        
        # Local swap from this random start
        if time.time() < deadline:
            inds4, s4 = _local_swap_search(inds3, deadline)
            exact4 = Fraction(0, 1)
            for i in inds4:
                exact4 += REC_FRACS[i]
            if exact4 < 1:
                deficit4 = Fraction(1, 1) - exact4
                if deficit4 < best_deficit:
                    best_deficit = deficit4
                    best_indices = inds4
                    best_sum_float = s4
    
    return best_indices
# EVOLVE-BLOCK-END
