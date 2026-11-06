# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _check_sunflower(s, family, pair_intersections):
    """Check if adding s to family creates a sunflower with any pair in family."""
    for i in range(len(family)):
        a = family[i]
        ab = a & s
        for j in range(i + 1, len(family)):
            b = family[j]
            if a & b == ab and b & s == ab:
                return True
    return False

def _greedy_build(order, family_max=250):
    """Greedy build using given order of candidate sets."""
    chosen = []
    for s in order:
        if len(chosen) >= family_max:
            break
        if _check_sunflower(s, chosen, None):
            continue
        chosen.append(s)
    return chosen

def _local_improve(family, all_sets, time_limit):
    """Try to improve family by swapping out one set for one or more new sets."""
    chosen = list(family)
    chosen_set = set(id(s) for s in chosen)
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each set and see if we can add more
        for idx in range(len(chosen)):
            if time.time() - start > time_limit:
                break
            trial = chosen[:idx] + chosen[idx+1:]
            trial_ids = set(id(s) for s in trial)
            added = 0
            for s in all_sets:
                if id(s) in trial_ids:
                    continue
                if _check_sunflower(s, trial, None):
                    continue
                trial.append(s)
                trial_ids.add(id(s))
                added += 1
            if added > 1:
                chosen = trial
                chosen_set = set(id(s) for s in chosen)
                improved = True
                break
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    best_family = []
    
    # First pass: lexicographic greedy (baseline)
    fam = _greedy_build(all_sets)
    if len(fam) > len(best_family):
        best_family = fam
    
    if time.time() > deadline:
        return [sorted(s) for s in best_family]
    
    # Random restarts
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        order = list(all_sets)
        rng.shuffle(order)
        fam = _greedy_build(order)
        
        if len(fam) > len(best_family):
            best_family = fam
        
        # Local improvement on best family
        if time.time() < deadline - 3.0:
            improved = _local_improve(best_family, all_sets, 
                                      max(0.5, deadline - time.time() - 2.0))
            if len(improved) > len(best_family):
                best_family = improved
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
