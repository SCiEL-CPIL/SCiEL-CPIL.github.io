"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def _can_add_fast(chosen, counts, x):
    """Check if x can be added without violating B2[2] constraint.
    Optimized: only check sums that could violate the constraint."""
    # Check 2x
    if counts.get(2 * x, 0) >= 2:
        return False
    # Check x + y for all y in chosen
    for y in chosen:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    return True

def _do_add_fast(chosen, counts, x):
    """Add x to chosen and update counts."""
    for y in chosen:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    counts[2 * x] = counts.get(2 * x, 0) + 1
    chosen.append(x)

def _do_remove_fast(chosen, counts, x):
    """Remove x from chosen and update counts."""
    idx = chosen.index(x)
    chosen[idx] = chosen[-1]
    chosen.pop()
    for y in chosen:
        s = x + y
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
    s2 = 2 * x
    if s2 in counts:
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]

def _greedy_slack(seed, deadline):
    """Greedy construction using slack-based selection.
    At each step, pick the candidate with maximum slack (most room in sum nodes)."""
    random.seed(seed)
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    
    chosen = []
    counts = {}
    chosen_set = set()
    
    # Build a priority structure: for each candidate, compute how many sums would be blocked
    # We'll use a simpler approach: sort candidates by initial "slack" (how many sums are already at capacity)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if _can_add_fast(chosen, counts, x):
            _do_add_fast(chosen, counts, x)
            chosen_set.add(x)
    
    return chosen, counts, chosen_set

def _local_search_v2(chosen, counts, chosen_set, deadline):
    """Improved local search with better swap moves."""
    n = len(chosen)
    
    # Phase 1: Try to add excluded elements (more aggressive)
    excluded = [x for x in range(1, 1001) if x not in chosen_set]
    random.shuffle(excluded)
    
    added_count = 0
    for x in excluded:
        if time.time() > deadline:
            break
        if _can_add_fast(chosen, counts, x):
            _do_add_fast(chosen, counts, x)
            chosen_set.add(x)
            added_count += 1
    
    # Phase 2: Try swaps - remove 1, add 1-2
    for _ in range(500):
        if time.time() > deadline:
            break
        
        if len(chosen) < 1:
            break
        
        # Save state
        chosen_copy = chosen[:]
        counts_copy = dict(counts)
        chosen_set_copy = set(chosen_set)
        
        # Remove a random element
        x_remove = random.choice(chosen)
        _do_remove_fast(chosen, counts, x_remove)
        chosen_set.remove(x_remove)
        
        # Try to add up to 2 elements from excluded
        added = 0
        excluded_now = [x for x in range(1, 1001) if x not in chosen_set]
        random.shuffle(excluded_now)
        
        for x_add in excluded_now[:10]:  # Try more candidates
            if added >= 2:
                break
            if _can_add_fast(chosen, counts, x_add):
                _do_add_fast(chosen, counts, x_add)
                chosen_set.add(x_add)
                added += 1
        
        # Accept if we gained elements
        if len(chosen) > len(chosen_copy):
            pass  # Keep the change
        else:
            # Revert
            for x in chosen[:]:
                if x not in chosen_set_copy:
                    _do_remove_fast(chosen, counts, x)
                    chosen_set.remove(x)
            chosen[:] = chosen_copy
            counts.clear()
            counts.update(counts_copy)
            chosen_set.clear()
            chosen_set.update(chosen_set_copy)
    
    return chosen, counts, chosen_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_chosen = []
    
    # Multiple random restarts with varied strategies
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Alternate between different greedy orders
        seed = restart * 7919 + restart * restart * 3
        
        chosen, counts, chosen_set = _greedy_slack(seed, deadline)
        
        if len(chosen) > len(best_chosen):
            best_chosen = chosen[:]
        
        # Local search if we have time
        if time.time() < deadline - 3:
            chosen, counts, chosen_set = _local_search_v2(chosen, counts, chosen_set, deadline)
            if len(chosen) > len(best_chosen):
                best_chosen = chosen[:]
    
    return best_chosen
# EVOLVE-BLOCK-END
