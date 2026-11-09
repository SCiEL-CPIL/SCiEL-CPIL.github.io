"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _can_add(chosen, counts, x):
    """Check if x can be added without violating B2[2] constraint."""
    for y in chosen:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    if counts.get(2 * x, 0) >= 2:
        return False
    return True

def _do_add(chosen, counts, x):
    """Add x to chosen and update counts. Assumes _can_add returned True."""
    for y in chosen:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    counts[2 * x] = counts.get(2 * x, 0) + 1
    chosen.append(x)

def _do_remove(chosen, counts, x):
    """Remove x from chosen and update counts."""
    chosen.remove(x)
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

def _greedy_random(seed):
    """Randomized greedy construction."""
    random.seed(seed)
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    
    chosen = []
    counts = {}
    
    for x in candidates:
        if _can_add(chosen, counts, x):
            _do_add(chosen, counts, x)
    
    return chosen, counts

def _local_search(chosen, counts, deadline):
    """Local search: try adding excluded elements, then try swaps."""
    chosen_set = set(chosen)
    
    # Phase 1: Try to add excluded elements
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        excluded = [x for x in range(1, 1001) if x not in chosen_set]
        if not excluded:
            break
        
        x = random.choice(excluded)
        if _can_add(chosen, counts, x):
            _do_add(chosen, counts, x)
            chosen_set.add(x)
    
    # Phase 2: Try swaps (remove 1, add 1 or 2)
    for _ in range(1000):
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
        _do_remove(chosen, counts, x_remove)
        chosen_set.remove(x_remove)
        
        # Try to add up to 2 elements
        added = 0
        excluded = [x for x in range(1, 1001) if x not in chosen_set]
        if excluded:
            random.shuffle(excluded)
            for x_add in excluded[:5]:
                if added >= 2:
                    break
                if _can_add(chosen, counts, x_add):
                    _do_add(chosen, counts, x_add)
                    chosen_set.add(x_add)
                    added += 1
        
        # Accept if we gained elements
        if len(chosen) > len(chosen_copy):
            pass  # Keep the change
        else:
            # Revert
            for x in chosen[:]:
                if x not in chosen_set_copy:
                    _do_remove(chosen, counts, x)
                    chosen_set.remove(x)
            chosen[:] = chosen_copy
            counts.clear()
            counts.update(counts_copy)
            chosen_set.clear()
            chosen_set.update(chosen_set_copy)
    
    return chosen, counts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_chosen = []
    
    # Multiple random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        chosen, counts = _greedy_random(seed=restart * 7919)
        
        if len(chosen) > len(best_chosen):
            best_chosen = chosen[:]
        
        # Local search if we have time
        if time.time() < deadline - 5:
            chosen, counts = _local_search(chosen, counts, deadline)
            if len(chosen) > len(best_chosen):
                best_chosen = chosen[:]
    
    return best_chosen
# EVOLVE-BLOCK-END
