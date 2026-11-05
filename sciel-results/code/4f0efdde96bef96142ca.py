"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random

# EVOLVE-BLOCK-START
def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    # Precompute all sums for fast checking
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would break Sidon property
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        # Also need to check that new sums don't collide with each other
        # But since we're just checking if x can be added, and x is new,
        # the only new sums are x+a for a in selected, and 2x.
        # We need to ensure these new sums don't collide with existing sums
        # and don't collide with each other.
        if can_add:
            new_sums = set()
            for a in selected:
                s = x + a
                if s in new_sums or s in sums:
                    can_add = False
                    break
                new_sums.add(s)
            if can_add:
                if 2 * x in new_sums or 2 * x in sums:
                    can_add = False
        if can_add:
            return False
    return True

def _greedy_construct(order):
    """Greedy construction given a specific order of elements to consider."""
    selected = []
    sums = set()
    for x in order:
        added = set()
        conflict = False
        for a in selected:
            s = x + a
            if s in sums or s in added:
                conflict = True
                break
            added.add(s)
        if not conflict:
            s = 2 * x
            if s in sums or s in added:
                conflict = True
            else:
                added.add(s)
        if not conflict:
            selected.append(x)
            sums.update(added)
    return selected

def construct():
    deadline = time.time() + 55.0  # 55s deadline to be safe (60s limit)
    best_set = None
    best_size = float('inf')
    
    # First, do a quick ascending greedy to get a baseline
    baseline = _greedy_construct(range(1, N + 1))
    best_set = baseline
    best_size = len(baseline)
    
    # Then try randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        result = _greedy_construct(order)
        
        if len(result) < best_size:
            best_set = result
            best_size = len(result)
        
        # If we've found a very small set, we can stop early
        if best_size <= 16:
            break
    
    # Ensure maximality of the best set found
    if best_set is not None and not _is_maximal(best_set):
        # Try to complete it to maximal
        # This is a simplified completion - just try adding elements greedily
        sel_set = set(best_set)
        sums = set()
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                sums.add(best_set[i] + best_set[j])
        
        # Try to add missing elements in ascending order to make it maximal
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = set()
            conflict = False
            for a in best_set:
                s = x + a
                if s in sums or s in added:
                    conflict = True
                    break
                added.add(s)
            if not conflict:
                s = 2 * x
                if s in sums or s in added:
                    conflict = True
                else:
                    added.add(s)
            if not conflict:
                best_set.append(x)
                sel_set.add(x)
                sums.update(added)
    
    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
