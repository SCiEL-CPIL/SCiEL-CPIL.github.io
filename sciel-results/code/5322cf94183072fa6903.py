"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _generate_all_sets():
    """Generate all 4-element subsets of [1, 16] as bitmasks."""
    sets = []
    for combo in itertools.combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        sets.append(mask)
    return sets

def _mask_to_list(mask):
    """Convert bitmask to sorted list of 1-based indices."""
    return sorted([i + 1 for i in range(16) if (mask >> i) & 1])

def _can_add_with_index(family, idx, C):
    """
    Check if C can be added to family without forming a sunflower triple.
    Uses idx[A][mask] = list of B in family such that A & B == mask.
    """
    for A in family:
        v = A & C
        # We need to check if there exists B in family such that:
        # 1. A & B == v
        # 2. B & C == v
        # From idx[A][v], we get candidates for B satisfying condition 1.
        if v in idx[A]:
            for B in idx[A][v]:
                if (B & C) == v:
                    return False
    return True

def _add_to_family(family, idx, in_family, C):
    """Add C to family and update index."""
    family.append(C)
    in_family.add(C)
    idx[C] = {}
    for A in family[:-1]:
        v = A & C
        if v not in idx[A]:
            idx[A][v] = []
        idx[A][v].append(C)
        
        if v not in idx[C]:
            idx[C][v] = []
        idx[C][v].append(A)

def _remove_from_family(family, idx, in_family, C):
    """Remove C from family and update index."""
    # Remove C from family list
    try:
        idx_val = family.index(C)
        family.pop(idx_val)
    except ValueError:
        return
    
    in_family.remove(C)
    
    # Update indices
    for A in family:
        v = A & C
        if v in idx[A]:
            if C in idx[A][v]:
                idx[A][v].remove(C)
                if not idx[A][v]:
                    del idx[A][v]
        
        if v in idx[C]:
            if A in idx[C][v]:
                idx[C][v].remove(A)
                if not idx[C][v]:
                    del idx[C][v]
    
    # Clean up idx[C]
    if C in idx:
        del idx[C]

def _greedy_pass(all_sets, order, deadline):
    """Run a greedy pass with the given order."""
    family = []
    idx = {}
    in_family = set()
    
    for C in order:
        if time.time() > deadline:
            break
        if C in in_family:
            continue
        if _can_add_with_index(family, idx, C):
            _add_to_family(family, idx, in_family, C)
            
    return family, idx, in_family

def _local_search(family, idx, in_family, all_sets, deadline):
    """
    Perform local search to improve the family.
    Tries to add sets or swap sets.
    """
    # Copy state to avoid modifying original if we fail? 
    # Actually, we modify in place and keep track of best.
    # But since we need to return the best, let's work on copies or careful undo.
    # Given the complexity, let's just try to improve and return the best size found.
    
    # We need a list of candidates not in family
    candidates = [s for s in all_sets if s not in in_family]
    random.shuffle(candidates)
    
    best_size = len(family)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a candidate
        added = False
        for C in candidates:
            if time.time() > deadline:
                break
            if C in in_family:
                continue
            if _can_add_with_index(family, idx, C):
                _add_to_family(family, idx, in_family, C)
                candidates.remove(C)
                improved = True
                added = True
                if len(family) > best_size:
                    best_size = len(family)
                break
        
        if added:
            continue
            
        # If we couldn't add, try a swap
        # Pick a random set from family to remove
        if family:
            idx_to_remove = random.randint(0, len(family) - 1)
            A_old = family[idx_to_remove]
            
            # Pick a random candidate to add
            if candidates:
                idx_to_add = random.randint(0, len(candidates) - 1)
                C_new = candidates[idx_to_add]
                
                # Check if swap is valid
                # Temporarily remove A_old
                _remove_from_family(family, idx, in_family, A_old)
                
                if _can_add_with_index(family, idx, C_new):
                    # Commit swap
                    _add_to_family(family, idx, in_family, C_new)
                    candidates.remove(C_new)
                    # Add A_old back to candidates
                    candidates.append(A_old)
                    improved = True
                    if len(family) > best_size:
                        best_size = len(family)
                else:
                    # Revert swap
                    _add_to_family(family, idx, in_family, A_old)
    
    return family, idx, in_family, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    all_sets = _generate_all_sets()
    
    best_family = []
    best_idx = {}
    best_in_family = set()
    best_size = 0
    
    random.seed(42)
    
    # Multiple greedy restarts
    while time.time() < deadline - 5.0: # Leave 5s for final polish/local search
        order = all_sets[:]
        random.shuffle(order)
        
        family, idx, in_family = _greedy_pass(all_sets, order, deadline - 5.0)
        
        if len(family) > best_size:
            best_family = family
            best_idx = idx
            best_in_family = in_family
            best_size = len(family)
            
        # If we have a decent family, try local search
        if best_size > 0 and time.time() < deadline - 2.0:
            # Work on copies? No, _local_search modifies in place.
            # Let's pass copies to be safe, or just assume it improves.
            # Actually, if it doesn't improve, we revert? 
            # The current _local_search doesn't revert if it gets stuck.
            # It only commits if valid. So it's monotonic non-decreasing in size?
            # Yes, it only adds or swaps to valid states.
            
            # Let's run local search on the current best
            # We need to pass the state. 
            # To avoid messing up the best state if local search crashes or times out badly,
            # we can work on a deep copy? 
            # Deep copy of idx is expensive.
            # Let's just run it on the best state directly.
            
            l_family, l_idx, l_in_family, l_size = _local_search(
                best_family[:], # Shallow copy of list
                {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in best_idx.items()}, # Deep copy of idx
                set(best_in_family), # Copy of set
                all_sets, 
                deadline - 1.0
            )
            
            if l_size > best_size:
                best_family = l_family
                best_idx = l_idx
                best_in_family = l_in_family
                best_size = l_size

    # Convert to sorted lists
    result = [_mask_to_list(s) for s in best_family]
    return result

# EVOLVE-BLOCK-END
