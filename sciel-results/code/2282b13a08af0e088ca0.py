"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_tails(values, x):
    """Compute all block products ending at x given values before x."""
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    return tails

def _all_products(values):
    """Compute set of all nonempty consecutive block products."""
    prods = set()
    for i in range(len(values)):
        prod = 1
        for j in range(i, len(values)):
            prod *= values[j]
            prods.add(prod)
    return prods

def _is_valid(values):
    """Check if all nonempty consecutive block products are distinct."""
    prods = []
    for i in range(len(values)):
        prod = 1
        for j in range(i, len(values)):
            prod *= values[j]
            prods.append(prod)
    return len(prods) == len(set(prods))

def _greedy_from(start, end, existing=None):
    """Greedy: try to add numbers from start to end, maintaining validity."""
    if existing is None:
        values = []
    else:
        values = list(existing)
    products = _all_products(values) if values else set()
    
    for x in range(start, end + 1):
        if x in values:
            continue
        # Check if adding x at the end keeps products distinct
        # New products are all blocks ending at x
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _try_insert(values, pos, x):
    """Try inserting x at position pos in values. Returns new values or None."""
    if x in values:
        return None
    # Check products in the new sequence
    new_vals = values[:pos] + [x] + values[pos:]
    return _is_valid(new_vals) and new_vals or None

def _local_search_2move(values, candidates, max_iters=500):
    """2-move lookahead: remove one element, then try adding two others."""
    best = list(values)
    best_len = len(best)
    random.shuffle(candidates)
    
    for _ in range(max_iters):
        current = list(best)
        current_set = set(current)
        
        # Try removing each element
        for i in range(len(current)):
            removed = current[i]
            shortened = current[:i] + current[i+1:]
            
            # Try adding two candidates at various positions
            for x in candidates:
                if x in current_set:
                    continue
                # Try inserting x at various positions in shortened
                for pos in range(len(shortened) + 1):
                    trial = shortened[:pos] + [x] + shortened[pos:]
                    if not _is_valid(trial):
                        continue
                    trial_set = set(trial)
                    
                    # Now try adding a second element
                    for y in candidates:
                        if y in trial_set or y == x:
                            continue
                        for pos2 in range(len(trial) + 1):
                            trial2 = trial[:pos2] + [y] + trial[pos2:]
                            if _is_valid(trial2):
                                if len(trial2) > best_len:
                                    best = trial2
                                    best_len = len(best)
                                break
                        else:
                            continue
                        break
                # Don't break outer loop; keep searching
        # Update best if improved
        if len(best) > best_len:
            best_len = len(best)
    
    return best

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Phase 1: Greedy construction
    values = _greedy_from(2, 128)
    base_len = len(values)
    
    if time.time() > deadline:
        return values
    
    # Phase 2: Local search with 2-move lookahead
    candidates = [x for x in range(2, 129) if x not in set(values)]
    random.seed(42)
    
    improved = _local_search_2move(values, candidates, max_iters=200)
    if len(improved) > base_len:
        values = improved
        base_len = len(values)
    
    if time.time() > deadline:
        return values
    
    # Phase 3: Try multiple random restarts with different orderings
    best = list(values)
    best_len = len(best)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Randomize candidate order
        random.shuffle(candidates)
        
        # Greedy with random perturbation
        trial = _greedy_from(2, 128)
        
        # Local search on this trial
        trial_cands = [x for x in range(2, 129) if x not in set(trial)]
        trial_improved = _local_search_2move(trial, trial_cands, max_iters=100)
        
        if len(trial_improved) > best_len:
            best = trial_improved
            best_len = len(best)
    
    return best

# EVOLVE-BLOCK-END
