import time

_START = time.time()
_DEADLINE = 90.0
_MARGIN = 1.0  # safety margin in seconds
_CAP = 200
_MASK = (1 << (_CAP + 1)) - 1

def _bitset_reach(elements):
    """Compute reachable sums as a bitset (Python int) using each element at most once."""
    bits = 1  # sum 0 reachable
    for x in elements:
        bits |= (bits << x) & _MASK
    return bits

def _valid(s, cap=_CAP):
    bits = _bitset_reach(s)
    return not (bits >> cap) & 1

def _greedy_construct():
    """Greedy: add elements in descending order if 200 is not newly reachable."""
    selected = []
    bits = 1  # only sum 0 reachable
    for x in range(100, 0, -1):
        if (time.time() - _START) > (_DEADLINE - _MARGIN):
            break
        new_bits = bits | ((bits << x) & _MASK)
        if (new_bits >> _CAP) & 1:
            continue
        selected.append(x)
        bits = new_bits
    return selected

def _try_add_all(bits, selected, excluded):
    """Try to add any excluded element that doesn't create sum 200. Returns (bits, selected, changed)."""
    changed = False
    for x in sorted(excluded, reverse=True):
        new_bits = bits | ((bits << x) & _MASK)
        if (new_bits >> _CAP) & 1:
            continue
        selected.append(x)
        bits = new_bits
        changed = True
    return bits, selected, changed

def _local_search(best):
    """Try swaps (remove one, add another) that enable adding more elements."""
    best_set = set(best)
    excluded = [x for x in range(100, 0, -1) if x not in best_set]
    bits = _bitset_reach(best)
    
    # First, try to add all possible
    bits, best, _ = _try_add_all(bits, best, excluded)
    best_set = set(best)
    excluded = [x for x in range(100, 0, -1) if x not in best_set]
    
    # Try 1-for-1 swaps: remove r, add a, then try to add more
    improved = True
    while improved and (time.time() - _START) < (_DEADLINE - _MARGIN):
        improved = False
        best_set = set(best)
        excluded = [x for x in range(100, 0, -1) if x not in best_set]
        bits = _bitset_reach(best)
        
        for r in list(best):
            if (time.time() - _START) > (_DEADLINE - _MARGIN):
                break
            # Remove r from best
            best.remove(r)
            best_set.discard(r)
            bits_r = _bitset_reach(best)
            
            for a in excluded:
                if a == r:
                    continue
                # Try adding a
                new_bits = bits_r | ((bits_r << a) & _MASK)
                if (new_bits >> _CAP) & 1:
                    continue
                
                # Tentatively add a
                trial = best + [a]
                trial_set = best_set | {a}
                trial_bits = new_bits
                
                # Try to add more excluded elements
                trial_excluded = [x for x in range(100, 0, -1) if x not in trial_set]
                trial_bits, trial, _ = _try_add_all(trial_bits, trial, trial_excluded)
                
                if len(trial) > len(best) + 1:  # strict improvement over current
                    best = trial
                    best_set = set(best)
                    excluded = [x for x in range(100, 0, -1) if x not in best_set]
                    bits = _bitset_reach(best)
                    improved = True
                    break
            
            if improved:
                break
            # Restore r
            best.append(r)
            best_set.add(r)
    
    return best

def construct():
    # Greedy construction
    best = _greedy_construct()
    
    # Local search to try to improve
    if (time.time() - _START) < (_DEADLINE - _MARGIN):
        best = _local_search(best)
    
    # Final exact verification
    if not _valid(best):
        best = [1]
    
    if (time.time() - _START) > (_DEADLINE - _MARGIN):
        if not _valid(best):
            best = [1]
    
    return best
