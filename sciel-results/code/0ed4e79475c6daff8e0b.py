"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 256
DEADLINE_FRACTION = 0.85  # safety margin for 60s wall-clock

# EVOLVE-BLOCK-START
def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all a+b with a<=b distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Try adding x
        new_sums = set(sums)
        valid = True
        for a in selected:
            s = x + a
            if s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if valid:
            s = 2 * x
            if s in new_sums:
                valid = False
            else:
                new_sums.add(s)
        if valid:
            return False
    return True

def _greedy_max_blocking(N, deadline):
    """
    Greedy construction that at each step picks the element that blocks
    the most remaining candidates, aiming for a small maximal Sidon set.
    """
    selected = []
    sums = set()
    remaining = set(range(1, N + 1))
    
    def _can_add(x, selected, sums):
        """Check if x can be added to selected while maintaining Sidon property."""
        added = []
        for a in selected:
            added.append(x + a)
        added.append(2 * x)
        if len(set(added)) != len(added):
            return False
        if sums.intersection(added):
            return False
        return True
    
    def _blocked_count(x, selected, sums):
        """Count how many remaining elements would be blocked by adding x."""
        added = [x + a for a in selected] + [2 * x]
        new_sums = sums.union(added)
        blocked = 0
        for y in remaining:
            if y == x:
                continue
            if y in selected:
                continue
            # Check if y can still be added after x is added
            y_added = [y + a for a in selected] + [y + x] + [2 * y]
            if len(set(y_added)) != len(y_added):
                blocked += 1
            elif new_sums.intersection(y_added):
                blocked += 1
        return blocked
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find the best candidate: one that can be added and blocks the most
        best_x = None
        best_blocked = -1
        
        for x in sorted(remaining):
            if not _can_add(x, selected, sums):
                continue
            blocked = _blocked_count(x, selected, sums)
            if blocked > best_blocked:
                best_blocked = blocked
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        remaining.discard(best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
    
    return selected

def _local_search(selected, N, deadline):
    """
    Try to reduce the size of the maximal Sidon set by removing elements
    and re-adding to restore maximality with fewer elements.
    """
    best = list(selected)
    best_size = len(best)
    
    def _try_remove_and_remaximalize(sel):
        """Try removing one element and see if we can get a smaller maximal set."""
        for i in range(len(sel)):
            candidate = sel[:i] + sel[i+1:]
            if not _is_valid_sidon(candidate):
                continue
            # Now try to make it maximal by adding elements
            # Greedily add elements to make it maximal
            sel_set = set(candidate)
            sums = set()
            for j, a in enumerate(candidate):
                for k in range(j, len(candidate)):
                    sums.add(a + candidate[k])
            
            new_sel = list(candidate)
            remaining = [x for x in range(1, N + 1) if x not in sel_set]
            
            # Greedily add to make maximal
            for x in remaining:
                if time.time() > deadline:
                    break
                added = [x + a for a in new_sel] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    new_sel.append(x)
                    sums.update(added)
            
            # Check if it's maximal
            if _is_maximal(new_sel, N):
                if len(new_sel) < best_size:
                    best_size = len(new_sel)
                    best = list(new_sel)
        
        return best, best_size
    
    # Try single removals
    for _ in range(5):  # limited iterations
        if time.time() > deadline:
            break
        best, best_size = _try_remove_and_remaximalize(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # First, get a baseline maximal Sidon set using greedy max-blocking
    selected = _greedy_max_blocking(N, deadline)
    
    if not selected:
        # Fallback to simple ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    # Try to improve via local search
    if time.time() < deadline:
        selected = _local_search(selected, N, deadline)
    
    return selected
# EVOLVE-BLOCK-END
