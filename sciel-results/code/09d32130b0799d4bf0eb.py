"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _can_add(x, selected, sums):
    """Check if x can be added to selected maintaining Sidon property."""
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _new_sums(x, selected, sums):
    """Compute new sums if x is added."""
    new = set(sums)
    for a in selected:
        new.add(x + a)
    new.add(2 * x)
    return new

def _blocking_score(x, selected, sums, candidates):
    """Count how many other candidates would be blocked if x is added."""
    new_sums = set(sums)
    for a in selected:
        new_sums.add(x + a)
    new_sums.add(2 * x)
    blocked = 0
    for y in candidates:
        if y == x:
            continue
        # Check if y can still be added with new_sums and selected + [x]
        ok = True
        for a in selected:
            if y + a in new_sums:
                ok = False
                break
        if ok and 2 * y in new_sums:
            ok = False
        if not ok:
            blocked += 1
    return blocked

def _greedy_maximal_sidon(seed):
    """Randomized greedy with blocking heuristic."""
    rng = random.Random(seed)
    elements = list(range(1, N + 1))
    rng.shuffle(elements)
    
    selected = []
    sums = set()
    candidates = set(elements)
    
    while candidates:
        # Find all addable candidates
        addable = []
        for x in candidates:
            if _can_add(x, selected, sums):
                addable.append(x)
        
        if not addable:
            break
        
        # Compute blocking score for each addable element
        # Use a subset of candidates for speed
        cand_list = list(candidates)
        best_x = None
        best_score = -1
        best_tie = -1
        
        for x in addable:
            score = _blocking_score(x, selected, sums, cand_list)
            # Add small random tiebreaker for diversity
            tiebreak = rng.random() * 0.01
            total = score + tiebreak
            if total > best_score:
                best_score = total
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        selected.append(best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        candidates.discard(best_x)
        
        # Remove candidates that can no longer be added
        to_remove = []
        for x in candidates:
            if not _can_add(x, selected, sums):
                to_remove.append(x)
        for x in to_remove:
            candidates.discard(x)
    
    return selected

def _verify(sidon_set):
    """Verify Sidon property and maximality."""
    sums = set()
    for i, a in enumerate(sidon_set):
        for b in sidon_set[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    # Check maximality
    for x in range(1, N + 1):
        if x in sidon_set:
            continue
        # Try adding x
        new_sums = set(sums)
        for a in sidon_set:
            s = x + a
            if s in new_sums:
                break
            new_sums.add(s)
        else:
            if 2 * x in new_sums:
                break
            else:
                return False  # Not maximal
    return True

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # First try the original greedy as a baseline
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    if len(selected) < best_size:
        best_size = len(selected)
        best = selected
    
    # Now try randomized greedy with blocking heuristic
    seed = 0
    while time.time() < deadline:
        if time.time() > deadline - 2:
            break
        result = _greedy_maximal_sidon(seed)
        if result and len(result) < best_size:
            best_size = len(result)
            best = result
        seed += 1
    
    if best is None:
        # Fallback: return the original greedy result
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    return best
# EVOLVE-BLOCK-END
