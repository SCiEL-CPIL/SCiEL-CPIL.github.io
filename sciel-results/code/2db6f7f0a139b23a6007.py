"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_3ap_with(selected, x):
    """Check if adding x to selected creates a 3-AP. O(|selected|)."""
    for a in selected:
        # x is middle: 2x = a + b => b = 2x - a
        b = 2*x - a
        if b != a and b in selected:
            return True
        # a is middle: 2a = b + x => b = 2a - x
        b = 2*a - x
        if b != a and b in selected:
            return True
    return False

def _reciprocal_sum(selected):
    """Compute sum of reciprocals as float."""
    return sum(1.0/x for x in selected)

def _greedy_by_reciprocal():
    """Greedy: add elements in order of largest reciprocal (smallest x)."""
    selected = set()
    for x in range(1, 257):
        if not _has_3ap_with(selected, x):
            selected.add(x)
    return selected

def _greedy_randomized(seed_val):
    """Greedy with randomization for diversity."""
    import random
    rng = random.Random(seed_val)
    selected = set()
    candidates = list(range(1, 257))
    # Bias towards smaller numbers but with some randomness
    weights = [1.0/x for x in candidates]
    total_w = sum(weights)
    while candidates and time.time() < _deadline:
        # Weighted random pick
        r = rng.random() * total_w
        cum = 0
        idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                idx = i
                break
        x = candidates[idx]
        if not _has_3ap_with(selected, x):
            selected.add(x)
        candidates.pop(idx)
        weights.pop(idx)
        total_w -= weights[idx] if idx < len(weights) else 0
    return selected

def _local_search(initial, deadline):
    """Local search with 1-add, 1-swap, and 2-swap moves."""
    selected = set(initial)
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Phase 1: Add all possible elements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if x not in selected and not _has_3ap_with(selected, x):
                selected.add(x)
                changed = True
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Phase 2: 1-swap moves (remove y, add x)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 257):
            if time.time() > deadline:
                return list(best_set)
            if x in selected:
                continue
            # Find conflicts: elements in selected that would block x
            conflicts = set()
            for a in selected:
                # x middle: b = 2x - a
                b = 2*x - a
                if b != a and b in selected:
                    conflicts.add(a)
                    conflicts.add(b)
                # a middle: b = 2a - x
                b = 2*a - x
                if b != a and b in selected:
                    conflicts.add(a)
                    conflicts.add(b)
            # Try removing one conflict to allow x
            for y in list(conflicts):
                if time.time() > deadline:
                    return list(best_set)
                new_set = selected - {y}
                if not _has_3ap_with(new_set, x):
                    new_set.add(x)
                    new_sum = _reciprocal_sum(new_set)
                    if new_sum > best_sum + 1e-15:
                        best_sum = new_sum
                        best_set = new_set
                        selected = set(new_set)
                        improved = True
    
    # Phase 3: 2-swap moves (remove y1, y2, add x1, x2)
    # Only try promising pairs: small x's not in set, and their conflicts
    missing = [x for x in range(1, 257) if x not in selected]
    # Sort missing by reciprocal (descending) - prefer small numbers
    missing.sort(key=lambda x: 1.0/x, reverse=True)
    
    for x1 in missing[:40]:  # Limit to top 40 missing
        if time.time() > deadline:
            return list(best_set)
        if x1 in selected:
            continue
        # Find conflicts for x1
        conflicts1 = set()
        for a in selected:
            b = 2*x1 - a
            if b != a and b in selected:
                conflicts1.add(a)
                conflicts1.add(b)
            b = 2*a - x1
            if b != a and b in selected:
                conflicts1.add(a)
                conflicts1.add(b)
        
        for x2 in missing[:40]:
            if time.time() > deadline:
                return list(best_set)
            if x2 == x1 or x2 in selected:
                continue
            # Check x1 and x2 don't conflict with each other
            b_check = 2*x1 - x2
            if b_check in selected:
                conflicts1.add(x2)  # mark as conflict
                continue
            b_check = 2*x2 - x1
            if b_check in selected:
                continue
            
            # Find combined conflicts
            conflicts2 = set()
            for a in selected:
                b = 2*x2 - a
                if b != a and b in selected:
                    conflicts2.add(a)
                    conflicts2.add(b)
                b = 2*a - x2
                if b != a and b in selected:
                    conflicts2.add(a)
                    conflicts2.add(b)
            
            all_conflicts = conflicts1 | conflicts2
            # Try removing pairs from all_conflicts
            conflict_list = list(all_conflicts)
            for i in range(len(conflict_list)):
                if time.time() > deadline:
                    return list(best_set)
                for j in range(i+1, len(conflict_list)):
                    if time.time() > deadline:
                        return list(best_set)
                    y1, y2 = conflict_list[i], conflict_list[j]
                    new_set = selected - {y1, y2}
                    if not _has_3ap_with(new_set, x1) and not _has_3ap_with(new_set, x2):
                        new_set.add(x1)
                        new_set.add(x2)
                        new_sum = _reciprocal_sum(new_set)
                        if new_sum > best_sum + 1e-15:
                            best_sum = new_sum
                            best_set = new_set
                            selected = set(new_set)
                            # After improvement, restart phase 2
                            improved = True
    
    return list(best_set)

def construct():
    global _deadline
    _deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Try multiple initializations
    best_result = None
    best_score = 0.0
    
    # Greedy by reciprocal (ascending)
    init1 = _greedy_by_reciprocal()
    result1 = _local_search(init1, _deadline)
    score1 = _reciprocal_sum(result1)
    if score1 > best_score:
        best_score = score1
        best_result = result1
    
    # A few randomized greedy starts
    for seed_val in [42, 123, 7]:
        if time.time() > _deadline:
            break
        init_r = _greedy_randomized(seed_val)
        result_r = _local_search(init_r, _deadline)
        score_r = _reciprocal_sum(result_r)
        if score_r > best_score:
            best_score = score_r
            best_result = result_r
    
    return best_result if best_result else init1
# EVOLVE-BLOCK-END
