"""Improved program: Local search with greedy restoration for minimum maximal no-isosceles set."""
import random
import time
import itertools

N = 16
DEADLINE = time.time() + 290  # Safety margin before 300s kill


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(point, chosen):
    """Check if point can be added to chosen without creating isosceles triangle."""
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(point, a, b):
            return False
    return True


def _is_maximal(chosen):
    """Check if no grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _is_valid(chosen):
    """Check if chosen is isosceles-free."""
    for a, b, c in itertools.combinations(chosen, 3):
        if _is_isosceles(a, b, c):
            return False
    return True


def _get_addable_points(chosen):
    """Return list of points not in chosen that can be added to chosen."""
    chosen_set = set(chosen)
    addable = []
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            addable.append(p)
    return addable


def _greedy_random(seed_val):
    """Randomized greedy construction."""
    rng = random.Random(seed_val)
    points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(points)
    
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    
    return chosen


def _local_shrink(chosen, rng):
    """
    Try to shrink the maximal set 'chosen' by removing a point and restoring maximality
    with the minimum number of additions (greedy).
    Returns (new_set, improved)
    """
    if not chosen:
        return chosen, False
    
    # Try removing each point
    # We pick a random point to remove to diversify
    idx = rng.randint(0, len(chosen) - 1)
    p_remove = chosen[idx]
    new_set = [p for i, p in enumerate(chosen) if i != idx]
    
    # Check if new_set is already maximal (unlikely if chosen was minimal, but possible if chosen wasn't)
    if _is_maximal(new_set):
        return new_set, True
    
    # new_set is not maximal. Find addable points.
    addable = _get_addable_points(new_set)
    
    if not addable:
        # Should not happen if _is_maximal returned False
        return new_set, True 

    # Greedy restoration:
    # We need to add points to 'new_set' such that no more points are addable.
    # Each added point 'q' will block some of the currently addable points.
    # We want to cover the 'addable' set with the fewest 'q's.
    
    current_set = list(new_set)
    current_addable = list(addable)
    
    added_count = 0
    max_adds = len(chosen) // 2  # Safety limit: don't add too many
    
    while current_addable and added_count < max_adds:
        # For each candidate q in current_addable, calculate how many points in current_addable it blocks.
        # Note: Adding q removes q from current_addable.
        # Also, adding q might block other points r in current_addable if (q, r, s) forms isosceles?
        # No, "blocking" means r cannot be added anymore.
        # r cannot be added if there exists s in current_set such that (r, s, t) is isosceles?
        # Actually, r is addable to current_set if it doesn't form isosceles with any pair in current_set.
        # If we add q to current_set, r is NOT addable to new_current_set if:
        # 1. r == q (handled by removal)
        # 2. r forms isosceles with q and some s in current_set.
        
        best_q = None
        best_blocked_count = -1
        
        # Optimization: Only check a subset if too slow?
        # For now, check all.
        for q in current_addable:
            blocked = 0
            # Check which other points in current_addable become non-addable
            for r in current_addable:
                if r == q:
                    continue
                # Check if r can be added to (current_set + [q])
                # This is equivalent to checking if r forms isosceles with q and any s in current_set
                # OR if r forms isosceles with two points in current_set (already false, since r was addable)
                # So we only need to check interactions with q.
                # If r forms isosceles with q and some s in current_set, then r is blocked.
                
                # Quick check:
                # Is there an s in current_set such that _is_isosceles(r, q, s)?
                blocked_here = False
                for s in current_set:
                    if _is_isosceles(r, q, s):
                        blocked_here = True
                        break
                if blocked_here:
                    blocked += 1
            
            # Also, q itself is removed from addable list, so it "blocks" itself (1)
            total_blocked = blocked + 1
            
            if total_blocked > best_blocked_count:
                best_blocked_count = total_blocked
                best_q = q
        
        if best_q is None:
            break
            
        # Add best_q to current_set
        current_set.append(best_q)
        added_count += 1
        
        # Update current_addable:
        # Remove best_q
        # Remove any r that is now blocked by best_q
        new_addable = []
        for r in current_addable:
            if r == best_q:
                continue
            # Check if r is still addable to current_set
            # We know it was addable to current_set without best_q.
            # It is NOT addable now if it forms isosceles with best_q and some s in old_current_set.
            blocked_by_q = False
            for s in [p for p in current_set if p != best_q]:
                if _is_isosceles(r, best_q, s):
                    blocked_by_q = True
                    break
            if not blocked_by_q:
                new_addable.append(r)
        
        current_addable = new_addable
        
        if not current_addable:
            break
        
    if not current_addable:
        # We restored maximality
        # Size change: -1 (removed) + added_count
        # If added_count == 0, size decreased.
        # If added_count == 1, size same.
        # If added_count > 1, size increased.
        if len(current_set) < len(chosen):
            return current_set, True
        else:
            return chosen, False # Don't accept if size didn't decrease
    else:
        # Failed to restore maximality within budget
        return chosen, False


def construct():
    best_set = None
    best_size = float('inf')
    
    # First, get a baseline from deterministic greedy (row-major)
    baseline = []
    for p in itertools.product(range(N), repeat=2):
        if _can_add(p, baseline):
            baseline.append(p)
    if _is_valid(baseline) and _is_maximal(baseline):
        best_set = baseline
        best_size = len(baseline)
    
    rng = random.Random(42)
    
    start_time = time.time()
    attempt = 0
    while time.time() < DEADLINE:
        attempt += 1
        
        # Generate a candidate
        if attempt % 10 == 0 and best_set is not None:
            # Occasionally try to improve the best found set
            candidate = list(best_set)
        else:
            candidate = _greedy_random(attempt)
            
        if not _is_valid(candidate):
            continue
            
        # Ensure it is maximal (greedy construction usually is, but just in case)
        # If not maximal, add points until maximal
        while True:
            addable = _get_addable_points(candidate)
            if not addable:
                break
            # Add a random addable point
            candidate.append(rng.choice(addable))
            
        size = len(candidate)
        if size >= best_size:
            # Try to shrink it
            # Only try to shrink if we have time
            if time.time() < DEADLINE - 5:
                new_candidate, improved = _local_shrink(candidate, rng)
                if improved:
                    size = len(new_candidate)
                    candidate = new_candidate
                    if size < best_size:
                        best_set = candidate
                        best_size = size
                        print(f"Shrunk to size={size} at attempt={attempt}")
            continue
        
        # If size < best_size, check if it's valid and maximal (it is)
        best_set = candidate
        best_size = size
        print(f"Found better: size={size} at attempt={attempt}")
        
        # Try to shrink the new best
        if time.time() < DEADLINE - 5:
            new_candidate, improved = _local_shrink(candidate, rng)
            if improved:
                best_set = new_candidate
                best_size = len(best_set)
                print(f"Shrunk best to size={best_size}")
                
        if best_size <= 10:
            break
            
    return best_set if best_set is not None else baseline
