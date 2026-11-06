"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _is_4ap_free(selected):
    """Check if a set contains no nonconstant 4-term arithmetic progression."""
    s = set(selected)
    n = max(s) if s else 0
    for a in s:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in s and (a + 2*d) in s and (a + 3*d) in s:
                return False
    return True


def _greedy_forward(selected=None):
    """Greedy construction: iterate 1..512, add x if it doesn't complete a 4-AP."""
    if selected is None:
        selected = set()
    for x in range(1, 513):
        if x in selected:
            continue
        valid = True
        # x as last term: need x-d, x-2d, x-3d in selected
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as first term: need x+d, x+2d, x+3d in selected
        for d in range(1, (512 - x) // 3 + 1):
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as second term: need x-d, x+d, x+2d in selected
        for d in range(1, x):
            if (x + 2*d) <= 512 and (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as third term: need x-2d, x-d, x+d in selected
        for d in range(1, (x - 1) // 2 + 1):
            if (x + d) <= 512 and (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                valid = False
                break
        if valid:
            selected.add(x)
    return selected


def _greedy_reverse():
    """Greedy construction: iterate 512..1, add x if it doesn't complete a 4-AP."""
    selected = set()
    for x in range(512, 0, -1):
        valid = True
        # x as last term
        for d in range(1, x // 3 + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as first term
        for d in range(1, (512 - x) // 3 + 1):
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as second term
        for d in range(1, x):
            if (x + 2*d) <= 512 and (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as third term
        for d in range(1, (x - 1) // 2 + 1):
            if (x + d) <= 512 and (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                valid = False
                break
        if valid:
            selected.add(x)
    return selected


def _greedy_random_order(seed):
    """Greedy construction with a specific random ordering."""
    rng = random.Random(seed)
    order = list(range(1, 513))
    rng.shuffle(order)
    selected = set()
    for x in order:
        valid = True
        # x as last term
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as first term
        for d in range(1, (512 - x) // 3 + 1):
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as second term
        for d in range(1, x):
            if (x + 2*d) <= 512 and (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                valid = False
                break
        if not valid:
            continue
        # x as third term
        for d in range(1, (x - 1) // 2 + 1):
            if (x + d) <= 512 and (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                valid = False
                break
        if valid:
            selected.add(x)
    return selected


def _local_improve(selected, deadline):
    """Try to add more elements by removing blockers. Returns improved set."""
    selected = set(selected)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Find elements that can be added
        for x in range(1, 513):
            if x in selected:
                continue
            # Check if x can be added
            blockers = set()
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    blockers.update([x - d, x - 2*d, x - 3*d])
            for d in range(1, (512 - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    blockers.update([x + d, x + 2*d, x + 3*d])
            for d in range(1, x):
                if (x + 2*d) <= 512 and (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                    blockers.update([x - d, x + d, x + 2*d])
            for d in range(1, (x - 1) // 2 + 1):
                if (x + d) <= 512 and (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                    blockers.update([x - 2*d, x - d, x + d])
            
            if not blockers:
                selected.add(x)
                improved = True
            elif len(blockers) == 1:
                # Try removing the single blocker
                b = next(iter(blockers))
                new_selected = selected - {b}
                # Check if we can add x and maybe more
                if x not in new_selected:
                    new_selected.add(x)
                    # Try to add more
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, 513):
                            if y in new_selected:
                                continue
                            valid = True
                            for dd in range(1, (y - 1) // 3 + 1):
                                if (y - dd) in new_selected and (y - 2*dd) in new_selected and (y - 3*dd) in new_selected:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for dd in range(1, (512 - y) // 3 + 1):
                                if (y + dd) in new_selected and (y + 2*dd) in new_selected and (y + 3*dd) in new_selected:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for dd in range(1, y):
                                if (y + 2*dd) <= 512 and (y - dd) in new_selected and (y + dd) in new_selected and (y + 2*dd) in new_selected:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            for dd in range(1, (y - 1) // 2 + 1):
                                if (y + dd) <= 512 and (y - 2*dd) in new_selected and (y - dd) in new_selected and (y + dd) in new_selected:
                                    valid = False
                                    break
                            if valid:
                                new_selected.add(y)
                                changed = True
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        improved = True
    return selected


def construct():
    deadline = time.time() + 85  # Safety margin within 90s limit
    
    # Start with deterministic greedy
    best_set = _greedy_forward()
    best_size = len(best_set)
    
    # Try reverse greedy
    rev_set = _greedy_reverse()
    if len(rev_set) > best_size:
        best_set = rev_set
        best_size = len(rev_set)
    
    # Try multiple random orderings
    random.seed(42)
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        rand_set = _greedy_random_order(seed)
        if len(rand_set) > best_size:
            best_set = rand_set
            best_size = len(best_set)
    
    # Local improvement on the best set found
    if time.time() < deadline - 5:
        improved = _local_improve(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final verification
    if not _is_4ap_free(best_set):
        # Fallback to simple greedy
        best_set = _greedy_forward()
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
