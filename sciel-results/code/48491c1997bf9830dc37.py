"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_200(selected):
    """Check if any subset of selected sums to 200."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return True
        reachable |= new
    return False

def _reachable_sums(selected):
    """Compute all reachable sums up to 200."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        reachable |= new
    return reachable

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Greedy phase: add elements from high to low
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    
    # Local search: try to add back excluded elements
    excluded = [x for x in range(1, 101) if x not in selected]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded:
            if time.time() > deadline:
                break
            # Try adding x
            new_reachable = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reachable:
                selected.append(x)
                reachable |= new_reachable
                excluded.remove(x)
                improved = True
    
    return selected
# EVOLVE-BLOCK-END
