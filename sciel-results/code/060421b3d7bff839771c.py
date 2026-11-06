import time

_START = time.time()
_DEADLINE = 90.0
_MARGIN = 1.0  # safety margin in seconds
_MASK = (1 << 201) - 1  # bits 0..200

def _valid(s, cap=200):
    """Check if cap is NOT reachable as a subset sum of s."""
    reach = 1  # bit 0 set
    for x in s:
        reach |= (reach << x)
        reach &= _MASK
    return not ((reach >> cap) & 1)

def construct():
    # Greedy seed using bitset: descending order
    selected = []
    reach = 1  # bit 0 set
    for x in range(100, 0, -1):
        # Check if adding x would make 200 reachable
        # 200 becomes reachable if (reach << x) has bit 200 set
        # i.e., if reach has bit (200-x) set
        if not ((reach >> (200 - x)) & 1):
            selected.append(x)
            reach |= (reach << x)
            reach &= _MASK
    best = list(selected)

    # Local improvement: try to add back excluded elements
    excluded = [x for x in range(100, 0, -1) if x not in set(best)]
    changed = True
    while changed and (time.time() - _START) < (_DEADLINE - _MARGIN):
        changed = False
        best_set = set(best)
        reach = 1
        for x in best:
            reach |= (reach << x)
            reach &= _MASK
        for x in excluded:
            if x in best_set:
                continue
            # would adding x create 200?
            if not ((reach >> (200 - x)) & 1):
                best.append(x)
                best_set.add(x)
                reach |= (reach << x)
                reach &= _MASK
                changed = True
        if not changed:
            break

    # Final exact verification
    if not _valid(best):
        best = [1]
    if (time.time() - _START) > (_DEADLINE - _MARGIN):
        if not _valid(best):
            best = [1]
    return best
