"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from fractions import Fraction

def _find_all_conflicts():
    """Find all (target, frozenset(subset)) where 1/target = sum(1/x for x in subset)."""
    conflicts = []
    universe = list(range(1, 25))
    for target in universe:
        tr = Fraction(1, target)
        others = [x for x in universe if x != target]
        # Sort by value descending (so reciprocals ascending) for pruning
        others_desc = sorted(others, reverse=True)
        for size in range(1, len(others_desc) + 1):
            for subset in itertools.combinations(others_desc, size):
                s = sum(Fraction(1, x) for x in subset)
                if s == tr:
                    conflicts.append((target, frozenset(subset)))
                elif s > tr:
                    # For this particular ordering, further elements are smaller (larger reciprocals)
                    # so sum will only increase. But combinations are not monotonic in this way.
                    # Actually with combinations in sorted order, we can't prune easily.
                    pass
    return conflicts

def _check_valid(selected_set, conflicts):
    """Return True if no conflict is active in selected_set."""
    for target, subset in conflicts:
        if target in selected_set and subset.issubset(selected_set):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 70.0  # safety margin
    
    conflicts = _find_all_conflicts()
    
    # Greedy: try multiple orderings
    best = set()
    best_size = 0
    
    # Ordering 1: large to small
    for order in [list(range(24, 0, -1)), list(range(1, 25))]:
        cur = set()
        for x in order:
            test = cur | {x}
            if _check_valid(test, conflicts):
                cur = test
        if len(cur) > best_size:
            best_size = len(cur)
            best = cur
    
    # Local search: try to add elements
    def local_search(cur):
        cur = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 25):
                if x in cur:
                    continue
                test = cur | {x}
                if _check_valid(test, conflicts):
                    cur = test
                    improved = True
                    break
        return cur
    
    best = local_search(best)
    best_size = len(best)
    
    # Try removing one element and adding two
    def swap_search(cur):
        cur = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(cur):
                for y in range(1, 25):
                    if y in cur or y == x:
                        continue
                    for z in range(1, 25):
                        if z in cur or z == x or z == y:
                            continue
                        test = (cur - {x}) | {y, z}
                        if len(test) > len(cur) and _check_valid(test, conflicts):
                            cur = test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return cur
    
    best = swap_search(best)
    best_size = len(best)
    
    # Try different starting points
    if time.time() < deadline - 5:
        for n in range(10, 25):
            if time.time() >= deadline:
                break
            start_set = set(range(25 - n, 25))
            cand = local_search(start_set)
            cand = swap_search(cand)
            if len(cand) > best_size:
                best_size = len(cand)
                best = cand
    
    result = list(best)
    if not result:
        result = list(range(13, 25))
    
    # Final safety check
    if not _check_valid(set(result), conflicts):
        # Fallback to known valid set
        result = list(range(13, 25))
    
    return result

# EVOLVE-BLOCK-END
