"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _valid_subset(S, target=200):
    """Check that no subset of S sums to target. S is a set of ints."""
    reachable = [False] * (target + 1)
    reachable[0] = True
    for x in S:
        for s in range(target - x, -1, -1):
            if reachable[s]:
                if s + x == target:
                    return False
                reachable[s + x] = True
    return True

def _greedy_construct(order, target=200):
    """Greedy: include x if it doesn't create target in reachable sums."""
    selected = []
    reachable = [False] * (target + 1)
    reachable[0] = True
    for x in order:
        creates_target = False
        new_reachable = [False] * (target + 1)
        for s in range(target + 1):
            if reachable[s]:
                new_reachable[s] = True
                if s + x <= target:
                    if s + x == target:
                        creates_target = True
                        break
                    new_reachable[s + x] = True
        if not creates_target:
            selected.append(x)
            reachable = new_reachable
    return selected

def _local_improve(best_set, target=200, deadline=None):
    """Try to add excluded elements, and also try swaps (remove one, add one/two)."""
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        # Phase 1: try to add any single excluded element
        for x in range(1, 101):
            if x not in best_set:
                trial = best_set | {x}
                if _valid_subset(trial, target):
                    best_set = trial
                    improved = True
                    break
        # Phase 2: try swaps - remove one element, add one or two others
        if not improved:
            removed = None
            added = None
            for x in sorted(best_set):
                candidate = best_set - {x}
                for y in range(1, 101):
                    if y not in candidate:
                        trial = candidate | {y}
                        if len(trial) > len(best_set) and _valid_subset(trial, target):
                            removed = x
                            added = [y]
                            break
                if added:
                    break
                for y in range(1, 101):
                    if y not in candidate and y != x:
                        for z in range(y + 1, 101):
                            if z not in candidate and z != x:
                                trial = candidate | {y, z}
                                if _valid_subset(trial, target):
                                    removed = x
                                    added = [y, z]
                                    break
                        if added:
                            break
                if added:
                    break
            if added:
                best_set = best_set - {removed}
                for y in added:
                    best_set.add(y)
                improved = True
    
    return best_set

def construct():
    deadline = time.time() + 85
    best = []
    
    best = _greedy_construct(list(range(100, 0, -1)))
    
    cand = _greedy_construct(list(range(1, 101)))
    if len(cand) > len(best):
        best = cand
    
    base = list(range(1, 101))
    for _ in range(300):
        if time.time() > deadline:
            break
        random.shuffle(base)
        cand = _greedy_construct(base)
        if len(cand) > len(best):
            best = cand
    
    best_set = set(best)
    best_set = _local_improve(best_set, 200, deadline)
    
    best = list(best_set)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
