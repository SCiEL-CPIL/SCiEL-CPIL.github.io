"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """Build conflict structure: for each x, the set of (u,v) pairs that form a bad triple with x."""
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].add((min(b,c), max(b,c)))
            conflicts[b].add((min(a,c), max(a,c)))
            conflicts[c].add((min(a,b), max(a,b)))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for (u, v) in conflicts[x]:
        if u in selected_set and v in selected_set:
            return False
    return True

def _greedy(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_improve(selected, conflicts, time_limit=30):
    """Improve by trying to swap out one element and add another (possibly multiple)."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            for x in range(1, 129):
                if x in candidate:
                    continue
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    # Try to greedily add more
                    for y in range(1, 129):
                        if y not in new_set and _can_add(y, new_set, conflicts):
                            new_set.add(y)
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def _swap_improve(selected, conflicts, time_limit=30):
    """Try to improve by swapping: remove one, add another, then add more."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            for x in range(1, 129):
                if x in candidate:
                    continue
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    # Try to add more elements
                    for y in range(1, 129):
                        if y not in new_set and _can_add(y, new_set, conflicts):
                            new_set.add(y)
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def _anneal(selected, conflicts, time_limit=30):
    """Simulated annealing: try random swaps and additions."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    current_temp = 10.0
    cooling = 0.95
    
    while time.time() - start < time_limit:
        if current_temp < 0.1:
            break
        current_temp *= cooling
        
        if selected_set:
            rem = random.choice(list(selected_set))
            candidate = selected_set - {rem}
            
            available = [x for x in range(1, 129) if x not in candidate]
            if available:
                x = random.choice(available)
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    # Try to greedily add more
                    for y in range(1, 129):
                        if y not in new_set and _can_add(y, new_set, conflicts):
                            new_set.add(y)
                    
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                    elif random.random() < current_temp / 10.0:
                        selected_set = new_set
        else:
            for x in range(1, 129):
                if _can_add(x, selected_set, conflicts):
                    selected_set.add(x)
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute conflict structure
    conflicts = _build_conflicts()
    
    best = []
    best_size = 0
    
    # Initial greedy with natural order
    sel = _greedy(conflicts, list(range(1, 129)))
    best = sel
    best_size = len(sel)
    
    # Local improvement
    if time.time() < deadline - 15:
        improved = _local_improve(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Swap-based improvement
    if time.time() < deadline - 15:
        improved = _swap_improve(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Multiple random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20 and restarts < 100:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Improve if close to best
        if len(sel) >= best_size - 2 and time.time() < deadline - 25:
            improved = _local_improve(sel, conflicts, time_limit=5)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
            
            if time.time() < deadline - 20:
                improved = _swap_improve(best, conflicts, time_limit=5)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(best)
    
    # Final annealing pass on best solution
    if time.time() < deadline - 5 and len(best) > 0:
        improved = _anneal(best, conflicts, time_limit=15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Final local improvement
    if time.time() < deadline - 2 and len(best) > 0:
        improved = _local_improve(best, conflicts, time_limit=3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
