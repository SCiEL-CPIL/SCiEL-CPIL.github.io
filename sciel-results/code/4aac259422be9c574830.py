"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """
    For each element x, precompute the set of pairs (y,z) such that {x,y,z} is a bad triple.
    conflicts[x] = set of (y, z) tuples where y < z, and {x,y,z} forms a bad triple.
    """
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].add((b, c))
            conflicts[b].add((a, c))
            conflicts[c].add((a, b))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for (y, z) in conflicts[x]:
        if y in selected_set and z in selected_set:
            return False
    return True

def _greedy(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = set()
    for x in order:
        if _can_add(x, selected, conflicts):
            selected.add(x)
    return selected

def _greedy_conflict_aware(conflicts, order):
    """
    Greedy with dynamic reordering: at each step, pick the candidate
    that creates the fewest new conflicts with already-selected elements.
    """
    selected = set()
    remaining = set(order)
    
    while remaining:
        best_x = None
        best_conflict_count = float('inf')
        
        for x in remaining:
            if not _can_add(x, selected, conflicts):
                continue
            # Count how many new conflicts this would create
            conflict_count = 0
            for (y, z) in conflicts[x]:
                if y in selected or z in selected:
                    conflict_count += 1
            if conflict_count < best_conflict_count:
                best_conflict_count = conflict_count
                best_x = x
                if conflict_count == 0:
                    break
        
        if best_x is None:
            break
        
        selected.add(best_x)
        remaining.discard(best_x)
    
    return selected

def _local_search(selected, conflicts, time_limit=15):
    """
    Local search: try additions, single swaps, and double swaps.
    Returns improved set or the original if no improvement.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding any element
        for x in range(1, 129):
            if x in selected:
                continue
            if _can_add(x, selected, conflicts):
                selected.add(x)
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(selected)
                improved = True
        
        # Try single swaps: remove one, add one
        if not improved:
            for rem in list(selected):
                candidate = selected - {rem}
                for x in range(1, 129):
                    if x in candidate:
                        continue
                    if _can_add(x, candidate, conflicts):
                        new_set = candidate | {x}
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(best)
                            selected = new_set
                            improved = True
                            break
                if improved:
                    break
        
        # Try double swaps: remove two, add two
        if not improved and len(selected) >= 2:
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    rem1, rem2 = sel_list[i], sel_list[j]
                    candidate = selected - {rem1, rem2}
                    # Try to add two elements
                    added = []
                    for x in range(1, 129):
                        if x in candidate or x == rem1 or x == rem2:
                            continue
                        if _can_add(x, candidate, conflicts):
                            added.append(x)
                            candidate.add(x)
                            if len(added) == 2:
                                break
                    
                    if len(added) == 2:
                        new_size = len(candidate)
                        if new_size > best_size:
                            best = candidate
                            best_size = new_size
                            selected = candidate
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def _simulated_annealing(conflicts, time_limit=20, start_set=None):
    """
    Simulated annealing with double-swap moves to escape local optima.
    """
    if start_set is not None:
        current = set(start_set)
    else:
        order = list(range(1, 129))
        random.shuffle(order)
        current = _greedy(conflicts, order)
    
    best = set(current)
    best_size = len(current)
    
    start = time.time()
    temp = 15.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        temp *= cooling
        if temp < 0.005:
            break
        
        move_type = random.random()
        
        if move_type < 0.25:
            # Add a random element
            candidates = [x for x in range(1, 129) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            if _can_add(x, current, conflicts):
                current.add(x)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                if current:
                    rem = random.choice(list(current))
                    candidate = current - {rem}
                    if _can_add(x, candidate, conflicts):
                        current = candidate | {x}
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
        
        elif move_type < 0.45:
            # Remove a random element
            if current:
                rem = random.choice(list(current))
                current.discard(rem)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
        
        elif move_type < 0.75:
            # Single swap
            if current:
                rem = random.choice(list(current))
                candidate = current - {rem}
                candidates = [x for x in range(1, 129) if x not in candidate]
                if candidates:
                    x = random.choice(candidates)
                    if _can_add(x, candidate, conflicts):
                        current = candidate | {x}
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                    else:
                        if random.random() < temp / 15.0:
                            current = candidate
        
        else:
            # Double swap: remove two, add two
            if len(current) >= 2:
                rem1, rem2 = random.sample(list(current), 2)
                candidate = current - {rem1, rem2}
                candidates = [x for x in range(1, 129) if x not in candidate]
                if len(candidates) >= 2:
                    x1, x2 = random.sample(candidates, 2)
                    # Try adding x1 then x2
                    if _can_add(x1, candidate, conflicts):
                        candidate.add(x1)
                        if _can_add(x2, candidate, conflicts):
                            candidate.add(x2)
                            current = candidate
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(current)
                        else:
                            candidate.discard(x1)
                            # Try x2 first
                            if _can_add(x2, candidate, conflicts):
                                candidate.add(x2)
                                if _can_add(x1, candidate, conflicts):
                                    candidate.add(x1)
                                    current = candidate
                                    if len(current) > best_size:
                                        best = set(current)
                                        best_size = len(current)
                                else:
                                    candidate.discard(x2)
                    else:
                        # Accept removal with some probability
                        if random.random() < temp / 20.0:
                            current = candidate
    
    return best

def _is_valid(selected, conflicts):
    """Check if selected set has no bad triple."""
    sel_set = set(selected)
    for x in sel_set:
        for (y, z) in conflicts[x]:
            if y in sel_set and z in sel_set:
                return False
    return True

def _conflict_aware_polish(selected, conflicts, time_limit=10):
    """
    Final polish: try to replace elements with ones that have fewer conflicts,
    and try to add elements that fit well.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try to add any remaining element
        for x in range(1, 129):
            if x not in selected and _can_add(x, selected, conflicts):
                selected.add(x)
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(selected)
                improved = True
        
        # Try conflict-aware swaps: replace an element with one that has fewer conflicts
        if not improved:
            for rem in list(selected):
                rem_conflicts = len(conflicts[rem])
                candidate = selected - {rem}
                for x in range(1, 129):
                    if x in candidate:
                        continue
                    if not _can_add(x, candidate, conflicts):
                        continue
                    # Prefer elements with fewer conflicts
                    if len(conflicts[x]) < rem_conflicts:
                        new_set = candidate | {x}
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(best)
                            selected = new_set
                            improved = True
                            break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    conflicts = _build_conflicts()
    
    best = set()
    best_size = 0
    
    random.seed(42)
    
    # Natural order greedy
    sel = _greedy(conflicts, list(range(1, 129)))
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Reverse order greedy
    if time.time() < deadline - 5:
        sel = _greedy(conflicts, list(range(128, 0, -1)))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Conflict-aware greedy
    if time.time() < deadline - 5:
        sel = _greedy_conflict_aware(conflicts, list(range(1, 129)))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Multiple random greedy starts with local search
    for restart in range(300):
        if time.time() > deadline - 3:
            break
        
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Local search on promising solutions
        if len(sel) >= best_size - 3 and time.time() < deadline - 8:
            improved = _local_search(sel, conflicts, time_limit=4)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Simulated annealing from best solution
    if time.time() < deadline - 12:
        sa_result = _simulated_annealing(conflicts, time_limit=25, start_set=best)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Additional SA runs from different starts
    for sa_restart in range(3):
        if time.time() > deadline - 8:
            break
        order = list(range(1, 129))
        random.shuffle(order)
        sa_start = _greedy(conflicts, order)
        if len(sa_start) >= best_size - 2:
            sa_result = _simulated_annealing(conflicts, time_limit=10, start_set=sa_start)
            if len(sa_result) > best_size:
                best = sa_result
                best_size = len(best)
    
    # Final local search on the best
    if time.time() < deadline - 5 and best:
        improved = _local_search(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Conflict-aware polish
    if time.time() < deadline - 2 and best:
        improved = _conflict_aware_polish(best, conflicts, time_limit=5)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    if not _is_valid(best, conflicts):
        best = _greedy(conflicts, list(range(1, 129)))
    
    return list(best)
# EVOLVE-BLOCK-END
