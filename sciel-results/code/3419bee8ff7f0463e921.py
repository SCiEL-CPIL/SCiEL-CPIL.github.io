"""Rewritten program: structural change to a conflict-aware greedy + local-search optimizer."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _precompute_conflicts(n=256):
    """Precompute for each pair (a,b) whether (a*b) % (a+b) == 0 (conflict)."""
    # conflicts[a][b] = True if a*b is divisible by a+b
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a][b] = True
                conflicts[b][a] = True
    return conflicts

def _greedy_build(conflicts, order, n=256):
    """Greedy: add numbers in the given order if they don't conflict with any already selected."""
    selected = set()
    for x in order:
        ok = True
        for y in selected:
            if conflicts[x][y]:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _local_search(conflicts, selected, n=256, max_iters=5000, deadline=None):
    """
    Local search: try single additions, single swaps, and double swaps to increase cardinality.
    """
    selected = set(selected)
    current_size = len(selected)
    
    # Precompute conflict counts for quick checks
    def conflict_count(x, sel):
        return sum(1 for y in sel if conflicts[x][y])
    
    def can_add(x, sel):
        return conflict_count(x, sel) == 0
    
    def try_add(sel, x):
        if can_add(x, sel):
            sel.add(x)
            return True
        return False
    
    best = set(selected)
    best_size = current_size
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Try to add any unselected number
        unselected = [x for x in range(1, n + 1) if x not in selected]
        random.shuffle(unselected)
        for x in unselected:
            if can_add(x, selected):
                selected.add(x)
                improved = True
                break
        
        if not improved:
            # Try swap: remove one, add two
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                temp = selected - {rem}
                # Try to add two numbers
                unselected = [x for x in range(1, n + 1) if x not in temp]
                random.shuffle(unselected)
                added = []
                for x in unselected:
                    if can_add(x, temp) and len(added) < 2:
                        temp.add(x)
                        added.append(x)
                if len(added) == 2:
                    selected = temp
                    improved = True
                    break
        
        if not improved:
            # Try removing one element that allows adding more
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                temp = selected - {rem}
                added = 0
                for x in range(1, n + 1):
                    if x not in temp and can_add(x, temp):
                        temp.add(x)
                        added += 1
                if len(temp) > len(selected):
                    selected = temp
                    improved = True
                    break
        
        if not improved:
            # Try a random restart-like perturbation: remove 2, try to add 3
            if len(selected) > 4:
                rem_pair = random.sample(list(selected), 2)
                temp = selected - set(rem_pair)
                added = 0
                for x in range(1, n + 1):
                    if x not in temp and can_add(x, temp):
                        temp.add(x)
                        added += 1
                if len(temp) > len(selected):
                    selected = temp
                    improved = True
        
        if not improved:
            break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 256
    conflicts = _precompute_conflicts(n)
    
    # Initial greedy with natural order
    order = list(range(1, n + 1))
    selected = _greedy_build(conflicts, order, n)
    
    # Try multiple random orderings for greedy
    best = set(selected)
    best_size = len(selected)
    
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random order
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_build(conflicts, order, n)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
    
    # Local search from best
    if time.time() < deadline:
        improved = _local_search(conflicts, best, n, max_iters=2000, deadline=deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # More local search iterations if time permits
    while time.time() < deadline - 5:
        improved = _local_search(conflicts, best, n, max_iters=1000, deadline=deadline)
        if len(improved) <= best_size:
            break
        best = improved
        best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
