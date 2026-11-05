"""Spectral-guided greedy + local-search optimizer for independent set in conflict graph."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _precompute_conflicts(n=256):
    """Precompute for each pair (a,b) whether (a*b) % (a+b) == 0 (conflict)."""
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a][b] = True
                conflicts[b][a] = True
    return conflicts

def _spectral_order(conflicts, n=256):
    """
    Compute eigenvector of conflict adjacency matrix to get a spectral ordering.
    Vertices with lower eigenvector values are in the 'low-conflict core'.
    Returns list of vertices sorted by eigenvector value (ascending = least connected first).
    """
    # Build adjacency matrix
    adj = [[0.0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if conflicts[a][b]:
                adj[a][b] = 1.0
                adj[b][a] = 1.0
    
    # Power iteration to find the principal eigenvector
    # Start with uniform vector
    v = [0.0] * (n + 1)
    for i in range(1, n + 1):
        v[i] = 1.0
    
    for _ in range(100):
        new_v = [0.0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                new_v[i] += adj[i][j] * v[j]
        # Normalize
        norm = sum(x * x for x in new_v) ** 0.5
        if norm > 1e-12:
            for i in range(1, n + 1):
                new_v[i] /= norm
        v = new_v
    
    # Sort vertices by eigenvector value ascending (low connection = good core)
    order = list(range(1, n + 1))
    order.sort(key=lambda x: v[x])
    return order

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
    
    def conflict_count(x, sel):
        return sum(1 for y in sel if conflicts[x][y])
    
    def can_add(x, sel):
        return conflict_count(x, sel) == 0
    
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
    
    # Spectral ordering: start from low-conflict core
    spec_order = _spectral_order(conflicts, n)
    selected = _greedy_build(conflicts, spec_order, n)
    
    best = set(selected)
    best_size = len(selected)
    
    rng = random.Random(42)
    
    # Try multiple random orderings for greedy
    for trial in range(50):
        if time.time() > deadline:
            break
        
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
