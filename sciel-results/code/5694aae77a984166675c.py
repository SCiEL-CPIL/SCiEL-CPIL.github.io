"""Rewritten program: structural change to a conflict-aware greedy + local-search optimizer."""
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
    Optimized local search with maintained conflict counts for O(1) add/remove checks.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    # Maintain conflict counts: conf_count[x] = number of selected elements conflicting with x
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in range(1, n + 1):
            if y != x and conflicts[x][y]:
                conf_count[y] += 1
    
    def try_add(x):
        """Add x if it has no conflicts with selected. Returns True if added."""
        if conf_count[x] == 0:
            selected.add(x)
            for y in range(1, n + 1):
                if y != x and conflicts[x][y]:
                    conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        """Remove x from selected."""
        selected.discard(x)
        for y in range(1, n + 1):
            if y != x and conflicts[x][y]:
                conf_count[y] -= 1
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Try to add any unselected number with zero conflicts
        unselected = [x for x in range(1, n + 1) if x not in selected]
        random.shuffle(unselected)
        for x in unselected:
            if conf_count[x] == 0:
                try_add(x)
                improved = True
                break
        
        if not improved:
            # Try swap: remove one, add two
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                # Temporarily remove
                remove(rem)
                # Try to add two numbers
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                added = []
                for x in unselected:
                    if conf_count[x] == 0 and len(added) < 2:
                        try_add(x)
                        added.append(x)
                if len(added) == 2:
                    improved = True
                    break
                else:
                    # Undo: remove what we added, put rem back
                    for x in added:
                        remove(x)
                    selected.add(rem)
                    for y in range(1, n + 1):
                        if y != rem and conflicts[rem][y]:
                            conf_count[y] += 1
        
        if not improved:
            # Try removing one element that allows adding more
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                # Temporarily remove
                remove(rem)
                # Greedily add as many as possible
                added = 0
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                        added += 1
                if len(selected) > len(best):
                    improved = True
                    break
                else:
                    # Undo: remove what we added, put rem back
                    to_remove = [x for x in selected if x != rem and x not in best]
                    for x in to_remove:
                        remove(x)
                    selected.add(rem)
                    for y in range(1, n + 1):
                        if y != rem and conflicts[rem][y]:
                            conf_count[y] += 1
        
        if not improved:
            # Try a random restart-like perturbation: remove 2, try to add 3
            if len(selected) > 4:
                rem_pair = random.sample(list(selected), 2)
                for r in rem_pair:
                    remove(r)
                # Greedily add as many as possible
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                if len(selected) > len(best):
                    improved = True
                else:
                    # Undo: this is complex, just restart from best
                    selected = set(best)
                    conf_count = [0] * (n + 1)
                    for x in selected:
                        for y in range(1, n + 1):
                            if y != x and conflicts[x][y]:
                                conf_count[y] += 1
        
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
