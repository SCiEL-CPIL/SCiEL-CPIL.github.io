"""Rewritten program: Exact solver (PULP) with robust heuristic fallback."""
import time
import random
from math import gcd

def _precompute_conflicts(n=256):
    """Precompute conflict matrix and neighbor lists."""
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    neighbors = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a][b] = True
                conflicts[b][a] = True
                neighbors[a].append(b)
                neighbors[b].append(a)
    return conflicts, neighbors

def _solve_with_pulp(conflicts_list, n=256, timeout=60):
    try:
        import pulp
        prob = pulp.LpProblem("MIS", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(1, n + 1)]
        
        prob += pulp.lpSum(x)
        
        for a, b in conflicts_list:
            prob += x[a-1] + x[b-1] <= 1
            
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
        prob.solve(solver)
        
        if pulp.LpStatus[prob.status] == "Optimal":
            selected = [i for i in range(1, n + 1) if x[i-1].value() == 1]
            return selected
        else:
            return None
    except Exception:
        return None

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

def _local_search(conflicts, neighbors, selected, n=256, max_iters=2000, deadline=None):
    """
    Optimized local search with maintained conflict counts using neighbor lists.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in neighbors[x]:
            if y != x:
                conf_count[y] += 1
    
    def try_add(x):
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                if y != x:
                    conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        selected.discard(x)
        for y in neighbors[x]:
            if y != x:
                conf_count[y] -= 1
    
    def snapshot_conf_count():
        return list(conf_count)
    
    def restore_conf_count(saved):
        for i in range(1, n + 1):
            conf_count[i] = saved[i]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try to add any unselected number with zero conflicts
        unselected = [x for x in range(1, n + 1) if x not in selected]
        random.shuffle(unselected)
        for x in unselected:
            if conf_count[x] == 0:
                try_add(x)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try 1-for-1 swap
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                saved_cc = snapshot_conf_count()
                saved_sel = set(selected)
                remove(rem)
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                found = False
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                        found = True
                        break
                if found:
                    improved = True
                    break
                else:
                    restore_conf_count(saved_cc)
                    selected = saved_sel
        
        if not improved:
            # Phase 3: Try remove-one-add-many
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                saved_cc = snapshot_conf_count()
                saved_sel = set(selected)
                remove(rem)
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                added_count = 0
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                        added_count += 1
                if added_count > 1:
                    improved = True
                    break
                else:
                    restore_conf_count(saved_cc)
                    selected = saved_sel
        
        if not improved:
            # Phase 4: Perturbation
            if len(selected) > 4:
                k = random.choice([2, 3])
                rem_pair = random.sample(list(selected), k)
                for r in rem_pair:
                    remove(r)
                # Greedily add
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                if len(selected) > best_size:
                    improved = True
                else:
                    selected = set(best)
                    conf_count = [0] * (n + 1)
                    for x in selected:
                        for y in neighbors[x]:
                            if y != x:
                                conf_count[y] += 1
        
        if not improved:
            break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    # Precompute conflicts for heuristics
    conflicts, neighbors = _precompute_conflicts(n)
    
    # List of conflicts for PULP
    conflict_pairs = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if conflicts[a][b]:
                conflict_pairs.append((a, b))
                
    # 1. Try Exact Solver
    if time.time() < deadline - 10:
        result = _solve_with_pulp(conflict_pairs, n, timeout=60)
        if result is not None and len(result) > 0:
            return sorted(result)
            
    # 2. Heuristic Fallback
    # Degree-sorted order
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = len(neighbors[i])
        
    order_degree = sorted(range(1, n + 1), key=lambda x: degrees[x])
    selected = _greedy_build(conflicts, order_degree, n)
    
    best = set(selected)
    best_size = len(selected)
    
    # Try a few random orders
    rng = random.Random(42)
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_build(conflicts, order, n)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
            
    # Local Search
    if time.time() < deadline - 5:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=3000, deadline=deadline - 3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            
    return sorted(best)
