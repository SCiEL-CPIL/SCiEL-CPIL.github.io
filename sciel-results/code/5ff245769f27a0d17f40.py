import time
import random
import numpy as np

N = 16
DEADLINE = 300.0  # seconds, leave margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 15.0  # 15s safety margin
    
    points = [(x, y) for x in range(N) for y in range(N)]
    npts = len(points)
    
    # Precompute isosceles conflicts using bitmasks for speed
    # pair_conflict[i][j] = bitmask of k such that i, j, k form an isosceles triangle
    # Using a list of lists of Python integers for fast bitwise ops
    pair_conflict = [[0]*npts for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            mask = 0
            for k in range(npts):
                if k != i and k != j and _is_isosceles(points[i], points[j], points[k]):
                    mask |= (1 << k)
            pair_conflict[i][j] = mask
            pair_conflict[j][i] = mask
    
    # Helper: compute blocked mask for a given set S
    def compute_blocked(S):
        blocked = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                blocked |= pair_conflict[S_list[i]][S_list[j]]
        return blocked
    
    # Helper: check if S is maximal
    def is_maximal(S, blocked=None):
        if blocked is None:
            blocked = compute_blocked(S)
        for p in range(npts):
            if p not in S and not (blocked & (1 << p)):
                return False
        return True
    
    # Helper: validate that S has no isosceles triples
    def is_valid(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    if _is_isosceles(points[S_list[i]], points[S_list[j]], points[S_list[k]]):
                        return False
        return True

    # Optimized Greedy construction
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = set()
        S_list = []
        blocked = 0
        
        while True:
            if time.time() > deadline:
                break
            
            # Find available points
            avail_list = []
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    avail_list.append(p)
            
            if not avail_list:
                break
            
            # Score each available point
            best_p = None
            best_score = -1
            rng.shuffle(avail_list)
            
            for p in avail_list:
                new_blocked = 0
                for a in S:
                    new_blocked |= pair_conflict[p][a]
                new_blocked &= ~blocked
                new_blocked &= ~(1 << p)
                score = bin(new_blocked).count('1')
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            S.add(best_p)
            S_list.append(best_p)
            for a in S:
                if a != best_p:
                    blocked |= pair_conflict[best_p][a]
        
        # Ensure maximality by adding any remaining valid points
        changed = True
        while changed:
            changed = False
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    # Check if p can be added without forming isosceles with existing pairs
                    # Actually, if p is not blocked, it means no pair in S forms isosceles with p.
                    # So it is safe to add.
                    S.add(p)
                    S_list.append(p)
                    for a in S:
                        if a != p:
                            blocked |= pair_conflict[p][a]
                    changed = True
        
        return S_list, blocked

    # Optimized Local Search: 1-1 swap
    def local_search(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        blocked = compute_blocked(S)
        
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= best_size:
                continue
            
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            for remove_p in S_list:
                if len(S) - 1 >= best_size:
                    continue
                
                S.remove(remove_p)
                # Recalculate blocked
                new_blocked = compute_blocked(S)
                
                # Try to add a point that was blocked by remove_p or was available
                # We want to find a point q such that S - {p} + {q} is maximal and valid
                # Actually, we just want to add ANY valid point to see if we can get a smaller set?
                # No, we remove 1, add 1. Size stays same.
                # We want to reduce size. So we should try to remove 1 and add 0?
                # If we remove 1 and don't add, is it maximal?
                # If S - {p} is maximal, then we found a smaller maximal set.
                
                # Check if S - {p} is maximal
                is_max = is_maximal(S, new_blocked)
                if is_max:
                    best_S = list(S)
                    best_size = len(S)
                    improved = True
                    blocked = new_blocked
                    break # Restart search with new S
                
                # If not maximal, try to add a point to make it maximal?
                # If we add a point, size increases. We only care about reducing size.
                # So if S-{p} is not maximal, we cannot simply drop p.
                # We must restore p.
                S.add(remove_p)
            
            if not improved:
                break
        
        return best_S

    # Main
    best_result = None
    best_size = npts
    
    # Baseline
    baseline_S, _ = greedy_construct(42)
    if baseline_S:
        best_result = baseline_S
        best_size = len(baseline_S)
    
    # Multiple trials
    num_trials = 100 # Increase trials due to faster greedy
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        seed = 1000 + trial
        S, _ = greedy_construct(seed)
        if not S:
            continue
        
        improved_S = local_search(S, seed)
        
        if len(improved_S) < best_size:
            best_size = len(improved_S)
            best_result = improved_S
    
    # Final validation
    if best_result is not None:
        S_set = set(best_result)
        if not is_valid(S_set) or not is_maximal(S_set):
            if baseline_S and is_valid(set(baseline_S)) and is_maximal(set(baseline_S)):
                best_result = baseline_S
            else:
                best_result = []
    
    return [points[i] for i in best_result] if best_result else [points[0]]
