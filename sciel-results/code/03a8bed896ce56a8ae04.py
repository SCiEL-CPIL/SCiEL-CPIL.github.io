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
    pair_conflict = np.zeros((npts, npts), dtype=object)
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
    
    # Greedy construction with randomization
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = set()
        S_list = []
        
        blocked = 0  # bitmask
        
        while True:
            if time.time() > deadline:
                break
            
            # Find all compatible points (not in S, not blocked)
            available = 0
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    available |= (1 << p)
            
            if available == 0:
                break
            
            # Score each available point by how many new points it would block
            best_p = None
            best_score = -1
            avail_list = [p for p in range(npts) if (available & (1 << p))]
            rng.shuffle(avail_list)
            
            for p in avail_list:
                # Count how many new points would be blocked
                new_blocked = 0
                for a in S:
                    new_blocked |= pair_conflict[p][a]
                new_blocked &= ~blocked
                new_blocked &= ~(1 << p)
                score = bin(new_blocked).count('1')
                # Add small random tiebreaker
                score += rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            S.add(best_p)
            S_list.append(best_p)
            # Update blocked
            for a in S:
                if a != best_p:
                    blocked |= pair_conflict[best_p][a]
        
        # Check if maximal
        if not is_maximal(S, blocked):
            # Try to add remaining unblocked points
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    # Check validity
                    valid = True
                    for a in S:
                        for b in S:
                            if a != b and (pair_conflict[a][b] & (1 << p)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S.add(p)
                        S_list.append(p)
                        for a in S:
                            if a != p:
                                blocked |= pair_conflict[p][a]
        
        return S_list, blocked
    
    # Simulated Annealing local search
    def simulated_annealing(initial_S, seed, max_iters=200):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        
        # Penalty function: size + lambda * (unblocked_count)
        # We want to minimize size while ensuring maximality
        lambda_val = 50.0  # penalty weight for non-maximality
        
        def objective(S_set):
            blocked = compute_blocked(S_set)
            unblocked = 0
            for p in range(npts):
                if p not in S_set and not (blocked & (1 << p)):
                    unblocked += 1
            return len(S_set) + lambda_val * unblocked
        
        current_obj = objective(S)
        T = 10.0  # initial temperature
        cooling = 0.99
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            if T < 0.01:
                break
            
            # Random move: remove a random point
            if len(S) > 0:
                remove_p = rng.choice(list(S))
                S.remove(remove_p)
            else:
                # Add a random point
                add_p = rng.randint(0, npts-1)
                S.add(add_p)
                remove_p = None
            
            new_obj = objective(S)
            
            # Accept if better or with probability exp(-(new-old)/T)
            delta_obj = new_obj - current_obj
            if delta_obj < 0 or rng.random() < np.exp(-delta_obj / T):
                current_obj = new_obj
                if len(S) < best_size and is_maximal(S):
                    best_S = list(S)
                    best_size = len(S)
            else:
                # Reject: revert
                if remove_p is not None:
                    S.add(remove_p)
                else:
                    S.discard(add_p)
            
            T *= cooling
        
        # Ensure best_S is valid and maximal
        if not is_valid(set(best_S)) or not is_maximal(set(best_S)):
            # Fallback to initial
            return initial_S
        
        return best_S
    
    # Local search: try to remove points and re-add to improve
    def local_search(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        
        max_iterations = 100
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= best_size:
                continue
            
            blocked = compute_blocked(S)
            
            # Try removing one point and see if we can do better
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            for remove_p in S_list:
                if len(S) - 1 >= best_size:
                    continue
                
                S.remove(remove_p)
                new_blocked = compute_blocked(S)
                
                # Try to add back different points
                added = False
                for add_p in range(npts):
                    if add_p in S or (new_blocked & (1 << add_p)):
                        continue
                    # Check if adding add_p keeps S valid (no isosceles)
                    valid = True
                    for a in S:
                        for b in S:
                            if a != b and (pair_conflict[a][b] & (1 << add_p)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S.add(add_p)
                        added = True
                        break
                
                if not added:
                    S.add(remove_p)
                    continue
                
                # Check if this new S is maximal
                final_blocked = compute_blocked(S)
                is_max = is_maximal(S, final_blocked)
                
                if is_max and len(S) < best_size:
                    best_S = list(S)
                    best_size = len(S)
                    improved = True
                else:
                    S.remove(add_p)
                    S.add(remove_p)
            
            if not improved:
                break
        
        return best_S
    
    # Branch and bound for finding smaller maximal independent sets
    def branch_and_bound(current_S, blocked_mask, depth, best_size, seed):
        if time.time() > deadline:
            return None
        
        if len(current_S) >= best_size:
            return None
        
        # Check if current_S is already maximal
        is_max = is_maximal(current_S, blocked_mask)
        
        if is_max:
            if len(current_S) < best_size:
                return list(current_S)
            return None
        
        # Find available points
        available = []
        for p in range(npts):
            if p not in current_S and not (blocked_mask & (1 << p)):
                available.append(p)
        
        if not available:
            return None
        
        # Sort by coverage (points that would be blocked)
        def coverage_score(p):
            new_blocked = 0
            for a in current_S:
                new_blocked |= pair_conflict[p][a]
            new_blocked &= ~blocked_mask
            return bin(new_blocked).count('1')
        
        available.sort(key=coverage_score, reverse=True)
        
        # Try each available point
        for p in available[:min(15, len(available))]:  # Limit branching
            if time.time() > deadline:
                break
            
            new_blocked = blocked_mask
            for a in current_S:
                new_blocked |= pair_conflict[p][a]
            
            result = branch_and_bound(
                current_S | {p}, 
                new_blocked, 
                depth + 1, 
                best_size, 
                seed
            )
            if result is not None:
                return result
        
        return None
    
    # Main: run multiple greedy + local search + SA iterations
    best_result = None
    best_size = npts
    
    # First, get a baseline greedy solution
    baseline_S, _ = greedy_construct(42)
    if baseline_S:
        best_result = baseline_S
        best_size = len(baseline_S)
    
    # Run multiple randomized greedy + local search + SA
    num_trials = 40
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        seed = 1000 + trial
        S, _ = greedy_construct(seed)
        if not S:
            continue
        
        # Local search
        improved_S = local_search(S, seed)
        
        if len(improved_S) < best_size:
            best_size = len(improved_S)
            best_result = improved_S
        
        # Simulated annealing
        sa_S = simulated_annealing(improved_S, seed + 5000, max_iters=100)
        if len(sa_S) < best_size:
            best_size = len(sa_S)
            best_result = sa_S
    
    # Try branch and bound from best solution
    if best_result is not None and time.time() < deadline:
        for remove_attempt in range(min(5, len(best_result))):
            if time.time() > deadline:
                break
            
            S_copy = set(best_result)
            to_remove = random.Random(2000 + remove_attempt).choice(list(S_copy))
            S_copy.remove(to_remove)
            
            blocked = compute_blocked(S_copy)
            
            result = branch_and_bound(S_copy, blocked, 0, best_size, 3000)
            if result is not None and len(result) < best_size:
                best_size = len(result)
                best_result = result
    
    # Final validation: ensure best_result is valid and maximal
    if best_result is not None:
        S_set = set(best_result)
        if not is_valid(S_set) or not is_maximal(S_set):
            # Fallback to baseline
            if baseline_S and is_valid(set(baseline_S)) and is_maximal(set(baseline_S)):
                best_result = baseline_S
            else:
                # Return empty set (shouldn't happen)
                best_result = []
    
    return [points[i] for i in best_result] if best_result else [points[0]]
