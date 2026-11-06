"""Combined approach: Greedy construction + SA-guided Minimality Reduction."""
import time
import random
import math

N = 16
DEADLINE = 325.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_maximal(S, all_points):
    """
    Check if S is a maximal no-isosceles set.
    S must be no-isosceles (assumed true by construction).
    Maximal: for every p not in S, there exist a, b in S such that {p, a, b} is isosceles.
    """
    S_set = set(S)
    S_list = list(S)
    n_S = len(S_list)
    
    if n_S < 2:
        return False
        
    for p in all_points:
        if p in S_set:
            continue
            
        blocked = False
        for i in range(n_S):
            if blocked:
                break
            a = S_list[i]
            da = _dist2(p, a)
            for j in range(i + 1, n_S):
                b = S_list[j]
                d_pb = _dist2(p, b)
                d_ab = _dist2(a, b)
                
                if da == d_pb or da == d_ab or d_pb == d_ab:
                    blocked = True
                    break
                    
        if not blocked:
            return False
            
    return True

def _reduce_to_minimal(S, all_points, time_budget=5.0):
    """
    Use simulated annealing to find a minimal maximal set.
    State: set of points (represented as frozenset for hashability)
    Objective: minimize |S| subject to S being maximal no-isosceles
    Moves: remove a point (if still maximal), add a point (if no-isosceles preserved)
    """
    S_list = list(S)
    S_set = frozenset(S_list)
    current_size = len(S_list)
    
    best_set = S_list[:]
    best_size = current_size
    
    start_time = time.time()
    
    # Phase 1: Greedy removal (fast, deterministic)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = S_list[:i] + S_list[i+1:]
            
            if len(S_test) < 2:
                continue
                
            if _is_maximal(S_test, all_points):
                S_list = S_test
                changed = True
                break
    
    best_set = S_list[:]
    best_size = len(S_list)
    
    # Phase 2: Simulated annealing for further reduction
    if time.time() - start_time < time_budget:
        T = 1.0
        cooling = 0.95
        min_T = 0.01
        
        current_S = S_list[:]
        current_size = len(current_S)
        
        sa_start = time.time()
        
        while T > min_T and time.time() - sa_start < time_budget - (time.time() - start_time):
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to remove a point
                if len(current_S) <= 2:
                    T *= cooling
                    continue
                    
                idx = random.randint(0, len(current_S) - 1)
                p_remove = current_S[idx]
                new_S = current_S[:idx] + current_S[idx+1:]
                
                if len(new_S) < 2:
                    T *= cooling
                    continue
                    
                # Check if new_S is still maximal
                if _is_maximal(new_S, all_points):
                    new_size = len(new_S)
                    delta = new_size - current_size  # negative = improvement
                    
                    # Accept if improvement or with probability exp(-delta/T)
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_S = new_S
                        current_size = new_size
                        
                        if current_size < best_size:
                            best_size = current_size
                            best_set = current_S[:]
                else:
                    # Move rejected, stay put
                    pass
                    
            else:
                # Try to add a point (to explore, then potentially remove others)
                candidates = [p for p in all_points if p not in set(current_S)]
                if not candidates:
                    T *= cooling
                    continue
                    
                p_add = random.choice(candidates)
                
                # Check if adding p_add preserves no-isosceles property
                valid = True
                for i in range(len(current_S)):
                    if not valid:
                        break
                    a = current_S[i]
                    da = _dist2(p_add, a)
                    for j in range(i + 1, len(current_S)):
                        b = current_S[j]
                        d_pb = _dist2(p_add, b)
                        d_ab = _dist2(a, b)
                        if da == d_pb or da == d_ab or d_pb == d_ab:
                            valid = False
                            break
                
                if valid:
                    new_S = current_S + [p_add]
                    new_size = len(new_S)
                    delta = new_size - current_size  # positive = worse
                    
                    # Accept with probability exp(-delta/T)
                    if random.random() < math.exp(-delta / T):
                        current_S = new_S
                        current_size = new_size
                        
                # After potential add, try greedy removal again
                if time.time() - sa_start < time_budget - (time.time() - start_time):
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(current_S)):
                            p = current_S[i]
                            S_test = current_S[:i] + current_S[i+1:]
                            
                            if len(S_test) < 2:
                                continue
                                
                            if _is_maximal(S_test, all_points):
                                current_S = S_test
                                current_size = len(current_S)
                                changed = True
                                break
                            
                            if time.time() - sa_start > time_budget - (time.time() - start_time):
                                break
                            
                            if changed:
                                break
                            
                        if time.time() - sa_start > time_budget - (time.time() - start_time):
                            break
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_set = current_S[:]
            
            T *= cooling
            
            # Periodic greedy reduction
            if int((1.0 - T) * 100) % 5 == 0 and time.time() - sa_start < time_budget - (time.time() - start_time):
                changed = True
                while changed:
                    changed = False
                    for i in range(len(current_S)):
                        p = current_S[i]
                        S_test = current_S[:i] + current_S[i+1:]
                        
                        if len(S_test) < 2:
                            continue
                            
                        if _is_maximal(S_test, all_points):
                            current_S = S_test
                            current_size = len(current_S)
                            changed = True
                            break
                        
                        if time.time() - sa_start > time_budget - (time.time() - start_time):
                            break
                        
                        if changed:
                            break
                        
                    if time.time() - sa_start > time_budget - (time.time() - start_time):
                        break
                    
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_S[:]
    
    return best_set

def construct():
    start_time = time.time()
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_total = len(all_points)
    
    best_set = None
    best_size = n_total + 1
    
    rng = random.Random(42)
    
    for attempt in range(1000):
        if time.time() - start_time > DEADLINE:
            break
            
        # Greedy construction
        order = all_points[:]
        rng.shuffle(order)
        
        chosen = []
        for p in order:
            valid = True
            for i in range(len(chosen)):
                if not valid:
                    break
                a = chosen[i]
                da = _dist2(p, a)
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    d_pb = _dist2(p, b)
                    d_ab = _dist2(a, b)
                    if da == d_pb or da == d_ab or d_pb == d_ab:
                        valid = False
                        break
            if valid:
                chosen.append(p)
        
        # The set 'chosen' is maximal by construction.
        # Now reduce it to a minimal maximal set using SA-guided reduction.
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time < 2.0:
            # Not enough time for SA, just do quick greedy reduction
            reduced = _reduce_to_minimal(chosen, all_points, time_budget=1.0)
        else:
            # Allocate time for SA based on remaining budget
            sa_budget = min(3.0, remaining_time * 0.3)
            reduced = _reduce_to_minimal(chosen, all_points, time_budget=sa_budget)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        # Early termination
        if best_size <= 5:
            break
            
    if best_set is None:
        # Fallback: deterministic greedy
        chosen = []
        for p in all_points:
            valid = True
            for i in range(len(chosen)):
                if not valid: break
                a = chosen[i]
                da = _dist2(p, a)
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    d_pb = _dist2(p, b)
                    d_ab = _dist2(a, b)
                    if da == d_pb or da == d_ab or d_pb == d_ab:
                        valid = False
                        break
            if valid:
                chosen.append(p)
        return _reduce_to_minimal(chosen, all_points, time_budget=2.0)
        
    return best_set
