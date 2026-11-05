import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # We want to maximize n. Try 34 down to 10.
    # We allocate more time to higher n.
    time_budgets = {
        34: 80.0,
        33: 15.0,
        32: 5.0,
        31: 2.0,
        30: 1.0,
    }
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
        budget = time_budgets.get(n, 0.5)
        budget = min(budget, deadline - time.time())
        if budget < 0.5:
            continue
            
        try:
            result = solve_n(n, budget)
            if result is not None:
                return result
        except Exception:
            continue
            
    # Fallback to seed solution
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_n(n, time_limit):
    start = time.time()
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    M = len(triples)
    idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and 5-sets
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    # Map triple index to list of 4-set indices containing it
    triple_to_4 = [[] for _ in range(M)]
    for i4, s4 in enumerate(sets4):
        for j in range(4):
            t = tuple(sorted(s4[:j] + s4[j+1:]))
            triple_to_4[idx[t]].append(i4)
            
    # Map triple index to list of 5-set indices containing it
    triple_to_5 = [[] for _ in range(M)]
    for i5, s5 in enumerate(sets5):
        for j in range(5):
            t = tuple(sorted(s5[:j] + s5[j+1:]))
            triple_to_5[idx[t]].append(i5)
            
    # Precompute triples for each 4-set and 5-set
    s4_triples = []
    for s4 in sets4:
        ts = [idx[tuple(sorted(s4[:j] + s4[j+1:]))] for j in range(4)]
        s4_triples.append(ts)
        
    s5_triples = []
    for s5 in sets5:
        ts = [idx[t] for t in combinations(s5, 3)]
        s5_triples.append(ts)
        
    # Local search with restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            return None
            
        # Initialize random coloring
        # 0 for red, 1 for blue
        colors = np.random.randint(0, 2, M)
        
        # Initialize counts
        # red_count_4[i4] = number of red triples in 4-set i4
        # blue_count_5[i5] = number of blue triples in 5-set i5
        red_count_4 = np.zeros(N4, dtype=np.int16)
        blue_count_5 = np.zeros(N5, dtype=np.int16)
        
        # Calculate initial counts
        for i, t in enumerate(triples):
            c = colors[i]
            for i4 in triple_to_4[i]:
                if c == 0:
                    red_count_4[i4] += 1
            for i5 in triple_to_5[i]:
                if c == 1:
                    blue_count_5[i5] += 1
                    
        # Count violations
        # Violation for 4-set: red_count_4[i4] == 4
        # Violation for 5-set: blue_count_5[i5] == 10
        viol_4 = np.sum(red_count_4 == 4)
        viol_5 = np.sum(blue_count_5 == 10)
        total_viol = viol_4 + viol_5
        
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Greedy local search
        # We will try to find a flip that reduces total_viol.
        # To speed up, we can maintain a list of "candidate" flips.
        # But calculating delta for all M flips is O(M * (deg4 + deg5)).
        # For n=34, M ~ 6000. deg4 ~ 10, deg5 ~ 20.
        # 6000 * 30 = 180k operations per iteration.
        # This is fast enough to do many iterations.
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start > time_limit:
                return None
                
            if total_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            best_delta = 0
            best_idx = -1
            
            # Calculate delta for each triple flip
            # Delta = change in (viol_4 + viol_5)
            # If we flip triple t from Red to Blue:
            #   For each 4-set i4 containing t:
            #     red_count_4[i4] decreases by 1.
            #     If red_count_4[i4] was 4, viol_4 decreases by 1 (delta -1).
            #     If red_count_4[i4] was 3, viol_4 unchanged.
            #   For each 5-set i5 containing t:
            #     blue_count_5[i5] increases by 1.
            #     If blue_count_5[i5] was 9, viol_5 increases by 1 (delta +1).
            #     If blue_count_5[i5] was 10, viol_5 decreases by 1 (delta -1).
            
            # If we flip triple t from Blue to Red:
            #   For each 4-set i4 containing t:
            #     red_count_4[i4] increases by 1.
            #     If red_count_4[i4] was 3, viol_4 increases by 1 (delta +1).
            #     If red_count_4[i4] was 4, viol_4 decreases by 1 (delta -1).
            #   For each 5-set i5 containing t:
            #     blue_count_5[i5] decreases by 1.
            #     If blue_count_5[i5] was 10, viol_5 decreases by 1 (delta -1).
            #     If blue_count_5[i5] was 9, viol_5 increases by 1 (delta +1).
            
            # We want to minimize total_viol, so we want the most negative delta.
            
            # Optimization: We can just check if a flip reduces violations.
            # If multiple, pick one.
            
            # To avoid O(M) scan every time, we could maintain a list of "promising" edges.
            # But for now, let's just scan all edges. M=6000 is small.
            
            # We can also add a random component to escape local minima.
            # If no improving move, pick a random move.
            
            improved = False
            for t_idx in range(M):
                c = colors[t_idx]
                delta = 0
                
                if c == 0: # Red -> Blue
                    # Check 4-sets
                    for i4 in triple_to_4[t_idx]:
                        if red_count_4[i4] == 4:
                            delta -= 1
                    # Check 5-sets
                    for i5 in triple_to_5[t_idx]:
                        if blue_count_5[i5] == 9:
                            delta += 1
                        elif blue_count_5[i5] == 10:
                            delta -= 1
                else: # Blue -> Red
                    # Check 4-sets
                    for i4 in triple_to_4[t_idx]:
                        if red_count_4[i4] == 3:
                            delta += 1
                        elif red_count_4[i4] == 4:
                            delta -= 1
                    # Check 5-sets
                    for i5 in triple_to_5[t_idx]:
                        if blue_count_5[i5] == 10:
                            delta -= 1
                        elif blue_count_5[i5] == 9:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_idx = t_idx
                    improved = True
                elif delta == best_delta and best_idx == -1 and delta < 0:
                    best_idx = t_idx
                    improved = True
                    
            if best_idx == -1:
                # No improving move found. Pick a random move.
                best_idx = random.randint(0, M - 1)
                best_delta = 0 # We don't know the delta exactly without re-calculating, but we will update counts anyway.
                # Actually, we should calculate delta for the random move too to update total_viol correctly.
                # Or just update counts and recalculate total_viol? Recalculating is O(N4+N5) which is large.
                # Better to calculate delta for the random move.
                c = colors[best_idx]
                if c == 0:
                    for i4 in triple_to_4[best_idx]:
                        if red_count_4[i4] == 4: best_delta -= 1
                    for i5 in triple_to_5[best_idx]:
                        if blue_count_5[i5] == 9: best_delta += 1
                        elif blue_count_5[i5] == 10: best_delta -= 1
                else:
                    for i4 in triple_to_4[best_idx]:
                        if red_count_4[i4] == 3: best_delta += 1
                        elif red_count_4[i4] == 4: best_delta -= 1
                    for i5 in triple_to_5[best_idx]:
                        if blue_count_5[i5] == 10: best_delta -= 1
                        elif blue_count_5[i5] == 9: best_delta += 1
            
            # Apply flip
            t_idx = best_idx
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            total_viol += best_delta
            
            # Update counts
            for i4 in triple_to_4[t_idx]:
                if old_color == 0: # Red -> Blue
                    red_count_4[i4] -= 1
                else: # Blue -> Red
                    red_count_4[i4] += 1
                    
            for i5 in triple_to_5[t_idx]:
                if old_color == 1: # Blue -> Red
                    blue_count_5[i5] -= 1
                else: # Red -> Blue
                    blue_count_5[i5] += 1
                    
            # If we stuck in a local minimum, we might need to do a random walk for a few steps.
            # The random move above helps with that.
            
    return None
