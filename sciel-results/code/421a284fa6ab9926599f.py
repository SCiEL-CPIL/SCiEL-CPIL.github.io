import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Try to find a valid coloring for the largest n possible
    # We know n=34 is the target.
    
    def solve_for_n(n):
        # Build structures
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: sets of 4 vertices, 4 triples
        red_k4s = []
        for v4 in combinations(range(n), 4):
            k4_triples = []
            for t in combinations(v4, 3):
                k4_triples.append(triple_to_idx[t])
            red_k4s.append(k4_triples)
            
        # Blue K5s: sets of 5 vertices, 10 triples
        blue_k5s = []
        for v5 in combinations(range(n), 5):
            k5_triples = []
            for t in combinations(v5, 3):
                k5_triples.append(triple_to_idx[t])
            blue_k5s.append(k5_triples)
            
        # Inverse indices
        num_red_k4 = len(red_k4s)
        num_blue_k5 = len(blue_k5s)
        
        triple_to_red_k4s = [[] for _ in range(M)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                triple_to_red_k4s[t].append(i)
                
        triple_to_blue_k5s = [[] for _ in range(M)]
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                triple_to_blue_k5s[t].append(i)
                
        # State
        # colors: 0 (Red), 1 (Blue)
        # Use random init
        colors = [random.randint(0, 1) for _ in range(M)]
        
        # Initial violation check
        red_k4_viol = [False] * num_red_k4
        blue_k5_viol = [False] * num_blue_k5
        total_viol = 0
        
        for i, k4 in enumerate(red_k4s):
            if all(colors[t] == 0 for t in k4):
                red_k4_viol[i] = True
                total_viol += 1
                
        for i, k5 in enumerate(blue_k5s):
            if all(colors[t] == 1 for t in k5):
                blue_k5_viol[i] = True
                total_viol += 1
                
        if total_viol == 0:
            return colors
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        best_viol = total_viol
        best_colors = colors.copy()
        
        # If we find a valid coloring, we can return immediately
        # But we should try to improve if possible? No, valid is enough.
        
        while time.time() < deadline:
            # Pick random triple
            i = random.randint(0, M - 1)
            
            # Calculate Delta
            delta = 0
            
            if colors[i] == 0: # Flip Red -> Blue
                # Red K4s containing i break
                for k4_idx in triple_to_red_k4s[i]:
                    if red_k4_viol[k4_idx]:
                        delta -= 1
                        # It will become non-violation
                        # We don't update state yet, just delta
                        # But we need to know if it *stays* non-violation?
                        # Yes, flipping the edge makes it non-red-K4.
                        
                # Blue K5s containing i might complete
                for k5_idx in triple_to_blue_k5s[i]:
                    # Check if other 9 are Blue
                    # If it was already a violation, it's impossible (since i was Red)
                    # So it was not a violation.
                    # Check if it becomes one.
                    is_blue_k5 = True
                    for t in blue_k5s[k5_idx]:
                        if t == i:
                            continue
                        if colors[t] != 1:
                            is_blue_k5 = False
                            break
                    if is_blue_k5:
                        delta += 1
                        
            else: # Flip Blue -> Red
                # Blue K5s containing i break
                for k5_idx in triple_to_blue_k5s[i]:
                    if blue_k5_viol[k5_idx]:
                        delta -= 1
                        
                # Red K4s containing i might complete
                for k4_idx in triple_to_red_k4s[i]:
                    # Check if other 3 are Red
                    is_red_k4 = True
                    for t in red_k4s[k4_idx]:
                        if t == i:
                            continue
                        if colors[t] != 0:
                            is_red_k4 = False
                            break
                    if is_red_k4:
                        delta += 1
                        
            # Acceptance
            if delta < 0:
                accept = True
            else:
                # Simulated Annealing acceptance
                if T > 0.01:
                    accept = random.random() < np.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                # Apply flip
                colors[i] = 1 - colors[i]
                
                # Update state
                if colors[i] == 1: # Became Blue
                    for k4_idx in triple_to_red_k4s[i]:
                        if red_k4_viol[k4_idx]:
                            red_k4_viol[k4_idx] = False
                            total_viol -= 1
                            
                    for k5_idx in triple_to_blue_k5s[i]:
                        # Check if it became a violation
                        # We already checked in delta calc, but we need to update flag
                        # Re-check to be safe or store result? Re-check is cheap.
                        is_blue_k5 = True
                        for t in blue_k5s[k5_idx]:
                            if t == i:
                                continue
                            if colors[t] != 1:
                                is_blue_k5 = False
                                break
                        if is_blue_k5 and not blue_k5_viol[k5_idx]:
                            blue_k5_viol[k5_idx] = True
                            total_viol += 1
                            
                else: # Became Red
                    for k5_idx in triple_to_blue_k5s[i]:
                        if blue_k5_viol[k5_idx]:
                            blue_k5_viol[k5_idx] = False
                            total_viol -= 1
                            
                    for k4_idx in triple_to_red_k4s[i]:
                        is_red_k4 = True
                        for t in red_k4s[k4_idx]:
                            if t == i:
                                continue
                            if colors[t] != 0:
                                is_red_k4 = False
                                break
                        if is_red_k4 and not red_k4_viol[k4_idx]:
                            red_k4_viol[k4_idx] = True
                            total_viol += 1
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    
                if total_viol == 0:
                    return colors
                    
                T *= cooling
                
        return best_colors if best_viol == 0 else None

    # Try n=34
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        # Give each n a chance
        # If n is large, give more time?
        # Just try. If it returns None, try smaller n.
        # To avoid spending all time on n=34, we can limit iterations or time per n.
        # But the loop inside solve_for_n checks deadline.
        # So it will stop when deadline is reached.
        # This means if n=34 fails, we have no time for n=33.
        
        # Better: Allocate time budget.
        # 120s total.
        # Try n=34 for 60s.
        # Try n=33 for 30s.
        # Try n=32 for 20s.
        # Try n=31 for 10s.
        
        pass # Logic needs adjustment

    # Revised Strategy:
    # We will call solve_for_n with a specific time limit.
    
    def solve_for_n_limited(n, time_limit):
        start = time.time()
        # ... same as solve_for_n but with local deadline ...
        pass

    # Let's just run solve_for_n(34) first.
    # If it fails, we are out of time.
    # To be safe, we should try smaller n if 34 fails quickly?
    # No, if 34 fails after 110s, we have 10s left.
    # We can try n=33 with 5s.
    
    # Let's modify solve_for_n to accept a deadline.
    
    def solve(n, deadline):
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        red_k4s = []
        for v4 in combinations(range(n), 4):
            k4_triples = []
            for t in combinations(v4, 3):
                k4_triples.append(triple_to_idx[t])
            red_k4s.append(k4_triples)
            
        blue_k5s = []
        for v5 in combinations(range(n), 5):
            k5_triples = []
            for t in combinations(v5, 3):
                k5_triples.append(triple_to_idx[t])
            blue_k5s.append(k5_triples)
            
        num_red_k4 = len(red_k4s)
        num_blue_k5 = len(blue_k5s)
        
        triple_to_red_k4s = [[] for _ in range(M)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                triple_to_red_k4s[t].append(i)
                
        triple_to_blue_k5s = [[] for _ in range(M)]
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                triple_to_blue_k5s[t].append(i)
                
        # Random Init
        colors = [random.randint(0, 1) for _ in range(M)]
        
        red_k4_viol = [False] * num_red_k4
        blue_k5_viol = [False] * num_blue_k5
        total_viol = 0
        
        for i, k4 in enumerate(red_k4s):
            if all(colors[t] == 0 for t in k4):
                red_k4_viol[i] = True
                total_viol += 1
                
        for i, k5 in enumerate(blue_k5s):
            if all(colors[t] == 1 for t in k5):
                blue_k5_viol[i] = True
                total_viol += 1
                
        if total_viol == 0:
            return colors
            
        T = 1.0
        cooling = 0.999
        best_viol = total_viol
        best_colors = colors.copy()
        
        while time.time() < deadline:
            i = random.randint(0, M - 1)
            delta = 0
            
            if colors[i] == 0:
                for k4_idx in triple_to_red_k4s[i]:
                    if red_k4_viol[k4_idx]:
                        delta -= 1
                for k5_idx in triple_to_blue_k5s[i]:
                    is_blue_k5 = True
                    for t in blue_k5s[k5_idx]:
                        if t == i: continue
                        if colors[t] != 1:
                            is_blue_k5 = False
                            break
                    if is_blue_k5:
                        delta += 1
            else:
                for k5_idx in triple_to_blue_k5s[i]:
                    if blue_k5_viol[k5_idx]:
                        delta -= 1
                for k4_idx in triple_to_red_k4s[i]:
                    is_red_k4 = True
                    for t in red_k4s[k4_idx]:
                        if t == i: continue
                        if colors[t] != 0:
                            is_red_k4 = False
                            break
                    if is_red_k4:
                        delta += 1
                        
            if delta < 0:
                accept = True
            else:
                if T > 0.01:
                    accept = random.random() < np.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                colors[i] = 1 - colors[i]
                
                if colors[i] == 1:
                    for k4_idx in triple_to_red_k4s[i]:
                        if red_k4_viol[k4_idx]:
                            red_k4_viol[k4_idx] = False
                            total_viol -= 1
                    for k5_idx in triple_to_blue_k5s[i]:
                        is_blue_k5 = True
                        for t in blue_k5s[k5_idx]:
                            if t == i: continue
                            if colors[t] != 1:
                                is_blue_k5 = False
                                break
                        if is_blue_k5 and not blue_k5_viol[k5_idx]:
                            blue_k5_viol[k5_idx] = True
                            total_viol += 1
                else:
                    for k5_idx in triple_to_blue_k5s[i]:
                        if blue_k5_viol[k5_idx]:
                            blue_k5_viol[k5_idx] = False
                            total_viol -= 1
                    for k4_idx in triple_to_red_k4s[i]:
                        is_red_k4 = True
                        for t in red_k4s[k4_idx]:
                            if t == i: continue
                            if colors[t] != 0:
                                is_red_k4 = False
                                break
                        if is_red_k4 and not red_k4_viol[k4_idx]:
                            red_k4_viol[k4_idx] = True
                            total_viol += 1
                            
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                if total_viol == 0:
                    return colors
                T *= cooling
                
        return best_colors if best_viol == 0 else None

    # Time allocation
    # 110s total
    # Try 34: 50s
    # Try 33: 30s
    # Try 32: 20s
    # Try 31: 10s
    
    targets = [
        (34, 50),
        (33, 30),
        (32, 20),
        (31, 10)
    ]
    
    elapsed = 0
    for n, t_alloc in targets:
        if elapsed >= 110:
            break
        t_limit = min(t_alloc, 110 - elapsed)
        if t_limit < 1:
            break
            
        start_t = time.time()
        res = solve(n, start_t + t_limit)
        elapsed = time.time() - start_t
        
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback: n=1
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
