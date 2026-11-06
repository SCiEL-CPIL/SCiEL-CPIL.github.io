import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We want to maximize n. Let's try from high to low.
    # Known lower bound is 34. Let's try 34, 33, 32...
    # If we find a valid coloring for n, return it.
    
    best_n = 0
    best_colors = None
    
    # Precompute triples for max n=34
    max_n = 34
    triples = list(combinations(range(max_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s for max_n
    # K4s: all 4-subsets
    k4s = list(combinations(range(max_n), 4))
    k5s = list(combinations(range(max_n), 5))
    
    # Map each triple to the list of K4 indices and K5 indices it belongs to
    # This is for fast update.
    # But for n < max_n, we only care about subsets within range(n).
    # It's easier to just re-evaluate for a given n.
    
    def try_solve(n, time_limit=10.0):
        """
        Attempts to find a valid coloring for n vertices.
        Returns (colors, valid)
        """
        if time.time() > deadline:
            return None, False
            
        # Get triples for this n
        triples_n = [t for t in triples if t[2] < n]
        num_t = len(triples_n)
        # Map triple to index in triples_n
        t_idx = {t: i for i, t in enumerate(triples_n)}
        
        # Initialize colors randomly
        colors = [random.randint(0, 1) for _ in range(num_t)]
        
        # Precompute constraints for this n
        # K4s: subsets of size 4 from range(n)
        k4s_n = [t for t in k4s if t[3] < n]
        # K5s: subsets of size 5 from range(n)
        k5s_n = [t for t in k5s if t[4] < n]
        
        # For each K4, store the indices of its 4 triples
        k4_triples = []
        for k4 in k4s_n:
            t_set = [t_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_set)
            
        # For each K5, store the indices of its 10 triples
        k5_triples = []
        for k5 in k5s_n:
            t_set = [t_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_set)
            
        # Map triple index to list of (constraint_type, constraint_idx)
        # constraint_type 0 for K4, 1 for K5
        t_to_constraints = [[] for _ in range(num_t)]
        for i, t_set in enumerate(k4_triples):
            for t in t_set:
                t_to_constraints[t].append((0, i))
        for j, t_set in enumerate(k5_triples):
            for t in t_set:
                t_to_constraints[t].append((1, j))
        
        # State: is_red_k4, is_blue_k5
        is_red_k4 = [False] * len(k4s_n)
        is_blue_k5 = [False] * len(k5s_n)
        bad_red = 0
        bad_blue = 0
        
        # Initialize state
        for i, t_set in enumerate(k4_triples):
            if all(colors[t] == 0 for t in t_set):
                is_red_k4[i] = True
                bad_red += 1
        for j, t_set in enumerate(k5_triples):
            if all(colors[t] == 1 for t in t_set):
                is_blue_k5[j] = True
                bad_blue += 1
        
        total_bad = bad_red + bad_blue
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        current_time = time.time()
        
        while time.time() < current_time + time_limit and total_bad > 0:
            if time.time() > deadline:
                break
                
            # Pick a random triple
            t = random.randint(0, num_t - 1)
            
            # Calculate delta cost
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_bad = 0
            
            # Check K4s containing t
            for c_type, c_idx in t_to_constraints[t]:
                if c_type == 0:
                    # K4
                    t_set = k4_triples[c_idx]
                    # Check if K4 is currently red
                    was_red = is_red_k4[c_idx]
                    # Check if K4 will be red
                    # All other 3 must be 0, and new_color must be 0
                    others_red = all(colors[u] == 0 for u in t_set if u != t)
                    will_be_red = (new_color == 0) and others_red
                    
                    if was_red and not will_be_red:
                        delta_bad -= 1
                    elif not was_red and will_be_red:
                        delta_bad += 1
                else:
                    # K5
                    t_set = k5_triples[c_idx]
                    was_blue = is_blue_k5[c_idx]
                    others_blue = all(colors[u] == 1 for u in t_set if u != t)
                    will_be_blue = (new_color == 1) and others_blue
                    
                    if was_blue and not will_be_blue:
                        delta_bad -= 1
                    elif not was_blue and will_be_blue:
                        delta_bad += 1
            
            # Accept move
            if delta_bad < 0 or (random.random() < pow(2.718, -delta_bad / max(T, 0.001))):
                # Apply move
                colors[t] = new_color
                bad_red += sum(1 for ct, ci in t_to_constraints[t] if ct==0 and ((is_red_k4[ci] and new_color!=0) or (not is_red_k4[ci] and new_color==0 and all(colors[u]==0 for u in k4_triples[ci] if u!=t))))
                # This update logic is getting messy and slow.
                # Let's just recompute the affected constraints' states.
                
                # Recompute affected K4s
                for ct, ci in t_to_constraints[t]:
                    if ct == 0:
                        t_set = k4_triples[ci]
                        is_red_k4[ci] = all(colors[u] == 0 for u in t_set)
                    else:
                        t_set = k5_triples[ci]
                        is_blue_k5[ci] = all(colors[u] == 1 for u in t_set)
                
                # Recalculate total_bad? Or just update counters?
                # Updating counters is tricky. Let's just recalculate total_bad from scratch?
                # No, that's O(num_constraints).
                
                # Let's stick to the delta calculation but fix the apply logic.
                # We already calculated delta_bad.
                # We just need to update the state arrays.
                
                # Update state for affected constraints
                for ct, ci in t_to_constraints[t]:
                    if ct == 0:
                        t_set = k4_triples[ci]
                        is_red_k4[ci] = all(colors[u] == 0 for u in t_set)
                    else:
                        t_set = k5_triples[ci]
                        is_blue_k5[ci] = all(colors[u] == 1 for u in t_set)
                
                total_bad += delta_bad
                T *= cooling
            else:
                T *= cooling
                
            if T < min_T:
                T = min_T
                
        if total_bad == 0:
            return colors, True
        return None, False

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
        colors, valid = try_solve(n, time_limit=15.0)
        if valid:
            return {'n': n, 'colors': colors}
            
    # Fallback: return seed result
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
