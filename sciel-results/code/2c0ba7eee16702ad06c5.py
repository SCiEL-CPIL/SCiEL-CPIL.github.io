import time
import math
import random

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def try_greedy_safety(order):
        S = []
        L = 1
        sums = {0}
        
        # We iterate through the provided order, but we dynamically decide
        # which one to pick next based on safety.
        # Actually, the 'order' list is just the pool of available candidates.
        # We will repeatedly scan the pool for the best candidate.
        
        available = set(order)
        
        while available:
            if time.time() - start_time > timeout - 5:
                break
                
            safe_candidates = []
            risky_candidates = []
            
            for x in available:
                if L % x != 0:
                    safe_candidates.append(x)
                else:
                    risky_candidates.append(x)
            
            # Prefer safe candidates
            chosen_x = None
            if safe_candidates:
                # Pick the largest safe candidate
                chosen_x = max(safe_candidates)
            else:
                # All are risky. Try to find one that doesn't collide.
                # Sort risky by some heuristic? Largest first?
                for x in sorted(risky_candidates, reverse=True):
                    L_new = math.lcm(L, x)
                    w_new = L_new // x
                    scale = L_new // L
                    
                    collision = False
                    if scale == 1:
                        for s in sums:
                            if (s + w_new) in sums:
                                collision = True
                                break
                    else:
                        # If L % x != 0, it would be safe.
                        # Here L % x == 0, so scale = 1?
                        # If L % x == 0, then L is a multiple of x.
                        # L_new = L. scale = 1.
                        # So the 'else' branch for scale != 1 is never taken
                        # in the risky case?
                        # Wait.
                        # If L % x == 0, then L is a multiple of x.
                        # lcm(L, x) = L.
                        # So L_new = L.
                        # scale = L_new / L = 1.
                        # So yes, for risky candidates, scale is always 1.
                        pass
                    
                    if not collision:
                        chosen_x = x
                        break
            
            if chosen_x is None:
                break
                
            # Add chosen_x
            available.remove(chosen_x)
            S.append(chosen_x)
            
            L_new = math.lcm(L, chosen_x)
            w_new = L_new // chosen_x
            scale = L_new // L
            
            # Update sums
            if scale == 1:
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w_new)
                sums = new_sums
            else:
                scaled_sums = {s * scale for s in sums}
                new_sums = set(scaled_sums)
                for s in scaled_sums:
                    new_sums.add(s + w_new)
                sums = new_sums
                
            L = L_new
            
        return S

    best_set = []
    
    # Try a few different random orders to break ties or explore different paths
    # Although the logic is deterministic given the set of available candidates,
    # the 'max' and 'sorted' calls might be affected if we randomize the initial list
    # but since we use max/sorted, the result is deterministic for a given set.
    # So we need to randomize the choice among safe candidates if we want diversity.
    # Let's modify try_greedy_safety to accept a random seed or choice function.
    
    def try_greedy_safety_random(order, random_func):
        S = []
        L = 1
        sums = {0}
        available = set(order)
        
        while available:
            if time.time() - start_time > timeout - 5:
                break
                
            safe_candidates = []
            risky_candidates = []
            
            for x in available:
                if L % x != 0:
                    safe_candidates.append(x)
                else:
                    risky_candidates.append(x)
            
            chosen_x = None
            if safe_candidates:
                # Randomly pick one from safe candidates?
                # Or pick the one with max "value"?
                # Let's try picking randomly to explore different LCMs.
                chosen_x = random_func(safe_candidates)
            else:
                for x in sorted(risky_candidates, reverse=True):
                    L_new = math.lcm(L, x)
                    w_new = L_new // x
                    scale = L_new // L
                    
                    collision = False
                    if scale == 1:
                        for s in sums:
                            if (s + w_new) in sums:
                                collision = True
                                break
                    # scale != 1 is impossible here as L%x==0 implies scale=1
                    
                    if not collision:
                        chosen_x = x
                        break
            
            if chosen_x is None:
                break
                
            available.remove(chosen_x)
            S.append(chosen_x)
            
            L_new = math.lcm(L, chosen_x)
            w_new = L_new // chosen_x
            scale = L_new // L
            
            if scale == 1:
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w_new)
                sums = new_sums
            else:
                scaled_sums = {s * scale for s in sums}
                new_sums = set(scaled_sums)
                for s in scaled_sums:
                    new_sums.add(s + w_new)
                sums = new_sums
                
            L = L_new
            
        return S

    # Run deterministic (max) version
    s_det = try_greedy_safety(candidates)
    if len(s_det) > len(best_set):
        best_set = s_det

    # Run random versions
    for _ in range(500):
        if time.time() - start_time > timeout - 5:
            break
        # Random choice among safe
        def random_choice(lst):
            return random.choice(lst)
        
        s_rand = try_greedy_safety_random(candidates, random_choice)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    if not best_set:
        best_set = [1]
        
    return best_set
