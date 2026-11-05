"""Strategy: Simulated Annealing with Critical Violation Prioritization for Hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (150s hard kill)
    
    # Target n=34 is the known published bound.
    # We will attempt to find a valid coloring for n=34.
    # If we fail, we fall back to n=33, 32, etc., using the same solver logic
    # but with less time or just returning the best valid found.
    
    def solve(n, time_limit):
        """
        Attempts to find a valid coloring for n vertices.
        Returns (n, colors) if valid, else (0, []).
        """
        if n < 4:
            return n, []
            
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_map = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 and K5 sets
        k4_sets = []
        for quad in itertools.combinations(range(n), 4):
            t_indices = [triple_map[t] for t in itertools.combinations(quad, 3)]
            k4_sets.append(t_indices)
            
        k5_sets = []
        for quint in itertools.combinations(range(n), 5):
            t_indices = [triple_map[t] for t in itertools.combinations(quint, 3)]
            k5_sets.append(t_indices)
            
        # Indices: which K4s/K5s contain a given triple
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, t_indices in enumerate(k4_sets):
            for t in t_indices:
                triple_to_k4s[t].append(i)
                
        for i, t_indices in enumerate(k5_sets):
            for t in t_indices:
                triple_to_k5s[t].append(i)
        
        # Initialize colors
        # Slight bias towards Red (0) for K4? No, symmetry is usually best for start.
        # But Blue K5 is "denser" in constraints (10 edges vs 4).
        # Let's try p=0.5.
        colors = [0 if random.random() < 0.5 else 1 for _ in range(num_triples)]
        
        # State tracking
        is_red_k4 = [False] * len(k4_sets)
        is_blue_k5 = [False] * len(k5_sets)
        
        # Initial scan
        red_count = 0
        blue_count = 0
        for i, t_indices in enumerate(k4_sets):
            if all(colors[x] == 0 for x in t_indices):
                is_red_k4[i] = True
                red_count += 1
        for i, t_indices in enumerate(k5_sets):
            if all(colors[x] == 1 for x in t_indices):
                is_blue_k5[i] = True
                blue_count += 1
                
        total_viol = red_count + blue_count
        best_viol = total_viol
        best_colors = colors[:]
        
        # Simulated Annealing Parameters
        # Start temperature high enough to accept bad moves, cool down.
        # We want to reach 0 violations.
        # Cost function: total_viol.
        # We accept a move if delta < 0 or with probability exp(-delta / T).
        
        # Dynamic time-based cooling
        # We don't know the number of steps in advance.
        # We can use a time-based temperature schedule.
        # T(t) = T0 * exp(-alpha * t)
        # Or simply iterate until time runs out, cooling T based on step count.
        
        T = 10.0
        cooling_rate = 0.9999 # Adjust based on expected steps
        
        steps = 0
        last_improve_step = 0
        stall_limit = 5000 # If no improvement for this many steps, restart or cool faster?
        
        while time.time() < deadline - 5: # Reserve 5s for final check
            if total_viol == 0:
                break
                
            # Select a triple to flip
            # Strategy: Pick a triple involved in a violation with some probability,
            # otherwise pick a random triple.
            # To find a triple in a violation quickly:
            # We can maintain a list of active triples, but that's expensive to update.
            # Alternative: Pick a random K4/K5 violation, then a random triple in it.
            
            # Let's pick a random violation type.
            # If red_count > 0 and blue_count > 0:
            #   Pick Red with prob red_count/(red_count+blue_count)
            # Else pick the non-zero one.
            
            if red_count == 0 and blue_count == 0:
                break
                
            if red_count > 0 and (blue_count == 0 or random.random() < red_count / (red_count + blue_count)):
                # Pick a random active Red K4
                # We don't have a list of active K4s.
                # We can pick a random K4 index and check if it's active.
                # This is inefficient if violations are sparse.
                # Let's maintain a list of active violation indices?
                # Updating it is O(1) per change if we use a set?
                # For now, let's try: Pick random t from 0..N.
                # If viol_count[t] > 0, good.
                # We need viol_count.
                
                pass # Placeholder, let's use the viol_count approach from before but optimized.
            
            # Actually, let's stick to a simpler selection that doesn't require complex data structures.
            # Pick a random triple t.
            # Calculate delta.
            # If delta < 0, accept.
            # If delta > 0, accept with prob exp(-delta/T).
            
            t = random.randint(0, num_triples - 1)
            
            # Calculate delta
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta = 0
            
            # Check K4s
            for k4_idx in triple_to_k4s[t]:
                t_indices = k4_sets[k4_idx]
                was_viol = is_red_k4[k4_idx]
                # If was_viol, flipping t (which must be 0) to 1 removes it. Delta -1.
                # If not was_viol, flipping t to 0 might create it if others are 0.
                # We need to check the other 3.
                if was_viol:
                    delta -= 1
                else:
                    # Check if other 3 are 0
                    # t_indices has 4 elements. One is t.
                    # We can just check all 4 against new_color?
                    # No, we need to know if it BECOMES a violation.
                    # It becomes a violation if all 4 are new_color (0).
                    # Since t becomes 0, we need others to be 0.
                    c0, c1, c2, c3 = t_indices
                    if c0 == t:
                        if colors[c1] == 0 and colors[c2] == 0 and colors[c3] == 0:
                            delta += 1
                    elif c1 == t:
                        if colors[c0] == 0 and colors[c2] == 0 and colors[c3] == 0:
                            delta += 1
                    elif c2 == t:
                        if colors[c0] == 0 and colors[c1] == 0 and colors[c3] == 0:
                            delta += 1
                    else:
                        if colors[c0] == 0 and colors[c1] == 0 and colors[c2] == 0:
                            delta += 1
            
            # Check K5s
            for k5_idx in triple_to_k5s[t]:
                t_indices = k5_sets[k5_idx]
                was_viol = is_blue_k5[k5_idx]
                if was_viol:
                    delta -= 1
                else:
                    # Becomes violation if all 5 are new_color (1).
                    # t becomes 1. Check others.
                    # 10 triples in K5. t is one of them.
                    # We need to check if the other 9 are 1? No, the other 9 TRIPLES.
                    # Wait, K5 has 10 triples.
                    # If t is flipped to 1, and the other 9 triples are already 1, it becomes a violation.
                    # Checking 9 triples is fast.
                    # Find the index of t in t_indices?
                    # t_indices is a list of 10 triple IDs.
                    # We can just iterate and skip t.
                    all_others_blue = True
                    for x in t_indices:
                        if x == t:
                            continue
                        if colors[x] != 1:
                            all_others_blue = False
                            break
                    if all_others_blue:
                        delta += 1
            
            # Acceptance
            accepted = False
            if delta <= 0:
                accepted = True
            else:
                # Boltzmann factor
                # Avoid overflow/underflow
                if delta < T * 10:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accepted = True
            
            if accepted:
                # Apply change
                colors[t] = new_color
                
                # Update state
                for k4_idx in triple_to_k4s[t]:
                    t_indices = k4_sets[k4_idx]
                    if is_red_k4[k4_idx]:
                        # It was viol, t was 0, now 1. Not viol anymore.
                        is_red_k4[k4_idx] = False
                        red_count -= 1
                    else:
                        # Check if it became viol
                        if all(colors[x] == 0 for x in t_indices):
                            is_red_k4[k4_idx] = True
                            red_count += 1
                            
                for k5_idx in triple_to_k5s[t]:
                    t_indices = k5_sets[k5_idx]
                    if is_blue_k5[k5_idx]:
                        # It was viol, t was 1, now 0. Not viol anymore.
                        is_blue_k5[k5_idx] = False
                        blue_count -= 1
                    else:
                        # Check if it became viol
                        if all(colors[x] == 1 for x in t_indices):
                            is_blue_k5[k5_idx] = True
                            blue_count += 1
                            
                total_viol = red_count + blue_count
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
                    last_improve_step = steps
                    # Optional: Reset T or cool slower on improvement?
                    # Standard SA cools regardless.
                    
                steps += 1
                
                # Cooling
                T *= cooling_rate
                if T < 0.01:
                    T = 0.01
            else:
                # Rejected, no state change
                steps += 1
                T *= cooling_rate
                if T < 0.01:
                    T = 0.01
            
            # Stall detection: If no improvement for long, maybe restart with new random seed?
            # Or just let it run.
            if steps - last_improve_step > 20000:
                # Restart logic:
                # If we are stuck, break and let the outer loop try again?
                # For simplicity, just continue. SA should eventually cool down.
                pass

        if best_viol == 0:
            return n, best_colors
        else:
            return 0, []

    # Main logic
    # Try n=34 first.
    # If it fails, try n=33, 32...
    # But solving n=34 takes most of the time.
    # If we don't find it in ~100s, we have ~10s left.
    # We can't solve n=33 in 10s reliably from scratch.
    # However, if we fail n=34, we can return a known valid small n?
    # The seed returns n=9.
    # We can try to find a valid n=20 or n=30 quickly?
    
    # Strategy:
    # 1. Spend 100s trying n=34.
    # 2. If success, return n=34.
    # 3. If fail, check time.
    #    If > 10s left, try n=33? (Unlikely to succeed from scratch).
    #    Better: Return a valid coloring for a smaller n that we can generate quickly.
    #    We can generate a valid n=9 coloring instantly.
    #    Can we generate a valid n=10, 11... quickly?
    #    Local search for n=10 is very fast.
    
    # Let's try to find the MAX n we can solve within the remaining time if 34 fails.
    # But we don't know the remaining time until we try 34.
    
    # Alternative:
    # Run solver for n=34 for 100s.
    # If it finds 0 viol, return 34.
    # Else, use remaining time to try n=33, 32...
    # But initializing n=33 takes time too.
    
    # Let's just try n=34. If it fails, return a safe fallback.
    # What is a safe fallback?
    # n=9 is safe.
    # Can we do better?
    # n=10: C(10,4)=210, C(10,5)=252. Very small.
    # n=20: C(20,4)=4845, C(20,5)=15504. Small.
    # n=30: C(30,4)=27405, C(30,5)=142506. Moderate.
    
    # Let's try to solve n=30 if n=34 fails and we have time?
    # No, let's keep it simple.
    # The goal is to beat 1.0 (n=9).
    # If we find n=34, score is 34/9 = 3.77.
    # If we don't, we should return the best valid n we can find quickly.
    
    # Let's allocate 105s to n=34.
    t_start_34 = time.time()
    n_found, colors_found = solve(34, time.time() + 105)
    
    if n_found > 0:
        return {'n': n_found, 'colors': colors_found}
        
    # Fallback
    # Time remaining: 115 - (time.time() - start_time)
    time_elapsed = time.time() - start_time
    time_left = 115 - time_elapsed
    
    # Try n=33?
    if time_left > 15:
        n_found, colors_found = solve(33, time.time() + min(15, time_left - 5))
        if n_found > 0:
            return {'n': n_found, 'colors': colors_found}
            
    # Try n=30?
    time_elapsed = time.time() - start_time
    time_left = 115 - time_elapsed
    if time_left > 10:
        n_found, colors_found = solve(30, time.time() + min(10, time_left - 5))
        if n_found > 0:
            return {'n': n_found, 'colors': colors_found}
            
    # Try n=20?
    time_elapsed = time.time() - start_time
    time_left = 115 - time_elapsed
    if time_left > 5:
        n_found, colors_found = solve(20, time.time() + min(5, time_left - 2))
        if n_found > 0:
            return {'n': n_found, 'colors': colors_found}
            
    # Final fallback: n=9
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
