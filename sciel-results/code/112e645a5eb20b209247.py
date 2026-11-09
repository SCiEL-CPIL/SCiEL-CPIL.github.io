import time
import random
from itertools import combinations
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Target n=34
    # If we fail, we might try smaller n, but the task asks to maximize n.
    # The prompt says "Maximize n". The seed returned 9.
    # We should try to get 34.
    
    n = 34
    
    # Precomputation
    # This is done once per n. Since we only try n=34, we do it once.
    # If it fails, we could try n=33, etc., but let's focus on 34 first.
    
    triples = list(combinations(range(n), 3))
    N = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # 4-sets
    four_sets = list(combinations(range(n), 4))
    N4 = len(four_sets)
    four_set_to_idx = {s: i for i, s in enumerate(four_sets)}
    
    # 5-sets
    five_sets = list(combinations(range(n), 5))
    N5 = len(five_sets)
    five_set_to_idx = {s: i for i, s in enumerate(five_sets)}
    
    # Incidence: triple -> list of 4-set indices
    triple_to_fours = [[] for _ in range(N)]
    for s in four_sets:
        idx_s = four_set_to_idx[s]
        # 4 triples in this 4-set
        for i in range(4):
            t = s[:i] + s[i+1:]
            idx_t = triple_to_idx[t]
            triple_to_fours[idx_t].append(idx_s)
            
    # Incidence: triple -> list of 5-set indices
    triple_to_fives = [[] for _ in range(N)]
    for s in five_sets:
        idx_s = five_set_to_idx[s]
        # 10 triples in this 5-set
        for t in combinations(s, 3):
            idx_t = triple_to_idx[t]
            triple_to_fives[idx_t].append(idx_s)
            
    def count_violations(colors):
        """Calculate initial counts and total violations."""
        red_4_counts = [0] * N4
        blue_5_counts = [0] * N5
        
        for idx_t, t in enumerate(triples):
            c = colors[idx_t]
            for idx_s in triple_to_fours[idx_t]:
                if c == 0: # red
                    red_4_counts[idx_s] += 1
            for idx_s in triple_to_fives[idx_t]:
                if c == 1: # blue
                    blue_5_counts[idx_s] += 1
                    
        num_red_k4 = sum(1 for c in red_4_counts if c == 4)
        num_blue_k5 = sum(1 for c in blue_5_counts if c == 10)
        return red_4_counts, blue_5_counts, num_red_k4 + num_blue_k5

    # Try multiple restarts
    max_restarts = 50
    steps_per_restart = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        colors = [random.randint(0, 1) for _ in range(N)]
        red_4_counts, blue_5_counts, total_violations = count_violations(colors)
        
        if total_violations == 0:
            return {'n': n, 'colors': colors}
            
        best_violations = total_violations
        best_colors = colors.copy()
        
        stagnant_steps = 0
        max_stagnant = 5000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            if total_violations == 0:
                return {'n': n, 'colors': colors}
                
            # If we improve, reset stagnant counter
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            # If stuck for too long, restart the local search phase (kick)
            # Or just accept a bad move
            if stagnant_steps > max_stagnant:
                # Kick: flip a random triple
                idx_t = random.randint(0, N - 1)
                old_c = colors[idx_t]
                new_c = 1 - old_c
                
                # Update counts without checking delta, just apply
                colors[idx_t] = new_c
                for idx_s in triple_to_fours[idx_t]:
                    if old_c == 0:
                        red_4_counts[idx_s] -= 1
                    else:
                        red_4_counts[idx_s] += 1
                for idx_s in triple_to_fives[idx_t]:
                    if old_c == 1:
                        blue_5_counts[idx_s] -= 1
                    else:
                        blue_5_counts[idx_s] += 1
                
                # Recalculate total violations? 
                # It's expensive to recalculate from scratch.
                # But we don't have the delta.
                # Let's just recalculate total_violations from counts.
                # sum(1 for c in red_4_counts if c == 4) + sum(1 for c in blue_5_counts if c == 10)
                # This is O(N4 + N5) which is ~ 46k + 278k = 324k ops.
                # Acceptable for a kick every 5000 steps.
                total_violations = sum(1 for c in red_4_counts if c == 4) + sum(1 for c in blue_5_counts if c == 10)
                stagnant_steps = 0
                continue

            # Normal step: pick a random triple
            # To be more effective, we could pick from triples involved in violations.
            # But maintaining that list is complex. Random is often sufficient with enough steps.
            idx_t = random.randint(0, N - 1)
            
            old_c = colors[idx_t]
            new_c = 1 - old_c
            
            delta = 0
            
            # Update 4-sets
            for idx_s in triple_to_fours[idx_t]:
                if old_c == 0: # was red, becomes blue
                    # red count decreases by 1
                    if red_4_counts[idx_s] == 4:
                        delta -= 1
                    elif red_4_counts[idx_s] == 3:
                        delta += 1
                else: # was blue, becomes red
                    # red count increases by 1
                    if red_4_counts[idx_s] == 3:
                        delta += 1
                    elif red_4_counts[idx_s] == 4:
                        delta -= 1
                        
            # Update 5-sets
            for idx_s in triple_to_fives[idx_t]:
                if old_c == 1: # was blue, becomes red
                    # blue count decreases by 1
                    if blue_5_counts[idx_s] == 10:
                        delta -= 1
                    elif blue_5_counts[idx_s] == 9:
                        delta += 1
                else: # was red, becomes blue
                    # blue count increases by 1
                    if blue_5_counts[idx_s] == 9:
                        delta += 1
                    elif blue_5_counts[idx_s] == 10:
                        delta -= 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                # Simulated annealing style: accept bad moves with low prob
                # Or just reject.
                # Let's try a small probability to escape local minima
                if random.random() < 0.01:
                    accept = True
            
            if accept:
                # Apply flip
                colors[idx_t] = new_c
                
                # Update counts
                for idx_s in triple_to_fours[idx_t]:
                    if old_c == 0:
                        red_4_counts[idx_s] -= 1
                    else:
                        red_4_counts[idx_s] += 1
                        
                for idx_s in triple_to_fives[idx_t]:
                    if old_c == 1:
                        blue_5_counts[idx_s] -= 1
                    else:
                        blue_5_counts[idx_s] += 1
                        
                total_violations += delta
                
                if total_violations < 0:
                    total_violations = 0 # Should not happen if logic is correct

    # If we didn't find a valid coloring for n=34, try smaller n?
    # The task says "Maximize n". If we can't get 34, we should return the best we found.
    # But the current best is likely 0 (invalid) or some lower n if we implemented fallback.
    # The seed program returned n=9.
    # Let's add a fallback to n=9 if 34 fails, just to be safe and score > 1.
    # Actually, if we return an invalid coloring for n=34, the score is 0.
    # So we MUST return a valid coloring.
    
    # Let's try n=33, 32... down to 9 if 34 fails.
    # But precomputation is heavy.
    # Let's just return a known valid small coloring if 34 fails.
    # A simple construction for small n:
    # For n=9, the seed provided a valid one.
    
    # However, it is highly likely we will find n=34 with this local search.
    # If not, we risk scoring 0.
    # Let's implement a quick check for n=9 as fallback.
    
    # To save time, if we are close to deadline and haven't found 34, 
    # we should return a valid smaller n.
    
    # Let's just return the seed's n=9 solution if we fail to find 34.
    n_fallback = 9
    return {'n': n_fallback, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]}
