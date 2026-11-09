"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    def time_left():
        return deadline - time.time()

    def solve_n(n, time_budget=5.0):
        if n < 4:
            return None
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        # Initialize colors: 0=Red, 1=Blue
        # Random initialization
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_count_k4 = [0] * len(k4s)
        blue_count_k5 = [0] * len(k5s)
        
        for i, ts in enumerate(k4_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 0:
                    cnt += 1
            red_count_k4[i] = cnt
            
        for i, ts in enumerate(k5_triples):
            cnt = 0
            for t_idx in ts:
                if colors[t_idx] == 1:
                    cnt += 1
            blue_count_k5[i] = cnt
            
        red_violations = sum(1 for c in red_count_k4 if c == 4)
        blue_violations = sum(1 for c in blue_count_k5 if c == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        max_steps = 1000000
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if red_violations == 0 and blue_violations == 0:
                return colors
                
            # Pick a triple to flip
            # Bias towards violations if possible, else random
            # For simplicity and speed, just pick random. 
            # The delta calculation will guide the acceptance.
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4[t_idx]:
                if old_color == 0: # Was Red, becoming Blue
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 4 and new_cnt == 3:
                        delta_red -= 1
                    elif old_cnt == 3 and new_cnt == 4:
                        delta_red += 1
                    red_count_k4[k4_idx] = new_cnt
                else: # Was Blue, becoming Red
                    old_cnt = red_count_k4[k4_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 3 and new_cnt == 4:
                        delta_red += 1
                    elif old_cnt == 4 and new_cnt == 3:
                        delta_red -= 1
                    red_count_k4[k4_idx] = new_cnt
                    
            # Update K5 counts
            for k5_idx in triple_to_k5[t_idx]:
                if old_color == 1: # Was Blue, becoming Red
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt - 1
                    if old_cnt == 10 and new_cnt == 9:
                        delta_blue -= 1
                    elif old_cnt == 9 and new_cnt == 10:
                        delta_blue += 1
                    blue_count_k5[k5_idx] = new_cnt
                else: # Was Red, becoming Blue
                    old_cnt = blue_count_k5[k5_idx]
                    new_cnt = old_cnt + 1
                    if old_cnt == 9 and new_cnt == 10:
                        delta_blue += 1
                    elif old_cnt == 10 and new_cnt == 9:
                        delta_blue -= 1
                    blue_count_k5[k5_idx] = new_cnt
                    
            delta_E = delta_red + delta_blue
            
            if delta_E < 0:
                # Accept
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
            else:
                # Accept with probability
                if T > min_T:
                    # Avoid overflow in exp
                    if delta_E / T < 100:
                        prob = math.exp(-delta_E / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        red_violations += delta_red
                        blue_violations += delta_blue
                    else:
                        # Rollback
                        for k4_idx in triple_to_k4[t_idx]:
                            if old_color == 0:
                                red_count_k4[k4_idx] += 1
                            else:
                                red_count_k4[k4_idx] -= 1
                        for k5_idx in triple_to_k5[t_idx]:
                            if old_color == 1:
                                blue_count_k5[k5_idx] += 1
                            else:
                                blue_count_k5[k5_idx] -= 1
                else:
                    # Rollback
                    for k4_idx in triple_to_k4[t_idx]:
                        if old_color == 0:
                            red_count_k4[k4_idx] += 1
                        else:
                            red_count_k4[k4_idx] -= 1
                    for k5_idx in triple_to_k5[t_idx]:
                        if old_color == 1:
                            blue_count_k5[k5_idx] += 1
                        else:
                            blue_count_k5[k5_idx] -= 1
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try to solve for increasing n
    # Start from a reasonable lower bound to save time
    # n=10 is easy. Let's start from 10.
    for target_n in range(10, 35):
        if time_left() < 5.0:
            break
            
        # Allocate more time for larger n? 
        # Or fixed time? 
        # Let's use a fixed budget of 5s per n. 
        # If n is large, it might fail, and we move on.
        # But if n is small, it will succeed quickly.
        
        # We can also try multiple random restarts if time permits
        result = solve_n(target_n, time_budget=5.0)
        
        if result is not None:
            best_n = target_n
            best_colors = result
            # Continue to try n+1
        else:
            # If we failed, it might be due to bad luck or local minimum.
            # We could try again, but to ensure progress, we might just move on 
            # or retry once.
            # Given the time constraint, let's just move on to the next n 
            # to see if a larger n is somehow easier (unlikely) or 
            # just stop if we think we've hit the limit.
            # Actually, if we fail for n, we likely fail for n+1.
            # But maybe a different random seed works.
            # Let's try one more time with a different seed if we have time.
            if time_left() > 10.0:
                result = solve_n(target_n, time_budget=5.0)
                if result is not None:
                    best_n = target_n
                    best_colors = result
                    # Continue
                else:
                    # Failed twice, assume this n is too hard or impossible 
                    # in our time budget. Break.
                    break
            else:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
