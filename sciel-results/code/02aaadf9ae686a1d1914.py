"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def solve_n(n, time_limit):
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

        # Try a few initializations
        inits = []
        # Mod 3 init
        inits.append([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples])
        # Random init
        inits.append([random.randint(0, 1) for _ in range(num_triples)])
        # Another random
        inits.append([random.randint(0, 1) for _ in range(num_triples)])
        
        local_deadline = time.time() + time_limit
        
        for init_colors in inits:
            if time_left() < 5.0 or time.time() > local_deadline:
                break
            
            colors = init_colors[:]
            
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
            min_T = 0.01
            max_steps = 500000
            
            for step in range(max_steps):
                if time_left() < 5.0 or time.time() > local_deadline:
                    break
                
                # Pick a random triple
                t_idx = random.randint(0, num_triples - 1)
                
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Calculate Delta E
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
                
                # Acceptance criterion
                if delta_E < 0:
                    accept = True
                else:
                    if T > min_T:
                        prob = math.exp(-delta_E / T)
                        accept = random.random() < prob
                    else:
                        accept = False
                
                if accept:
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
                
                if red_violations == 0 and blue_violations == 0:
                    return colors
                    
                T *= cooling

        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try increasing n
    for n in range(10, 35):
        if time_left() < 15.0:
            break
            
        # Allocate time: 10s for small n, more for larger?
        # Let's just give 15s per attempt, but capped by remaining time.
        time_limit = 15.0
        if time_left() < time_limit + 10.0:
            time_limit = time_left() - 10.0
            if time_limit < 5.0:
                break
                
        result = solve_n(n, time_limit)
        if result is not None:
            best_n = n
            best_colors = result
            # Continue to try larger n if time permits
        else:
            # If we fail, we might still try the next one?
            # Larger n is harder. But maybe we just got unlucky.
            # Let's try the next one.
            pass

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
