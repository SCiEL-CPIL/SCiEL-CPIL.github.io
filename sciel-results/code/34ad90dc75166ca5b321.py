import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    # Safe fallback: n=3, all red (or blue). No K4 or K5 possible.
    def get_safe_seed():
        n = 3
        triples = list(combinations(range(n), 3))
        return {'n': n, 'colors': [0] * len(triples)}
    
    best_solution = get_safe_seed()
    
    # We will focus on n=34. If it fails, we return the safe seed.
    # Trying to iterate down from 34 is risky if 34 takes all time.
    # We will allocate 90% of time to n=34.
    
    n = 34
    if time.time() < deadline - 10.0:
        # Precomputation for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate all K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # For each triple, list of K4s and K5s containing it
        k4_containing = [[] for _ in range(num_triples)]
        k5_containing = [[] for _ in range(num_triples)]
        
        # Map K4s to their triples
        k4_triples = []
        for i, k4 in enumerate(k4s):
            tris = []
            for j in range(4):
                t = tuple(k4[k] for k in range(4) if k != j)
                tris.append(triple_to_idx[t])
            k4_triples.append(tris)
            for t_idx in tris:
                k4_containing[t_idx].append(i)
                
        # Map K5s to their triples
        k5_triples = []
        for i, k5 in enumerate(k5s):
            tris = []
            for t in combinations(k5, 3):
                tris.append(triple_to_idx[t])
            k5_triples.append(tris)
            for t_idx in tris:
                k5_containing[t_idx].append(i)
        
        # Initial random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_k4_count = [0] * len(k4s)
        blue_k5_count = [0] * len(k5s)
        
        for i, tris in enumerate(k4_triples):
            cnt = 0
            for t_idx in tris:
                if colors[t_idx] == 0:
                    cnt += 1
            red_k4_count[i] = cnt
            
        for i, tris in enumerate(k5_triples):
            cnt = 0
            for t_idx in tris:
                if colors[t_idx] == 1:
                    cnt += 1
            blue_k5_count[i] = cnt
            
        # Calculate initial cost
        cost = sum(1 for c in red_k4_count if c == 4) + sum(1 for c in blue_k5_count if c == 10)
        
        # Simulated Annealing
        T_start = 1.0
        T_end = 0.001
        alpha = 0.9995 # Cooling rate
        T = T_start
        
        # We run until deadline - 5s
        while time.time() < deadline - 5.0:
            if cost == 0:
                best_solution = {'n': n, 'colors': colors[:]}
                return best_solution
            
            # Pick a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta cost WITHOUT modifying state
            delta = 0
            
            # Update K4s (virtual)
            for k4_idx in k4_containing[t_idx]:
                old_cnt = red_k4_count[k4_idx]
                if old_color == 0: # Was Red, becomes Blue
                    new_cnt = old_cnt - 1
                else: # Was Blue, becomes Red
                    new_cnt = old_cnt + 1
                
                if old_cnt == 4 and new_cnt != 4:
                    delta -= 1
                elif old_cnt != 4 and new_cnt == 4:
                    delta += 1
            
            # Update K5s (virtual)
            for k5_idx in k5_containing[t_idx]:
                old_cnt = blue_k5_count[k5_idx]
                if old_color == 1: # Was Blue, becomes Red
                    new_cnt = old_cnt - 1
                else: # Was Red, becomes Blue
                    new_cnt = old_cnt + 1
                
                if old_cnt == 10 and new_cnt != 10:
                    delta -= 1
                elif old_cnt != 10 and new_cnt == 10:
                    delta += 1
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    prob = pow(2.71828, -delta / T)
                    accept = (random.random() < prob)
                else:
                    accept = (delta == 0 and random.random() < 0.5)
            
            if accept:
                # Apply change to state
                colors[t_idx] = new_color
                cost += delta
                
                # Update counts in state
                for k4_idx in k4_containing[t_idx]:
                    if old_color == 0:
                        red_k4_count[k4_idx] -= 1
                    else:
                        red_k4_count[k4_idx] += 1
                        
                for k5_idx in k5_containing[t_idx]:
                    if old_color == 1:
                        blue_k5_count[k5_idx] -= 1
                    else:
                        blue_k5_count[k5_idx] += 1
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end

    return best_solution
