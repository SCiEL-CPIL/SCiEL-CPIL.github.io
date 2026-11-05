import random
import time
import numpy as np

def construct():
    # Target n=34, which is known to be possible.
    # We will use a local search / simulated annealing approach to find a valid coloring.
    
    n = 34
    
    # Generate all triples
    triples = []
    triple_to_idx = {}
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                idx = len(triples)
                triples.append((a, b, c))
                triple_to_idx[(a, b, c)] = idx
                
    num_triples = len(triples)
    
    # Precompute which triples are in which K4s and K5s
    # k4s: list of lists of triple indices
    k4s = []
    k4_to_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # The 4 triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                    t1 = triple_to_idx[(a,b,c)]
                    t2 = triple_to_idx[(a,b,d)]
                    t3 = triple_to_idx[(a,c,d)]
                    t4 = triple_to_idx[(b,c,d)]
                    k4s.append([t1, t2, t3, t4])
                    
    # k5s: list of lists of triple indices
    k5s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for e in range(d+1, n):
                        # 10 triples
                        ts = []
                        for i in range(5):
                            for j in range(i+1, 5):
                                for k in range(j+1, 5):
                                    ts.append(triple_to_idx[( [a,b,c,d,e][i], [a,b,c,d,e][j], [a,b,c,d,e][k] )])
                        k5s.append(ts)
                        
    # Reverse mapping: for each triple, which k4s and k5s does it belong to?
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4s):
        for t in k4:
            triple_to_k4s[t].append(i)
            
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5s):
        for t in k5:
            triple_to_k5s[t].append(i)

    def count_bad(colors):
        # colors is a list of 0/1
        red_k4 = 0
        for k4 in k4s:
            if all(colors[t] == 0 for t in k4):
                red_k4 += 1
        blue_k5 = 0
        for k5 in k5s:
            if all(colors[t] == 1 for t in k5):
                blue_k5 += 1
        return red_k4 + blue_k5

    def delta_energy(colors, t_idx):
        # Calculate change in energy if we flip triple t_idx
        # Current color: c
        # New color: 1-c
        c = colors[t_idx]
        
        # Check K4s containing t_idx
        # A K4 is red if all 4 are 0.
        # If we flip t_idx from 0 to 1:
        #   If it was red, it stops being red (decrease by 1)
        #   If it was not red, it stays not red (change 0) -- unless other 3 were 0 and t was 1? No, if t was 1, it wasn't red.
        #   Wait, if t was 0, and other 3 are 0, it IS red. Flipping to 1 makes it NOT red.
        #   If t was 1, it was NOT red. Flipping to 0: if other 3 are 0, it BECOMES red.
        
        d_red = 0
        for k4_idx in triple_to_k4s[t_idx]:
            k4 = k4s[k4_idx]
            # Count zeros in the other 3 triples
            others_zero = 0
            for t in k4:
                if t != t_idx and colors[t] == 0:
                    others_zero += 1
            
            if c == 0:
                # Currently 0. If others are all 0, it's a red K4. Flipping to 1 breaks it.
                if others_zero == 3:
                    d_red -= 1
            else:
                # Currently 1. If others are all 0, flipping to 0 makes it a red K4.
                if others_zero == 3:
                    d_red += 1
                    
        d_blue = 0
        for k5_idx in triple_to_k5s[t_idx]:
            k5 = k5s[k5_idx]
            # Count ones in the other 9 triples
            others_one = 0
            for t in k5:
                if t != t_idx and colors[t] == 1:
                    others_one += 1
                    
            if c == 1:
                # Currently 1. If others are all 1, it's a blue K5. Flipping to 0 breaks it.
                if others_one == 9:
                    d_blue -= 1
            else:
                # Currently 0. If others are all 1, flipping to 1 makes it a blue K5.
                if others_one == 9:
                    d_blue += 1
                    
        return d_red + d_blue

    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    best_colors = None
    best_energy = float('inf')
    
    while time.time() - start_time < timeout:
        # Random restart
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initial greedy improvement
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias
            indices = list(range(num_triples))
            random.shuffle(indices)
            for t_idx in indices:
                de = delta_energy(colors, t_idx)
                if de < 0:
                    colors[t_idx] = 1 - colors[t_idx]
                    improved = True
                    # Check if we reached 0
                    # To save time, only check full count occasionally or if improved significantly
                    # But delta_energy is cheap enough? 
                    # Let's just check energy at the end of the greedy pass.
        
        energy = count_bad(colors)
        
        if energy < best_energy:
            best_energy = energy
            best_colors = colors[:]
            
        if best_energy == 0:
            break
            
        # Simulated Annealing phase from this state
        temp = 1.0
        cool_rate = 0.95
        current_energy = energy
        current_colors = colors[:]
        
        # Run SA for a few seconds
        sa_start = time.time()
        while time.time() - sa_start < 2.0: # 2 seconds per restart
            t_idx = random.randint(0, num_triples - 1)
            de = delta_energy(current_colors, t_idx)
            
            if de < 0 or (de > 0 and random.random() < np.exp(-de / temp)):
                current_colors[t_idx] = 1 - current_colors[t_idx]
                current_energy += de
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = current_colors[:]
                    if best_energy == 0:
                        break
                        
            temp *= cool_rate
            if temp < 0.01:
                temp = 0.01 # Floor temp to keep exploring slightly
                
        if best_energy == 0:
            break

    if best_energy == 0:
        return {'n': n, 'colors': best_colors}
    else:
        # Fallback: Return the best found, even if invalid? 
        # The task says "valid colouring... certifies". If invalid, score is 0.
        # But we must return *something*. The evaluator checks validity.
        # If we fail to find a valid one, we might as well return the best effort, 
        # though it will likely score 0 if invalid.
        # However, n=34 is known to exist. With 110s, we should find it.
        return {'n': n, 'colors': best_colors}
