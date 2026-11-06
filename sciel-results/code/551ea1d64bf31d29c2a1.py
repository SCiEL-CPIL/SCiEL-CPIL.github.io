import random
import time
import itertools
import math
import numpy as np

# Precomputation
N = 16
EDGES = []
EDGE_IDX = {}
for i in range(N):
    for j in range(i+1, N):
        EDGE_IDX[(i,j)] = len(EDGES)
        EDGES.append((i,j))
NUM_EDGES = len(EDGES) # 120

K4S = []
K4_EDGES = [] # List of edge indices for each K4
for combo in itertools.combinations(range(N), 4):
    a, b, c, d = combo
    edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
    edge_idxs = [EDGE_IDX[e] for e in edges]
    K4_EDGES.append(edge_idxs)
    K4S.append(combo)
NUM_K4 = len(K4S) # 1820

EDGE_TO_K4S = [[] for _ in range(NUM_EDGES)]
for k4_idx, e_idxs in enumerate(K4_EDGES):
    for e_idx in e_idxs:
        EDGE_TO_K4S[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try C from 13 to 15
    for C in [13, 14, 15]:
        # Try multiple restarts
        for restart in range(300):
            if time.time() > deadline - 5:
                break
            
            # Smart initialization: start from (i+j)%17 base and remap to C colors
            if restart == 0:
                # Use the known 17-color construction as seed, then remap
                colors = [((i + j) % 17) % C for i in range(16) for j in range(i + 1, 16)]
            elif restart % 3 == 1:
                # Perturbed version of the base
                colors = [((i + j) % 17) % C for i in range(16) for j in range(i + 1, 16)]
                for _ in range(30):
                    e_idx = random.randint(0, NUM_EDGES - 1)
                    colors[e_idx] = random.randint(0, C - 1)
            elif restart % 3 == 2:
                # Structured perturbation: swap color classes
                colors = [((i + j) % 17) % C for i in range(16) for j in range(i + 1, 16)]
                # Find a color that appears frequently and swap some of its edges
                color_counts = [0] * C
                for c in colors:
                    color_counts[c] += 1
                if max(color_counts) > 0:
                    target_color = color_counts.index(max(color_counts))
                    swap_count = min(15, color_counts[target_color] // 2)
                    for _ in range(swap_count):
                        e_idx = random.randint(0, NUM_EDGES - 1)
                        if colors[e_idx] == target_color:
                            colors[e_idx] = random.randint(0, C - 1)
            else:
                # Random initialization
                colors = [random.randint(0, C-1) for _ in range(NUM_EDGES)]
            
            # Initialize distinct counts and color histograms
            distinct = [0] * NUM_K4
            hist = [[0]*C for _ in range(NUM_K4)]
            
            total_violation = 0
            violating_k4s = []
            
            for k4_idx in range(NUM_K4):
                e_idxs = K4_EDGES[k4_idx]
                h = hist[k4_idx]
                for e_idx in e_idxs:
                    c = colors[e_idx]
                    h[c] += 1
                d = sum(1 for cnt in h if cnt > 0)
                distinct[k4_idx] = d
                if d < 5:
                    total_violation += (5 - d)
                    violating_k4s.append(k4_idx)
            
            if total_violation == 0:
                return colors
            
            # Build list of edges in violating K4s for targeted moves
            violating_edges = set()
            for k4_idx in violating_k4s:
                for e_idx in K4_EDGES[k4_idx]:
                    violating_edges.add(e_idx)
            violating_edges_list = list(violating_edges)
            
            # Simulated Annealing with adaptive parameters
            T = 2.0
            current_violation = total_violation
            best_violation = total_violation
            best_colors = colors.copy()
            
            steps = 0
            max_steps = 20000
            stuck_count = 0
            no_improve_steps = 0
            last_best_step = 0
            
            while steps < max_steps:
                if time.time() > deadline - 5:
                    break
                
                # Targeted move: 75% chance pick edge from violating K4, 25% random
                if violating_edges_list and random.random() < 0.75:
                    e_idx = violating_edges_list[random.randint(0, len(violating_edges_list) - 1)]
                else:
                    e_idx = random.randint(0, NUM_EDGES - 1)
                
                old_color = colors[e_idx]
                new_color = random.randint(0, C - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta violation
                old_total_v = current_violation
                new_total_v = current_violation
                
                affected_k4s = EDGE_TO_K4S[e_idx]
                
                # Apply change
                colors[e_idx] = new_color
                
                # Update affected K4s
                for k4_idx in affected_k4s:
                    h = hist[k4_idx]
                    old_d = distinct[k4_idx]
                    
                    # Update histogram
                    h[old_color] -= 1
                    h[new_color] += 1
                    
                    # Update distinct count
                    new_d = old_d
                    if h[old_color] == 0:
                        new_d -= 1
                    if h[new_color] == 1:
                        new_d += 1
                        
                    distinct[k4_idx] = new_d
                    
                    # Update violation
                    old_v = max(0, 5 - old_d)
                    new_v = max(0, 5 - new_d)
                    new_total_v += (new_v - old_v)
                
                # Acceptance criterion with adaptive threshold
                delta = new_total_v - old_total_v
                accept = False
                if delta < 0:
                    accept = True
                else:
                    # Adaptive threshold: tighter when close to solution
                    if current_violation < 10:
                        threshold = 0.005
                    elif current_violation < 50:
                        threshold = 0.05
                    else:
                        threshold = 0.15
                    
                    if T > threshold:
                        # Use a slightly modified acceptance to encourage exploration
                        if random.random() < math.exp(-delta / T):
                            accept = True
                
                if accept:
                    current_violation = new_total_v
                    stuck_count = 0
                else:
                    # Revert
                    colors[e_idx] = old_color
                    for k4_idx in affected_k4s:
                        h = hist[k4_idx]
                        old_d = distinct[k4_idx]
                        
                        # Revert histogram
                        h[new_color] -= 1
                        h[old_color] += 1
                        
                        # Revert distinct count
                        new_d = old_d
                        if h[new_color] == 0:
                            new_d -= 1
                        if h[old_color] == 1:
                            new_d += 1
                        distinct[k4_idx] = new_d
                    stuck_count += 1
                
                if current_violation < best_violation:
                    best_violation = current_violation
                    best_colors = colors.copy()
                    if best_violation == 0:
                        return colors
                    stuck_count = 0
                    no_improve_steps = 0
                    last_best_step = steps
                else:
                    no_improve_steps += 1
                
                # Periodically refresh violating edges list
                if steps % 300 == 0 and current_violation > 0:
                    violating_edges = set()
                    for k4_idx in range(NUM_K4):
                        if distinct[k4_idx] < 5:
                            for e_idx in K4_EDGES[k4_idx]:
                                violating_edges.add(e_idx)
                    violating_edges_list = list(violating_edges)
                
                # Adaptive cooling
                if stuck_count > 300:
                    T *= 0.92
                    stuck_count = 0
                    if T < 0.01:
                        break
                
                # Restart if stuck for too long
                if no_improve_steps > 5000 and current_violation > 0:
                    # Inject some randomness to escape local optima
                    for _ in range(20):
                        e_idx = random.randint(0, NUM_EDGES - 1)
                        old_c = colors[e_idx]
                        new_c = random.randint(0, C - 1)
                        if new_c != old_c:
                            affected = EDGE_TO_K4S[e_idx]
                            for k4_idx in affected:
                                h = hist[k4_idx]
                                old_d = distinct[k4_idx]
                                h[old_c] -= 1
                                h[new_c] += 1
                                new_d = old_d
                                if h[old_c] == 0:
                                    new_d -= 1
                                if h[new_c] == 1:
                                    new_d += 1
                                distinct[k4_idx] = new_d
                            colors[e_idx] = new_c
                    # Recalculate violation
                    current_violation = 0
                    for k4_idx in range(NUM_K4):
                        current_violation += max(0, 5 - distinct[k4_idx])
                    no_improve_steps = 0
                    T = max(T, 0.5)
                
                steps += 1
                
            # If loop finishes, check if we found a valid solution
            if best_violation == 0:
                return best_colors
    
    # Fallback: return the best found or the base construction
    # Try to find a valid solution with the base 17-color construction
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Check if base is valid
    distinct_check = [0] * NUM_K4
    for k4_idx in range(NUM_K4):
        e_idxs = K4_EDGES[k4_idx]
        seen = set()
        for e_idx in e_idxs:
            seen.add(base_colors[e_idx])
        distinct_check[k4_idx] = len(seen)
    
    if all(d >= 5 for d in distinct_check):
        return base_colors
    
    # Return the base construction as fallback
    return base_colors
