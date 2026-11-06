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
        for restart in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = [random.randint(0, C-1) for _ in range(NUM_EDGES)]
            
            # Initialize distinct counts and color histograms
            distinct = [0] * NUM_K4
            hist = [[0]*C for _ in range(NUM_K4)]
            
            total_violation = 0
            
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
            
            if total_violation == 0:
                return colors
            
            # Simulated Annealing
            T = 1.0
            current_violation = total_violation
            best_violation = total_violation
            best_colors = colors.copy()
            
            steps = 0
            max_steps = 10000 # Reduced steps per restart to allow more restarts
            stuck_count = 0
            
            while steps < max_steps:
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge
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
                
                # Acceptance criterion
                delta = new_total_v - old_total_v
                accept = False
                if delta < 0:
                    accept = True
                else:
                    # If violation is low, be stricter
                    threshold = 0.01 if current_violation < 20 else 0.1
                    if T > threshold:
                        # Accept with probability exp(-delta / T)
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
                
                # Cool down more aggressively
                if stuck_count > 500:
                    T *= 0.9
                    stuck_count = 0
                    if T < 0.01:
                        break
                
                steps += 1
                
            # If loop finishes, check if we found a valid solution
            if best_violation == 0:
                return best_colors
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
