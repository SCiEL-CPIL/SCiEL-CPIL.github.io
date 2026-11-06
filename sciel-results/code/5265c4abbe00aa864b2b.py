import time
import random
import numpy as np

# Precompute K4s and edge-to-K4 mapping
N = 16
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i+1, N):
        edge_to_idx[(i,j)] = len(edges)
        edges.append((i,j))

k4s = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                k4s.append((a,b,c,d))

# Map each edge to the list of K4 indices it participates in
edge_k4s = [[] for _ in edges]
for k4_idx, (a,b,c,d) in enumerate(k4s):
    # Edges in K4: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
    for u, v in [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]:
        e_idx = edge_to_idx[(u,v)]
        edge_k4s[e_idx].append(k4_idx)

# Helper to get distinct colours in a K4 given current colouring
def get_k4_colours(colours, a, b, c, d):
    # colours is a flat array of length 120
    c_ab = colours[edge_to_idx[(a,b)]]
    c_ac = colours[edge_to_idx[(a,c)]]
    c_ad = colours[edge_to_idx[(a,d)]]
    c_bc = colours[edge_to_idx[(b,c)]]
    c_bd = colours[edge_to_idx[(b,d)]]
    c_cd = colours[edge_to_idx[(c,d)]]
    return len({c_ab, c_ac, c_ad, c_bc, c_bd, c_cd})

# Incremental penalty update is complex to code correctly in short time.
# Given 1820 K4s, recalculating total penalty from scratch is O(1820 * 6) ~ 10k ops.
# In Python, 10k ops is very fast (~0.001s).
# 1000 iterations/s is easily achievable.
# 20s -> 20,000 iterations. Might be enough for small K.

def total_penalty(colours):
    pen = 0
    for a, b, c, d in k4s:
        n_col = get_k4_colours(colours, a, b, c, d)
        if n_col < 5:
            pen += (5 - n_col) ** 2
    return pen

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        # Run multiple restarts if time permits
        # Each restart gets ~5s?
        # Let's do one long run per K, or a few short runs.
        
        best_colours = None
        best_penalty = float('inf')
        
        # Restart loop
        max_restarts = 10
        time_per_restart = 5.0
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random
            colours = np.random.randint(0, K, size=120)
            current_penalty = total_penalty(colours)
            
            if current_penalty == 0:
                return colours.tolist()
                
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colours = colours.copy()
                
            # Simulated Annealing / Hill Climbing
            T = 100.0
            T_min = 1.0
            step = 0.95
            t_start = time.time()
            
            while time.time() - t_start < time_per_restart:
                # Pick random edge
                e_idx = random.randint(0, 119)
                old_c = colours[e_idx]
                
                # Pick new colour
                if K == 1:
                    new_c = 0
                else:
                    new_c = random.randint(0, K - 1)
                    while new_c == old_c:
                        new_c = random.randint(0, K - 1)
                        
                colours[e_idx] = new_c
                new_penalty = total_penalty(colours)
                
                # Accept if better or with probability exp(-delta/T)
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    if current_penalty == 0:
                        return colours.tolist()
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colours = colours.copy()
                else:
                    delta = new_penalty - current_penalty
                    if delta < 0:
                        accept = True
                    else:
                        accept = random.random() < np.exp(-delta / T)
                        
                    if accept:
                        current_penalty = new_penalty
                    else:
                        colours[e_idx] = old_c
                        
                T *= step
                if T < T_min:
                    T = T_min
                    
        if best_penalty == 0:
            return best_colours.tolist()
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
