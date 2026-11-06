import time
import random
import numpy as np
from itertools import combinations

N = 16
# Generate all edges
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

# Generate all K4s
k4s = []
k4_edges = [] # List of lists of edge indices for each K4
for a, b, c, d in combinations(range(N), 4):
    k4s.append((a, b, c, d))
    # Edges in K4(a,b,c,d)
    e_indices = [
        edge_to_idx[(a, b)],
        edge_to_idx[(a, c)],
        edge_to_idx[(a, d)],
        edge_to_idx[(b, c)],
        edge_to_idx[(b, d)],
        edge_to_idx[(c, d)]
    ]
    k4_edges.append(e_indices)

# Map each edge to the list of K4 indices it participates in
edge_k4s = [[] for _ in range(len(edges))]
for k4_idx, e_indices in enumerate(k4_edges):
    for e_idx in e_indices:
        edge_k4s[e_idx].append(k4_idx)

def count_distinct_colors(colours, k4_idx):
    e_indices = k4_edges[k4_idx]
    c0 = colours[e_indices[0]]
    c1 = colours[e_indices[1]]
    c2 = colours[e_indices[2]]
    c3 = colours[e_indices[3]]
    c4 = colours[e_indices[4]]
    c5 = colours[e_indices[5]]
    return len({c0, c1, c2, c3, c4, c5})

def total_bad_count(colours):
    bad = 0
    for i in range(len(k4s)):
        if count_distinct_colors(colours, i) < 5:
            bad += 1
    return bad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for Python overhead
    
    # Try K = 13, 14, 15
    # We want the minimum K that allows a valid coloring.
    # If we find one for 13, we return it.
    
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        # We will run a local search for a valid coloring with K colors.
        # If found, return immediately.
        
        # Initialize random coloring
        colours = np.random.randint(0, K, size=120)
        bad_count = total_bad_count(colours)
        
        if bad_count == 0:
            return colours.tolist()
            
        # Local Search Parameters
        T = 50.0
        T_min = 0.1
        cooling = 0.99
        max_iterations = 200000 # Safety cap, but time is the real limit
        
        # We use a simple hill climbing with random restarts if stuck.
        # But first, let's try a longer SA run.
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            if restart > 0:
                # Perturb or re-initialize
                if restart % 2 == 0:
                    # Random restart
                    colours = np.random.randint(0, K, size=120)
                    bad_count = total_bad_count(colours)
                    if bad_count == 0:
                        return colours.tolist()
                else:
                    # Perturb current solution
                    for _ in range(10):
                        e_idx = random.randint(0, 119)
                        colours[e_idx] = random.randint(0, K - 1)
                    bad_count = total_bad_count(colours)
                    if bad_count == 0:
                        return colours.tolist()
            
            T = 50.0
            # SA Loop
            while time.time() < deadline - 5:
                # Pick a random edge
                e_idx = random.randint(0, 119)
                old_c = colours[e_idx]
                new_c = random.randint(0, K - 1)
                
                if new_c == old_c:
                    continue
                
                # Calculate delta in bad_count
                # Only K4s containing e_idx are affected
                affected_k4s = edge_k4s[e_idx]
                
                delta = 0
                # Check if they were bad before
                # We need to know if they are bad *before* the change
                # and *after* the change.
                
                # To do this efficiently, we can just recalculate the status of these specific K4s.
                # Since K is small (13-15), distinct color count is fast.
                
                # Store old badness
                old_badness = 0
                for k4_idx in affected_k4s:
                    if count_distinct_colors(colours, k4_idx) < 5:
                        old_badness += 1
                        
                # Apply change
                colours[e_idx] = new_c
                
                # Check new badness
                new_badness = 0
                for k4_idx in affected_k4s:
                    if count_distinct_colors(colours, k4_idx) < 5:
                        new_badness += 1
                        
                delta = new_badness - old_badness
                bad_count += delta
                
                if delta < 0:
                    # Improvement
                    pass
                elif delta == 0:
                    # Neutral, accept to move around
                    pass
                else:
                    # Worse
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        # Reject
                        colours[e_idx] = old_c
                        bad_count -= delta
                        T *= cooling
                        if T < T_min:
                            T = T_min
                        continue
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if bad_count == 0:
                    return colours.tolist()
                
    # Fallback to seed solution (17 colors)
    # (i+j) % 17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
