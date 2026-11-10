import time
import random
import numpy as np
from itertools import combinations

# Precompute structures for efficiency
# We will target n=34 first, then fall back if needed.
# The evaluator kills at 150s. We should use ~140s for search.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a valid coloring for n=34
    # If that fails, try smaller n
    for target_n in [34, 33, 32, 31, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline - 5.0:
            break
            
        valid_colors, success = _local_search(target_n, deadline)
        if success:
            return {'n': target_n, 'colors': valid_colors}
            
    # Fallback to seed solution if all fail (should not happen)
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def _local_search(n, deadline):
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Generate all K4s and K5s
    # K4 is a set of 4 vertices, has 4 triples
    # K5 is a set of 5 vertices, has 10 triples
    
    k4s = []
    k4_triples = [] # List of lists of triple indices for each K4
    for quad in combinations(range(n), 4):
        tris = []
        for t in combinations(quad, 3):
            tris.append(triple_to_idx[t])
        k4s.append(quad)
        k4_triples.append(tris)
        
    k5s = []
    k5_triples = [] # List of lists of triple indices for each K5
    for quint in combinations(range(n), 5):
        tris = []
        for t in combinations(quint, 3):
            tris.append(triple_to_idx[t])
        k5s.append(quint)
        k5_triples.append(tris)
        
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    # Precompute for each triple, which K4s and K5s it belongs to
    # This allows fast delta calculation
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for i, tris in enumerate(k4_triples):
        for t_idx in tris:
            triple_to_k4s[t_idx].append(i)
            
    for i, tris in enumerate(k5_triples):
        for t_idx in tris:
            triple_to_k5s[t_idx].append(i)
            
    # Initialize coloring randomly
    colors = np.random.randint(0, 2, num_triples)
    
    # Count initial violations
    # red_k4_count[i] is 1 if K4 i is all red
    # blue_k5_count[i] is 1 if K5 i is all blue
    red_k4_count = np.zeros(num_k4, dtype=np.int8)
    blue_k5_count = np.zeros(num_k5, dtype=np.int8)
    
    # To count efficiently, we can check each K4/K5
    # This is O(num_k4 * 4 + num_k5 * 10) which is fine for n=34
    for i, tris in enumerate(k4_triples):
        if all(colors[t] == 0 for t in tris):
            red_k4_count[i] = 1
            
    for i, tris in enumerate(k5_triples):
        if all(colors[t] == 1 for t in tris):
            blue_k5_count[i] = 1
            
    total_violations = np.sum(red_k4_count) + np.sum(blue_k5_count)
    
    if total_violations == 0:
        return colors.tolist(), True
        
    # Local search
    # We use a simple hill climbing with random restarts if stuck
    # Or simulated annealing
    # Let's use a simple acceptance rule: accept if improves, or with small prob if worsens
    
    current_temp = 0.5
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random triple
        t_idx = random.randint(0, num_triples - 1)
        
        # Calculate delta
        # Flipping color of t_idx changes its status from 0 to 1 or 1 to 0
        old_color = colors[t_idx]
        new_color = 1 - old_color
        
        delta = 0
        
        # Check K4s containing t_idx
        for k4_idx in triple_to_k4s[t_idx]:
            tris = k4_triples[k4_idx]
            # Count reds in this K4 before flip
            # If old_color was 0 (red), removing it reduces red count by 1
            # If old_color was 1 (blue), removing it doesn't change red count
            # Then adding new_color
            
            # Efficiently: 
            # Current state: red_k4_count[k4_idx] is 1 if all 4 are red.
            # If old_color == 0 (red):
            #   If it was a red K4 (all red), after flip it has 3 reds -> not red K4. Delta -= 1
            #   If it was not a red K4, it had at least one blue. 
            #     If the other 3 are red, and we flip this to blue, it's still not red.
            #     Wait, if old_color==0, and it was NOT a red K4, that means at least one other is blue.
            #     Flipping this to blue doesn't make it red. So delta 0?
            #     Actually, if old_color==0, and other 3 are red, it WAS a red K4.
            #     So if old_color==0 and red_k4_count[k4_idx]==1, delta -= 1.
            #     If old_color==0 and red_k4_count[k4_idx]==0, delta 0.
            
            # If old_color == 1 (blue):
            #   If it was a red K4? Impossible, because this one is blue.
            #   So red_k4_count[k4_idx] must be 0.
            #   After flip to red (0):
            #     If the other 3 are red, it becomes a red K4. Delta += 1.
            #     How to check if other 3 are red?
            #     We can precompute or check on the fly.
            #     Checking on the fly: sum(colors[tris]) == 3 (since one is blue, sum of 4 is 1+3=4? No, colors are 0/1. Red=0, Blue=1.
            #     Sum of colors in K4. If all red, sum=0.
            #     If old_color=1, sum = 1 + sum(others).
            #     If sum(others)==0, then new sum=0 -> becomes red K4.
            
            if old_color == 0:
                if red_k4_count[k4_idx] == 1:
                    delta -= 1
                    red_k4_count[k4_idx] = 0
                # else no change
            else: # old_color == 1
                # Check if other 3 are red
                # Sum of colors of other 3
                s = 0
                for t in tris:
                    if t != t_idx:
                        s += colors[t]
                if s == 0:
                    delta += 1
                    red_k4_count[k4_idx] = 1
        
        # Check K5s containing t_idx
        for k5_idx in triple_to_k5s[t_idx]:
            tris = k5_triples[k5_idx]
            
            if old_color == 1: # Blue
                if blue_k5_count[k5_idx] == 1:
                    delta -= 1
                    blue_k5_count[k5_idx] = 0
            else: # old_color == 0 (Red)
                # Check if other 9 are blue
                s = 0
                for t in tris:
                    if t != t_idx:
                        s += colors[t]
                # If all 10 are blue, sum=10.
                # Old color 0, so sum = 0 + sum(others).
                # If sum(others) == 9, then new sum = 10 -> becomes blue K5.
                if s == 9:
                    delta += 1
                    blue_k5_count[k5_idx] = 1
        
        # Accept/Reject
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / current_temp)):
            colors[t_idx] = new_color
            total_violations += delta
            
            if total_violations == 0:
                return colors.tolist(), True
        
        current_temp *= cooling_rate
        if current_temp < 0.01:
            break
            
    return colors.tolist(), False

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a valid coloring for n=34
    # If that fails, try smaller n
    for target_n in [34, 33, 32, 31, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline - 5.0:
            break
            
        valid_colors, success = _local_search(target_n, deadline)
        if success:
            return {'n': target_n, 'colors': valid_colors}
            
    # Fallback to seed solution if all fail (should not happen)
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
