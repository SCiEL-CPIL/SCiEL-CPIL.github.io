import time
import random
import math
import numpy as np
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # 1. Precompute Overlap Matrix
    # M[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # M is N x N, uint8
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Build prefix maps for each k
    # prefix_maps[k] maps tuple of length k to list of indices
    prefix_maps = [{} for _ in range(n)] # indices 0 to n-1, we use 1 to n-1
    
    for k in range(1, n):
        pm = prefix_maps[k]
        for j, p in enumerate(perms):
            pre = p[:k]
            if pre not in pm:
                pm[pre] = []
            pm[pre].append(j)
            
    # Fill M
    # Iterate k from n-1 down to 1
    # For each i, get suffix of length k
    # Find j's with that prefix
    # Set M[i][j] = k if M[i][j] == 0 (i.e., not set by larger k)
    for k in range(n - 1, 0, -1):
        pm = prefix_maps[k]
        for i in range(N):
            suf = perms[i][n - k:]
            if suf in pm:
                js = pm[suf]
                for j in js:
                    if M[i][j] == 0:
                        M[i][j] = k
                        
    # 2. Generate Initial Order (Greedy Nearest Neighbor)
    visited = np.zeros(N, dtype=bool)
    order = [0]
    visited[0] = True
    current = 0
    
    for _ in range(N - 1):
        # Get overlaps from current to all
        scores = M[current].copy()
        scores[visited] = -1
        best_next = np.argmax(scores)
        order.append(best_next)
        visited[best_next] = True
        current = best_next
        
    current_order = order
    # Calculate initial score
    current_score = 0
    for i in range(N - 1):
        current_score += M[current_order[i]][current_order[i+1]]
        
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    deadline = 140 # 10s margin
    
    T = 1.0
    min_T = 0.01
    
    while time.time() - start_time < deadline:
        # Random swap
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate delta
        # Affected edges:
        # (i-1, i), (i, i+1)
        # (j-1, j), (j, j+1)
        
        old_score = 0
        new_score = 0
        
        # Old overlaps
        if i > 0: 
            old_score += M[current_order[i-1]][current_order[i]]
        old_score += M[current_order[i]][current_order[i+1]]
        if j > 0: 
            old_score += M[current_order[j-1]][current_order[j]]
        old_score += M[current_order[j]][current_order[j+1]]
        
        # Perform swap
        pi_i = current_order[i]
        pi_j = current_order[j]
        current_order[i] = pi_j
        current_order[j] = pi_i
        
        # New overlaps
        if i > 0: 
            new_score += M[current_order[i-1]][current_order[i]]
        new_score += M[current_order[i]][current_order[i+1]]
        if j > 0: 
            new_score += M[current_order[j-1]][current_order[j]]
        new_score += M[current_order[j]][current_order[j+1]]
        
        delta = new_score - old_score
        
        # Acceptance criterion
        if delta > 0 or (T > min_T and random.random() < math.exp(delta / T)):
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i] = pi_i
            current_order[j] = pi_j
            
        # Cool down
        T *= 0.999
        if T < min_T:
            T = min_T
            
    # Construct final word
    final_word = list(perms[best_order[0]])
    for i in range(1, len(best_order)):
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        prev = perms[prev_idx]
        curr = perms[curr_idx]
        
        # Find max overlap
        max_ov = M[prev_idx][curr_idx]
        final_word += curr[max_ov:]
        
    return "".join(map(str, final_word))
