import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs
    # An AP is (a, a+d, a+2d, a+3d) with 0-based indices
    # a in [0, N-1], d >= 1, a+3d < N
    aps = []
    aps_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)
            # Add to adjacency lists for fast delta calculation
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)
                
    # We can represent colors as a numpy array for faster operations if needed, 
    # but for delta calculation, a list of ints is often faster in pure Python loops 
    # due to overhead of numpy scalar access. Let's stick to list but optimize the loop.
    
    # Initial random coloring
    colors = [random.randint(0, 1) for _ in range(N)]
    
    # Helper to count monochromatic APs
    def count_mono(cols):
        cnt = 0
        for ap in aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt
        
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99995
    min_T = 0.01
    
    # Pre-calculate exp for common delta values? 
    # Delta is -1, 0, or 1.
    # exp(-(-1)/T) = exp(1/T) > 1 (always accept)
    # exp(-0/T) = 1 (always accept)
    # exp(-1/T) (probabilistic)
    
    while time.time() < deadline:
        # Pick random index
        i = random.randint(0, N - 1)
        
        # Calculate delta efficiently
        delta = 0
        # For each AP containing i
        for others in aps_at[i]:
            j, k, l = others
            c_i = colors[i]
            c_j = colors[j]
            c_k = colors[k]
            c_l = colors[l]
            
            # Check if current AP is monochromatic
            # If c_i == c_j == c_k == c_l, it contributes 1 to cost.
            # After flip, c_i' = 1 - c_i.
            # New AP is mono if c_i' == c_j == c_k == c_l.
            
            # Since c_i is either 0 or 1:
            # Case 1: c_i == c_j == c_k == c_l (was mono)
            #   Then c_i' != c_j, so new AP is NOT mono.
            #   Delta = -1
            # Case 2: c_i != c_j (was not mono, specifically first element differs)
            #   If c_j == c_k == c_l, then c_i' = 1-c_i.
            #   If c_i' == c_j, i.e., 1-c_i == c_j, i.e., c_i != c_j.
            #   Wait, if c_j == c_k == c_l, then:
            #   If c_i == c_j, it was mono (handled in Case 1).
            #   If c_i != c_j, then c_i' = 1-c_i = c_j. So new AP IS mono.
            #   Delta = +1
            # Case 3: c_j, c_k, c_l are not all equal.
            #   Then neither old nor new AP can be mono.
            #   Delta = 0
            
            # Optimization:
            if c_j == c_k == c_l:
                if c_i == c_j:
                    delta -= 1
                else:
                    delta += 1
            # else delta += 0
            
        if delta < 0:
            colors[i] = 1 - colors[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
        elif delta == 0:
            # Accept to explore, no cost change
            colors[i] = 1 - colors[i]
        else: # delta > 0
            # Accept with probability exp(-delta / T)
            # delta is always 1 in this problem for a single flip?
            # Let's check. Can delta be > 1?
            # Yes, if i is in multiple APs.
            # But for a single flip, each AP contributes -1, 0, or 1.
            # So delta is sum of contributions.
            
            # Calculate acceptance probability
            # To avoid underflow/overflow issues, use log?
            # Or just math.exp. T is small, so -delta/T is large negative.
            if T > 0:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
                
            if random.random() < prob:
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_colors
