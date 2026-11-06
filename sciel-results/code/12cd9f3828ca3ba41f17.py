import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
                
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    # Precompute delta data for fast delta calculation
    # delta_data[i] is a list of tuples (u, v, w) representing the other 3 indices in APs containing i
    delta_data = [[] for _ in range(N)]
    for a, b, c, d in aps:
        delta_data[a].append((b, c, d))
        delta_data[b].append((a, c, d))
        delta_data[c].append((a, b, d))
        delta_data[d].append((a, b, c))
        
    def count_mono_vec(state):
        # state is numpy array of 0/1
        # Check if all 4 elements in each AP are equal
        # state[aps_array] has shape (num_aps, 4)
        # Compare each column to the first column
        s = state[aps_array]
        return np.sum(np.all(s == s[:, 0:1], axis=1))
        
    def get_delta(state, i):
        delta = 0
        new_val = 1 - state[i]
        for u, v, w in delta_data[i]:
            # Mono before?
            if state[i] == state[u] == state[v] == state[w]:
                delta -= 1
            # Mono after?
            if new_val == state[u] == state[v] == state[w]:
                delta += 1
        return delta

    best_state = None
    best_score = float('inf')
    
    # Helper to polish
    def polish(state, score, max_time_left=5.0):
        improved = True
        while improved and (time.time() + max_time_left) < deadline:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                delta = get_delta(state, i)
                if delta < 0:
                    state[i] = 1 - state[i]
                    score += delta
                    improved = True
        return score

    # 1. Try polishing the seed
    seed_state = np.array([(i // 2) % 2 for i in range(N)])
    seed_score = count_mono_vec(seed_state)
    polished_seed_score = polish(seed_state.copy(), seed_score)
    if polished_seed_score < best_score:
        best_score = polished_seed_score
        best_state = seed_state.copy()
        
    # 2. Simulated Annealing from random and seed
    while time.time() < deadline - 10.0: # Leave 10s for final polish of best found
        # Initialize state
        if random.random() < 0.5:
            state = np.random.randint(0, 2, N)
        else:
            # Perturb best state or seed
            if best_state is not None:
                state = best_state.copy()
                # Randomly flip 10%
                for _ in range(N // 10):
                    state[random.randint(0, N-1)] = 1 - state[random.randint(0, N-1)]
            else:
                state = np.random.randint(0, 2, N)
                
        current_score = count_mono_vec(state)
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        steps = 20000
        
        for _ in range(steps):
            if time.time() > deadline - 10.0:
                break
            i = random.randint(0, N - 1)
            delta = get_delta(state, i)
            
            if delta < 0 or (random.random() < np.exp(-delta / T)):
                state[i] = 1 - state[i]
                current_score += delta
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # Polish this result
        current_score = polish(state, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_state = state.copy()
            
    # Final Polish of the best state found
    if best_state is not None:
        final_score = polish(best_state, best_score)
        if final_score < best_score:
            best_score = final_score
            
    if best_state is None:
        return [(i // 2) % 2 for i in range(N)]
        
    return best_state.tolist()
