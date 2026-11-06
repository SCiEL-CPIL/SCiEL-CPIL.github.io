import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    # Initialize adjacency matrix
    adj = np.zeros((42, 42), dtype=bool) # Pre-allocate max size
    for i, j in seed_edges:
        adj[i, j] = True
        adj[j, i] = True
        
    current_n = q
    
    # Precompute K4s and Ind4s and their incidence lists
    def get_k4s_ind4s_and_incidence(adj, n):
        k4s = []
        ind4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_ind = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = combo[i], combo[j]
                    if adj[u, v]:
                        is_ind = False
                    else:
                        is_clique = False
                    if not is_clique and not is_ind:
                        break
                if not is_clique and not is_ind:
                    break
            if is_clique:
                k4s.append(combo)
            if is_ind:
                ind4s.append(combo)
        
        # Build incidence lists
        # k4_inc[i] = list of indices of k4s containing i
        k4_inc = [[] for _ in range(n)]
        for idx, k4 in enumerate(k4s):
            for v in k4:
                k4_inc[v].append(idx)
                
        ind4_inc = [[] for _ in range(n)]
        for idx, i4 in enumerate(ind4s):
            for v in i4:
                ind4_inc[v].append(idx)
                
        return k4s, ind4s, k4_inc, ind4_inc

    k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
    
    def compute_soft_violation(x, k4s, ind4s, k4_inc, ind4_inc, n):
        """Compute a smooth surrogate objective: sum of squared excesses beyond 3 neighbors/non-neighbors in each K4/Ind4."""
        # For each K4, if x has 4 ones, that's a violation. We want to minimize the "excess"
        # Use a smooth penalty: for each K4, penalty = max(0, sum(x[k4]) - 3)^2
        # For each Ind4, penalty = max(0, sum(1-x[ind4]) - 3)^2
        total = 0.0
        for idx, k4 in enumerate(k4s):
            s = float(np.sum(x[list(k4)]))
            excess = max(0.0, s - 3.0)
            total += excess * excess
        for idx, i4 in enumerate(ind4s):
            s = float(np.sum(1 - x[list(i4)]))
            excess = max(0.0, s - 3.0)
            total += excess * excess
        return total
    
    def compute_hard_violations(x, k4s, ind4s, k4_inc, ind4_inc, n):
        """Count actual K5 and Ind5 violations."""
        k5_count = 0
        ind5_count = 0
        for idx, k4 in enumerate(k4s):
            if np.sum(x[list(k4)]) == 4:
                k5_count += 1
        for idx, i4 in enumerate(ind4s):
            if np.sum(1 - x[list(i4)]) == 4:
                ind5_count += 1
        return k5_count + ind5_count
    
    # Try to add vertices
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 20:
            break
            
        found_valid = False
        max_restarts = 300
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Better initialization: use greedy based on degree
            # Vertices with higher degree in the Paley graph are more "central"
            # Try to connect to a balanced set
            x = np.zeros(current_n, dtype=bool)
            
            # Strategy: randomly initialize but bias toward 50/50
            for i in range(current_n):
                x[i] = random.random() < 0.5
            
            # Compute initial soft violation
            current_soft = compute_soft_violation(x, k4s, ind4s, k4_inc, ind4_inc, current_n)
            current_hard = compute_hard_violations(x, k4s, ind4s, k4_inc, ind4_inc, current_n)
            
            if current_hard == 0:
                found_valid = True
                break
            
            # Simulated Annealing with soft objective
            temperature = 0.5
            cooling_rate = 0.9995
            steps = 5000
            
            best_x = x.copy()
            best_soft = current_soft
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                if current_hard == 0:
                    found_valid = True
                    break
                    
                # Flip a random bit
                idx_v = np.random.randint(0, current_n)
                old_val = x[idx_v]
                new_val = not old_val
                x[idx_v] = new_val
                
                new_soft = compute_soft_violation(x, k4s, ind4s, k4_inc, ind4_inc, current_n)
                new_hard = compute_hard_violations(x, k4s, ind4s, k4_inc, ind4_inc, current_n)
                
                delta_soft = new_soft - current_soft
                delta_hard = new_hard - current_hard
                
                # Accept if hard violations decrease, or with probability based on soft improvement
                if delta_hard < 0:
                    accept = True
                elif delta_hard == 0 and delta_soft < 0:
                    accept = True
                elif delta_hard > 0:
                    accept = random.random() < np.exp(-delta_hard / temperature)
                else:
                    # delta_hard == 0, delta_soft >= 0
                    accept = random.random() < np.exp(-delta_soft / temperature)
                
                if accept:
                    current_soft = new_soft
                    current_hard = new_hard
                    if current_soft < best_soft:
                        best_soft = current_soft
                        best_x = x.copy()
                else:
                    x[idx_v] = old_val  # Revert
                
                temperature *= cooling_rate
            
            if current_hard == 0:
                found_valid = True
                break
            
            # Try best found so far in this restart
            if best_soft < current_soft:
                x = best_x.copy()
                current_hard = compute_hard_violations(x, k4s, ind4s, k4_inc, ind4_inc, current_n)
                if current_hard == 0:
                    found_valid = True
                    break
        
        if found_valid:
            # Add vertex target_n - 1
            new_idx = target_n - 1
            for i in range(current_n):
                if x[i]:
                    adj[i, new_idx] = True
                    adj[new_idx, i] = True
            
            current_n = target_n
            best_n = current_n
            # Update best_edges
            best_edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if adj[i, j]:
                        best_edges.append((i, j))
            
            # Update k4s, ind4s, and incidence
            k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
        else:
            # If we can't add this vertex, stop
            break
            
    return best_n, best_edges
