import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # We aim for the maximum n. Known valid n is 34.
    # We will try n=34, then 33, etc.
    # If we find a valid coloring for n, we return it.
    
    # To be safe, we start with n=34.
    # If it fails, we try smaller n.
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
            
        # Allocate time for this n
        # Give more time to higher n? Or fixed time?
        # Let's give 20s to n=34, 10s to n=33, etc.
        # But if n=34 succeeds, we return immediately.
        
        time_limit = 20.0
        if n < 34:
            time_limit = 10.0
        if n < 30:
            time_limit = 5.0
        if n < 25:
            time_limit = 2.0
            
        # Adjust time limit to not exceed deadline
        remaining = deadline - time.time()
        if time_limit > remaining:
            time_limit = remaining
            if time_limit < 0.1:
                break

        solution = solve_hypergraph_ramsey(n, time_limit)
        if solution is not None:
            return solution

    # Fallback: Return a small valid construction if all else fails
    # For n=9, a simple algebraic construction might work, or just random search
    # Actually, the prompt says "published construction has 34 points".
    # If we fail to find 34, we should try to return the best we found.
    # But the loop returns immediately on success.
    # If we reach here, we failed to find any n in the range.
    # Let's try a very short random search for n=10 as a last resort.
    if time.time() < deadline - 1.0:
        solution = solve_hypergraph_ramsey(10, 1.0)
        if solution:
            return solution
            
    # Absolute fallback: n=1 with no triples? No, n must be >= 1.
    # For n=1, 2, 3, there are no K4 or K5, so any coloring is valid.
    # Let's return n=3 with all 0s.
    triples = list(combinations(range(3), 3))
    return {'n': 3, 'colors': [0] * len(triples)}

def solve_hypergraph_ramsey(n, time_limit):
    # Precomputation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Build K4s and K5s
    k4s = []
    k4_map = [[] for _ in range(num_triples)] # k4_map[i] = list of K4 indices containing triple i
    for quad in combinations(range(n), 4):
        edges = []
        for tri in combinations(quad, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k4_idx = len(k4s)
        k4s.append(edges)
        for e in edges:
            k4_map[e].append(k4_idx)
            
    k5s = []
    k5_map = [[] for _ in range(num_triples)] # k5_map[i] = list of K5 indices containing triple i
    for quint in combinations(range(n), 5):
        edges = []
        for tri in combinations(quint, 3):
            idx = triple_index[tri]
            edges.append(idx)
        k5_idx = len(k5s)
        k5s.append(edges)
        for e in edges:
            k5_map[e].append(k5_idx)
            
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    # State
    colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
    
    # Violation Counts
    # red_k4_counts[i] = number of red (0) edges in k4s[i]
    # blue_k5_counts[i] = number of blue (1) edges in k5s[i]
    red_k4_counts = np.zeros(num_k4, dtype=np.int8)
    blue_k5_counts = np.zeros(num_k5, dtype=np.int8)
    
    # Convert k4_map and k5_map to numpy arrays for faster access?
    # Actually, lists of lists are fine if we iterate.
    # But let's try to vectorize the count update.
    
    # Initialize counts
    for i, edges in enumerate(k4s):
        # edges is a list of indices
        # count red (0)
        # colors is 0 or 1. Red is 0.
        # So we want sum(colors[e] == 0)
        # Or sum(1 - colors[e])
        # Using numpy slicing on a list of indices
        red_k4_counts[i] = np.sum(1 - colors[edges])
        
    for i, edges in enumerate(k5s):
        # count blue (1)
        blue_k5_counts[i] = np.sum(colors[edges])
        
    total_violations = np.sum(red_k4_counts == 4) + np.sum(blue_k5_counts == 10)
    
    best_colors = colors.copy()
    best_violations = total_violations
    
    if total_violations == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    start = time.time()
    T = 1.0
    cooling = 0.9995
    
    # Pre-convert k4_map and k5_map to numpy arrays of lists?
    # Or just keep as lists.
    
    while time.time() - start < time_limit:
        # Pick a random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_val = colors[idx]
        new_val = 1 - old_val
        
        delta_violations = 0
        
        # Update K4 counts
        # For each K4 containing idx
        # If old_val == 0 (red), removing red -> count decreases by 1
        # If old_val == 1 (blue), removing blue -> red count increases by 1 (since it becomes red)
        # Wait, colors are 0 (red) and 1 (blue).
        # red_k4_counts counts number of 0s.
        # If we flip 0->1, red count decreases by 1.
        # If we flip 1->0, red count increases by 1.
        
        k4_indices = k4_map[idx]
        if k4_indices:
            old_counts = red_k4_counts[k4_indices].copy()
            if old_val == 0:
                new_counts = old_counts - 1
            else:
                new_counts = old_counts + 1
            
            # Check violation change
            # Violation if count == 4
            # Old violations: sum(old_counts == 4)
            # New violations: sum(new_counts == 4)
            delta_violations += np.sum(new_counts == 4) - np.sum(old_counts == 4)
            
            red_k4_counts[k4_indices] = new_counts
            
        k5_indices = k5_map[idx]
        if k5_indices:
            old_counts = blue_k5_counts[k5_indices].copy()
            if old_val == 1:
                # Removing blue -> blue count decreases by 1
                new_counts = old_counts - 1
            else:
                # Adding blue -> blue count increases by 1
                new_counts = old_counts + 1
                
            # Check violation change
            # Violation if count == 10
            delta_violations += np.sum(new_counts == 10) - np.sum(old_counts == 10)
            
            blue_k5_counts[k5_indices] = new_counts
            
        new_violations = total_violations + delta_violations
        
        # Accept or Reject
        if new_violations < total_violations:
            accept = True
        else:
            if T > 1e-6:
                # Metropolis criterion
                prob = math.exp(-delta_violations / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            colors[idx] = new_val
            total_violations = new_violations
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return {'n': n, 'colors': colors.tolist()}
        else:
            # Revert changes
            colors[idx] = old_val
            if k4_indices:
                if old_val == 0:
                    red_k4_counts[k4_indices] += 1
                else:
                    red_k4_counts[k4_indices] -= 1
            if k5_indices:
                if old_val == 1:
                    blue_k5_counts[k5_indices] += 1
                else:
                    blue_k5_counts[k5_indices] -= 1
            
        T *= cooling
        
    if best_violations == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    return None
