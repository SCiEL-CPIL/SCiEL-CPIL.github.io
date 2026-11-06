import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for 120s limit
    
    # Try to find a valid coloring for n=34 down to n=10
    # The known lower bound for R(4,5;3) is 34, meaning a coloring exists for n=34.
    # We aim for the highest n possible.
    
    for n in range(34, 9, -1):
        if time.time() - start_time > timeout:
            break
            
        # If we fail 34, we might still find 33, etc.
        # But usually, if 34 exists, local search finds it.
        
        result = local_search(n, timeout - (time.time() - start_time))
        if result is not None:
            return result
            
    # Fallback: return the seed result if time runs out or no solution found
    # The seed result was n=9.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def local_search(n, time_limit):
    """
    Finds a valid 2-coloring of triples for n vertices with no red K4^(3) and no blue K5^(3).
    Returns a dict {'n': n, 'colors': [...]} or None if not found within time_limit.
    """
    if n < 4:
        return {'n': n, 'colors': []}
        
    # Precomputation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Generate all K4s (4-sets) and their constituent triples
    k4s = list(combinations(range(n), 4))
    k4_triples = []
    for k4 in k4s:
        # Triples in a K4: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        a,b,c,d = k4
        t_indices = [
            triple_to_idx[(a,b,c)],
            triple_to_idx[(a,b,d)],
            triple_to_idx[(a,c,d)],
            triple_to_idx[(b,c,d)]
        ]
        k4_triples.append(t_indices)
    num_k4s = len(k4s)
    
    # Generate all K5s (5-sets) and their constituent triples
    k5s = list(combinations(range(n), 5))
    k5_triples = []
    for k5 in k5s:
        a,b,c,d,e = k5
        t_indices = [
            triple_to_idx[(a,b,c)],
            triple_to_idx[(a,b,d)],
            triple_to_idx[(a,b,e)],
            triple_to_idx[(a,c,d)],
            triple_to_idx[(a,c,e)],
            triple_to_idx[(a,d,e)],
            triple_to_idx[(b,c,d)],
            triple_to_idx[(b,c,e)],
            triple_to_idx[(b,d,e)],
            triple_to_idx[(c,d,e)]
        ]
        k5_triples.append(t_indices)
    num_k5s = len(k5s)
    
    # Map triple index to the list of K4 indices it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, t_indices in enumerate(k4_triples):
        for t_idx in t_indices:
            triple_to_k4s[t_idx].append(i)
            
    # Map triple index to the list of K5 indices it belongs to
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, t_indices in enumerate(k5_triples):
        for t_idx in t_indices:
            triple_to_k5s[t_idx].append(i)
            
    def solve_single_restart(time_left):
        # Initialize coloring: random
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initialize counts
        # red_k4_count[i] = number of red (0) edges in K4 i
        red_k4_count = np.zeros(num_k4s, dtype=np.int8)
        # blue_k5_count[i] = number of blue (1) edges in K5 i
        blue_k5_count = np.zeros(num_k5s, dtype=np.int8)
        
        # Calculate initial counts
        for k4_idx, t_indices in enumerate(k4_triples):
            # Red is 0
            red_k4_count[k4_idx] = np.sum(colors[t_indices] == 0)
            
        for k5_idx, t_indices in enumerate(k5_triples):
            # Blue is 1
            blue_k5_count[k5_idx] = np.sum(colors[t_indices] == 1)
            
        # Calculate initial cost
        # Violation: Red K4 (count==4) or Blue K5 (count==10)
        cost = np.sum(red_k4_count == 4) + np.sum(blue_k5_count == 10)
        
        if cost == 0:
            return colors.tolist()
            
        best_cost = cost
        best_colors = colors.copy()
        
        temp = 1.0
        cooling = 0.99
        # Simulated Annealing
        while time_left > 0.1:
            # Pick random triple
            t_idx = np.random.randint(num_triples)
            
            # Calculate delta cost
            delta_cost = 0
            
            # Check K4s
            old_red_counts = []
            new_red_counts = []
            for k4_idx in triple_to_k4s[t_idx]:
                old_c = red_k4_count[k4_idx]
                # If we flip t_idx:
                # If colors[t_idx] was 0 (Red), it becomes 1 (Blue). Red count decreases by 1.
                # If colors[t_idx] was 1 (Blue), it becomes 0 (Red). Red count increases by 1.
                if colors[t_idx] == 0:
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                
                # Check if violation status changes
                old_viol = 1 if old_c == 4 else 0
                new_viol = 1 if new_c == 4 else 0
                delta_cost += new_viol - old_viol
                
                old_red_counts.append(old_c)
                new_red_counts.append(new_c)
                
            # Check K5s
            old_blue_counts = []
            new_blue_counts = []
            for k5_idx in triple_to_k5s[t_idx]:
                old_c = blue_k5_count[k5_idx]
                if colors[t_idx] == 1:
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                
                old_viol = 1 if old_c == 10 else 0
                new_viol = 1 if new_c == 10 else 0
                delta_cost += new_viol - old_viol
                
                old_blue_counts.append(old_c)
                new_blue_counts.append(new_c)
                
            # Decide if we accept
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / temp):
                # Apply flip
                if colors[t_idx] == 0:
                    colors[t_idx] = 1
                    for i, k4_idx in enumerate(triple_to_k4s[t_idx]):
                        red_k4_count[k4_idx] = new_red_counts[i]
                    for i, k5_idx in enumerate(triple_to_k5s[t_idx]):
                        blue_k5_count[k5_idx] = new_blue_counts[i]
                else:
                    colors[t_idx] = 0
                    for i, k4_idx in enumerate(triple_to_k4s[t_idx]):
                        red_k4_count[k4_idx] = new_red_counts[i]
                    for i, k5_idx in enumerate(triple_to_k5s[t_idx]):
                        blue_k5_count[k5_idx] = new_blue_counts[i]
                        
                cost += delta_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors.copy()
                    if cost == 0:
                        return best_colors.tolist()
                        
                temp *= cooling
            else:
                # Reject, still cool down slightly? Or keep temp same?
                # Standard SA cools down regardless.
                temp *= cooling
                
            if time.time() - (start_time_global - time_left) > time_limit:
                break
                
        if best_cost == 0:
            return best_colors.tolist()
        return None

    # We need to pass time limits properly. 
    # The `time_left` in the while loop condition is static from the start of the function call.
    # We need to check actual elapsed time.
    
    # Let's refactor the time check.
    t_start = time.time()
    t_end = t_start + time_limit
    
    # Multiple restarts
    while time.time() < t_end:
        res = solve_single_restart(t_end - time.time())
        if res is not None:
            return res
            
    return None

# Global start time for the whole construct function
start_time_global = time.time()
