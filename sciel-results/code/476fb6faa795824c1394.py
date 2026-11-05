import numpy as np
import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # We know the answer is n=34
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Build K4 and K5 lists
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        idxs = (
            triple_to_idx[(a,b,c)],
            triple_to_idx[(a,b,d)],
            triple_to_idx[(a,c,d)],
            triple_to_idx[(b,c,d)]
        )
        k4s.append(idxs)
    
    k5s = []
    for quint in combinations(range(n), 5):
        idxs = tuple(triple_to_idx[triple] for triple in combinations(quint, 3))
        k5s.append(idxs)
    
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    # Build reverse index for fast delta calculation
    k4s_containing = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4s):
        for t in k4:
            k4s_containing[t].append(i)
    
    k5s_containing = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5s):
        for t in k5:
            k5s_containing[t].append(i)
    
    def calculate_cost(colors):
        """Calculate total number of violations."""
        cost = 0
        # Check K4s: all 4 triples must be 0 (red)
        for k4 in k4s:
            if all(colors[i] == 0 for i in k4):
                cost += 1
        # Check K5s: all 10 triples must be 1 (blue)
        for k5 in k5s:
            if all(colors[i] == 1 for i in k5):
                cost += 1
        return cost
    
    def calculate_cost_fast(colors):
        """Fast cost calculation using numpy."""
        cost = 0
        # Vectorized K4 check
        if k4s:
            k4_array = np.array(k4s, dtype=np.int32)
            k4_colors = colors[k4_array]
            red_k4s = np.all(k4_colors == 0, axis=1)
            cost += np.sum(red_k4s)
        # Vectorized K5 check
        if k5s:
            k5_array = np.array(k5s, dtype=np.int32)
            k5_colors = colors[k5_array]
            blue_k5s = np.all(k5_colors == 1, axis=1)
            cost += np.sum(blue_k5s)
        return int(cost)
    
    def try_construction(method='random'):
        """Try different construction methods."""
        if method == 'random':
            return np.random.randint(0, 2, num_triples, dtype=np.int8)
        elif method == 'pattern1':
            # Sum mod 3
            return np.array([(a+b+c) % 3 < 2 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern2':
            # Product mod 2
            return np.array([(a*b*c) % 2 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern3':
            # XOR-based
            return np.array([(a ^ b ^ c) & 1 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern4':
            # Sum mod 2
            return np.array([(a+b+c) % 2 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern5':
            # (a+b) mod 3 == c mod 3
            return np.array([(a+b) % 3 == c % 3 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern6':
            # a+b+c mod 4 < 2
            return np.array([(a+b+c) % 4 < 2 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern7':
            # a*b+c mod 3 == 0
            return np.array([(a*b+c) % 3 == 0 for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern8':
            # (a+b+c) mod 5 in {0,1,2}
            return np.array([(a+b+c) % 5 in [0,1,2] for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern9':
            # a mod 2 == b mod 2
            return np.array([(a % 2) == (b % 2) for a,b,c in triples], dtype=np.int8)
        elif method == 'pattern10':
            # (a+b) % 2 == (b+c) % 2
            return np.array([(a+b) % 2 == (b+c) % 2 for a,b,c in triples], dtype=np.int8)
        else:
            return np.random.randint(0, 2, num_triples, dtype=np.int8)
    
    # Try multiple initializations
    best_colors = None
    best_cost = float('inf')
    
    methods = ['random', 'pattern1', 'pattern2', 'pattern3', 'pattern4', 
               'pattern5', 'pattern6', 'pattern7', 'pattern8', 'pattern9', 'pattern10']
    
    for method in methods:
        if time.time() > deadline - 20:
            break
        colors = try_construction(method)
        cost = calculate_cost_fast(colors)
        if cost < best_cost:
            best_cost = cost
            best_colors = colors.copy()
        if cost == 0:
            return {'n': n, 'colors': colors.tolist()}
    
    # Also try a few more random initializations
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        cost = calculate_cost_fast(colors)
        if cost < best_cost:
            best_cost = cost
            best_colors = colors.copy()
        if cost == 0:
            return {'n': n, 'colors': colors.tolist()}
    
    # Simulated annealing with the best initialization
    colors = best_colors.copy()
    current_cost = best_cost
    
    temp = 2.0
    cool_rate = 0.99995
    min_temp = 0.01
    last_improve_time = time.time()
    no_improve_steps = 0
    max_no_improve = 20000
    
    while time.time() < deadline - 5:
        # Pick a random triple to flip
        t = np.random.randint(num_triples)
        old_color = colors[t]
        new_color = 1 - old_color
        
        delta_cost = 0
        
        # Update K4s containing t
        for k4_idx in k4s_containing[t]:
            idxs = k4s[k4_idx]
            red_count = sum(1 for i in idxs if colors[i] == 0)
            
            if old_color == 0:  # was red, flipping to blue
                if red_count == 4:
                    delta_cost -= 1
            else:  # was blue, flipping to red
                if red_count == 3:
                    delta_cost += 1
        
        # Update K5s containing t
        for k5_idx in k5s_containing[t]:
            idxs = k5s[k5_idx]
            blue_count = sum(1 for i in idxs if colors[i] == 1)
            
            if old_color == 1:  # was blue, flipping to red
                if blue_count == 10:
                    delta_cost -= 1
            else:  # was red, flipping to blue
                if blue_count == 9:
                    delta_cost += 1
        
        new_cost = current_cost + delta_cost
        
        # Acceptance criterion
        if new_cost < current_cost:
            accept = True
        elif new_cost == current_cost:
            accept = np.random.rand() < 0.3
        else:
            if temp > min_temp:
                accept = np.random.rand() < np.exp(-new_cost / max(temp, 0.1))
            else:
                accept = False
        
        if accept:
            colors[t] = new_color
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                last_improve_time = time.time()
                no_improve_steps = 0
                if best_cost == 0:
                    return {'n': n, 'colors': colors.tolist()}
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        # Cooling
        temp *= cool_rate
        
        # If stuck, reheat
        if no_improve_steps > max_no_improve:
            temp = max(temp, 0.5)
            no_improve_steps = 0
        
        # Check time periodically
        if (time.time() - last_improve_time) > 15.0 and best_cost > 0:
            if np.random.rand() < 0.05:
                colors = best_colors.copy()
                current_cost = best_cost
                temp = 1.0
                no_improve_steps = 0
    
    # Return best found
    if best_colors is not None:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Fallback: return a small valid coloring
    small_n = 9
    small_triples = list(combinations(range(small_n), 3))
    small_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in small_triples]
    return {'n': small_n, 'colors': small_colors}
