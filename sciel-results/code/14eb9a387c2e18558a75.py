import numpy as np
import random
import time
from itertools import combinations

def count_violations_fast(n, neighbors, combos):
    k5_count = 0
    bar_k5_count = 0
    
    for c in combos:
        v0, v1, v2, v3, v4 = c
        m0 = neighbors[v0]
        m1 = neighbors[v1]
        m2 = neighbors[v2]
        m3 = neighbors[v3]
        m4 = neighbors[v4]
        
        if (m0 & (1 << v1)) and (m0 & (1 << v2)) and (m0 & (1 << v3)) and (m0 & (1 << v4)) and \
           (m1 & (1 << v2)) and (m1 & (1 << v3)) and (m1 & (1 << v4)) and \
           (m2 & (1 << v3)) and (m2 & (1 << v4)) and \
           (m3 & (1 << v4)):
            k5_count += 1
        else:
            if not (m0 & (1 << v1)) and not (m0 & (1 << v2)) and not (m0 & (1 << v3)) and not (m0 & (1 << v4)) and \
               not (m1 & (1 << v2)) and not (m1 & (1 << v3)) and not (m1 & (1 << v4)) and \
               not (m2 & (1 << v3)) and not (m2 & (1 << v4)) and \
               not (m3 & (1 << v4)):
                bar_k5_count += 1
    return k5_count + bar_k5_count

def construct():
    start_time = time.time()
    time_limit = 295
    
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Try to find a 42-vertex graph
    n = 42
    combos = list(combinations(range(n), 5))
    
    best_energy = float('inf')
    best_neighbors = None
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        # Initialize random graph
        neighbors = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
                    
        current_energy = count_violations_fast(n, neighbors, combos)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_neighbors = neighbors[:]
            
        if current_energy == 0:
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if neighbors[i] & (1 << j)]
            return n, edges

        T = 5.0
        T_min = 0.1
        cooling = 0.995
        steps = 0
        max_steps = 500 # Limit steps per restart to ensure we try multiple restarts
        
        while steps < max_steps and time.time() - start_time < time_limit:
            steps += 1
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            neighbors[i] ^= (1 << j)
            neighbors[j] ^= (1 << i)
            
            new_energy = count_violations_fast(n, neighbors, combos)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_neighbors = neighbors[:]
                
            if new_energy == 0:
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if neighbors[i] & (1 << j)]
                return n, edges

            delta = new_energy - current_energy
            if delta < 0 or (T > 0 and random.random() < pow(2.71828, -delta / T)):
                current_energy = new_energy
            else:
                neighbors[i] ^= (1 << j)
                neighbors[j] ^= (1 << i)
                
            T *= cooling
            if T < T_min:
                break

    if best_energy == 0 and best_neighbors:
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_neighbors[i] & (1 << j)]
        return n, edges
        
    # Fallback
    return get_paley_37()
