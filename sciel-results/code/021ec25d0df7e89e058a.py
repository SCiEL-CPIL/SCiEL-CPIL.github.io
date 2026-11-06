import numpy as np
import random
import time
from itertools import combinations

def fast_count_violations(n, neighbors):
    """
    Fast count of K5s and independent K5s using bit operations.
    """
    k5_count = 0
    bar_k5_count = 0
    
    # Generate combinations on the fly to avoid memory overhead
    for v0 in range(n - 4):
        for v1 in range(v0 + 1, n - 3):
            m01 = neighbors[v0] & (1 << v1)
            if not m01:
                # Not an edge, check for independent set later
                pass
            for v2 in range(v1 + 1, n - 2):
                m02 = neighbors[v0] & (1 << v2)
                m12 = neighbors[v1] & (1 << v2)
                if not (m01 and m02 and m12):
                    # Not a K3, skip K5 check but might be independent
                    for v3 in range(v2 + 1, n - 1):
                        m03 = neighbors[v0] & (1 << v3)
                        m13 = neighbors[v1] & (1 << v3)
                        m23 = neighbors[v2] & (1 << v3)
                        if not (m01 and m02 and m12 and m03 and m13 and m23):
                            for v4 in range(v3 + 1, n):
                                m04 = neighbors[v0] & (1 << v4)
                                m14 = neighbors[v1] & (1 << v4)
                                m24 = neighbors[v2] & (1 << v4)
                                m34 = neighbors[v3] & (1 << v4)
                                # Check K5
                                if (m01 and m02 and m12 and m03 and m13 and m23 and m04 and m14 and m24 and m34):
                                    k5_count += 1
                                # Check independent set
                                elif not (m01 or m02 or m12 or m03 or m13 or m23 or m04 or m14 or m24 or m34):
                                    bar_k5_count += 1
                        else:
                            # Already have K4, no need to check for independent
                            for v4 in range(v3 + 1, n):
                                m04 = neighbors[v0] & (1 << v4)
                                m14 = neighbors[v1] & (1 << v4)
                                m24 = neighbors[v2] & (1 << v4)
                                m34 = neighbors[v3] & (1 << v4)
                                if (m04 and m14 and m24 and m34):
                                    k5_count += 1
                else:
                    # Have K3, only check for K5
                    for v3 in range(v2 + 1, n - 1):
                        m03 = neighbors[v0] & (1 << v3)
                        m13 = neighbors[v1] & (1 << v3)
                        m23 = neighbors[v2] & (1 << v3)
                        if m03 and m13 and m23:
                            # Have K4
                            for v4 in range(v3 + 1, n):
                                m04 = neighbors[v0] & (1 << v4)
                                m14 = neighbors[v1] & (1 << v4)
                                m24 = neighbors[v2] & (1 << v4)
                                m34 = neighbors[v3] & (1 << v4)
                                if (m04 and m14 and m24 and m34):
                                    k5_count += 1
    return k5_count + bar_k5_count

def construct():
    start_time = time.time()
    time_limit = 290  # Leave 10s margin
    
    # Fallback: Paley 37
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    def build_circulant_edges(n, state):
        """Build edge list from circulant state."""
        edges = []
        for i in range(n):
            for j_idx, present in enumerate(state):
                if present:
                    jump = j_idx + 1
                    v1 = (i + jump) % n
                    if i < v1:
                        edges.append((i, v1))
                    elif v1 != (i - jump) % n:
                        v2 = (i - jump) % n
                        if v2 < i:
                            edges.append((v2, i))
        # Remove duplicates
        edge_set = set(edges)
        return list(edge_set)

    def build_neighbors_from_state(n, state):
        """Build neighbor bitmasks from circulant state."""
        neighbors = [0] * n
        for i in range(n):
            mask = 0
            for j_idx, present in enumerate(state):
                if present:
                    jump = j_idx + 1
                    v1 = (i + jump) % n
                    v2 = (i - jump) % n
                    mask |= (1 << v1)
                    if v1 != v2:
                        mask |= (1 << v2)
            neighbors[i] = mask
        return neighbors

    # Try to find a better graph using Circulant SA
    # Focus on n=42 first as it's the known maximum
    for n in [42, 41, 40, 39, 38]:
        if time.time() - start_time > time_limit:
            break
            
        half_n = n // 2
        best_energy = float('inf')
        best_state = None
        
        # Multiple restarts for n=42
        num_restarts = 5 if n == 42 else 3
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initial state: random
            current_state = [random.randint(0, 1) for _ in range(half_n)]
            
            current_neighbors = build_neighbors_from_state(n, current_state)
            current_energy = fast_count_violations(n, current_neighbors)
            
            if current_energy == 0:
                edges = build_circulant_edges(n, current_state)
                return n, edges
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.98
            steps = 1000  # More steps for better convergence
            
            for step in range(steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random jump to flip
                idx = random.randint(0, half_n - 1)
                new_state = current_state.copy()
                new_state[idx] = 1 - new_state[idx]
                
                new_neighbors = build_neighbors_from_state(n, new_state)
                new_energy = fast_count_violations(n, new_neighbors)
                
                delta = new_energy - current_energy
                
                if delta < 0 or (T > 0 and random.random() < pow(2.71828, -delta / T)):
                    current_state = new_state
                    current_neighbors = new_neighbors
                    current_energy = new_energy
                    
                    if current_energy == 0:
                        edges = build_circulant_edges(n, current_state)
                        return n, edges
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = current_state.copy()
                
                T *= cooling
                if T < T_min:
                    T = T_min

    # If no better graph found, return seed
    return get_paley_37()
