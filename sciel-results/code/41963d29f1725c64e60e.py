# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    def build_triple_index(n):
        """Build mapping from triple to lexicographic index."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        return idx, i
    
    def fast_violation_count(colors, n, triple_idx, n_triples):
        """
        Count red K4^3 and blue K5^3 violations using vectorized ops.
        Returns (red_violations, blue_violations, violating_triple_indices_set)
        """
        # Precompute: for each 4-set, which triple indices are in it
        # For red K4^3: need all 4 triples in the 4-set to be color 0
        # For blue K5^3: need all 10 triples in the 5-set to be color 1
        
        red_viol = 0
        blue_viol = 0
        violating = set()
        
        # Red K4^3 check
        for quad in combinations(range(n), 4):
            tris = list(combinations(quad, 3))
            all_red = True
            for tri in tris:
                if colors[triple_idx[tri]] != 0:
                    all_red = False
                    break
            if all_red:
                red_viol += 1
                for tri in tris:
                    violating.add(triple_idx[tri])
        
        # Blue K5^3 check
        for quint in combinations(range(n), 5):
            tris = list(combinations(quint, 3))
            all_blue = True
            for tri in tris:
                if colors[triple_idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_viol += 1
                for tri in tris:
                    violating.add(triple_idx[tri])
        
        return red_viol, blue_viol, violating
    
    def simulated_annealing(n, triple_idx, n_triples, max_time):
        """Simulated annealing with violation-based energy."""
        # Initialize with random coloring
        colors = np.random.randint(0, 2, size=n_triples)
        
        # Compute initial energy
        red_v, blue_v, violating = fast_violation_count(colors, n, triple_idx, n_triples)
        current_energy = red_v + blue_v
        
        if current_energy == 0:
            return colors.tolist()
        
        best_energy = current_energy
        best_colors = colors.copy()
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        t_start = time.time()
        
        while T > T_min and time.time() < max_time:
            # Pick a random triple to flip
            tri_idx = np.random.randint(0, n_triples)
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            colors[tri_idx] = new_color
            
            # Recompute energy (expensive but necessary)
            red_v2, blue_v2, violating2 = fast_violation_count(colors, n, triple_idx, n_triples)
            new_energy = red_v2 + blue_v2
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return best_colors.tolist()
            else:
                colors[tri_idx] = old_color
            
            T *= cooling
        
        return best_colors.tolist() if best_energy == 0 else None
    
    def greedy_construct(n, triple_idx, n_triples):
        """
        Greedy construction: build coloring vertex by vertex.
        For each new vertex, assign colors to its new triples to minimize violations.
        """
        colors = np.zeros(n_triples, dtype=np.int8)
        
        # Process vertices in order, adding triples as we go
        # For vertex v, new triples are (a, b, v) for a < b < v
        for v in range(n):
            # Get all triples involving v where v is the largest
            new_triples = []
            for a in range(v):
                for b in range(a+1, v):
                    new_triples.append((a, b, v))
            
            if not new_triples:
                continue
            
            # Try both colors for each new triple, pick the one that minimizes violations
            for tri in new_triples:
                idx = triple_idx[tri]
                # Try color 0
                colors[idx] = 0
                r0, b0, _ = fast_violation_count(colors, n, triple_idx, n_triples)
                e0 = r0 + b0
                
                # Try color 1
                colors[idx] = 1
                r1, b1, _ = fast_violation_count(colors, n, triple_idx, n_triples)
                e1 = r1 + b1
                
                if e0 <= e1:
                    colors[idx] = 0
                else:
                    colors[idx] = 1
        
        # Check if valid
        r, b, viol = fast_violation_count(colors, n, triple_idx, n_triples)
        if r + b == 0:
            return colors.tolist()
        return None
    
    # Try increasing n
    best_n = 1
    best_colors = [0]
    
    for n in range(2, 35):
        if time.time() > deadline - 15:
            break
        
        triple_idx, n_triples = build_triple_index(n)
        
        # First try greedy construction
        result = greedy_construct(n, triple_idx, n_triples)
        if result is not None:
            best_n = n
            best_colors = result
            continue
        
        # Then try simulated annealing with limited time
        remaining = deadline - time.time() - 10
        if remaining > 2:
            result = simulated_annealing(n, triple_idx, n_triples, time.time() + remaining)
            if result is not None:
                best_n = n
                best_colors = result
                continue
        
        # If greedy+SA failed, try a few random restarts with SA
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            remaining = deadline - time.time() - 5
            if remaining < 1:
                break
            result = simulated_annealing(n, triple_idx, n_triples, time.time() + remaining)
            if result is not None:
                best_n = n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
