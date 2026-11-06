# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def build_index(n):
        """Build mapping from triple to index and precompute quad/quint triple lists."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        
        # Precompute which triples are in each quad (for red K4 check)
        quad_triples = {}
        for quad in combinations(range(n), 4):
            quad_triples[quad] = [idx[tri] for tri in combinations(quad, 3)]
        
        # Precompute which triples are in each quint (for blue K5 check)
        quint_triples = {}
        for quint in combinations(range(n), 5):
            quint_triples[quint] = [idx[tri] for tri in combinations(quint, 3)]
        
        return idx, quad_triples, quint_triples
    
    def find_violating_triples_fast(colors, n, quad_triples, quint_triples):
        """Find triples that are part of violations using precomputed lists."""
        violating = set()
        
        # Red K4^3 violations
        for quad, tri_indices in quad_triples.items():
            all_red = all(colors[i] == 0 for i in tri_indices)
            if all_red:
                violating.update(tri_indices)
        
        # Blue K5^3 violations
        for quint, tri_indices in quint_triples.items():
            all_blue = all(colors[i] == 1 for i in tri_indices)
            if all_blue:
                violating.update(tri_indices)
        
        return violating
    
    def local_search(n, p_red, max_iters=10000):
        """Run local search to find valid coloring."""
        idx, quad_triples, quint_triples = build_index(n)
        n_triples = len(idx)
        
        for restart in range(100):
            if time.time() > deadline - 3:
                break
            
            # Random initial coloring with bias
            colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    return None
                
                violating = find_violating_triples_fast(colors, n, quad_triples, quint_triples)
                if not violating:
                    return colors.tolist()
                
                # Pick a random violating triple and flip it
                tri_idx = random.choice(list(violating))
                colors[tri_idx] = 1 - colors[tri_idx]
        
        return None
    
    # Try increasing n from small to large
    best_n = 1
    best_colors = [0]  # trivial valid coloring for n=1
    
    for n in range(2, 35):
        if time.time() > deadline - 5:
            break
        
        # Try multiple bias values
        for p in [0.5, 0.3, 0.4, 0.6, 0.7]:
            if time.time() > deadline - 5:
                break
            
            result = local_search(n, p, max_iters=5000)
            if result is not None:
                best_n = n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
