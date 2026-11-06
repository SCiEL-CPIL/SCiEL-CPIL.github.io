import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_incidence(n, triples):
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        quad_triples = []
        for quad in quads:
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
            
        quint_triples = []
        for quint in quints:
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)
            
        # Inverse
        triple_to_quads = [[] for _ in range(len(triples))]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)
                
        triple_to_quints = [[] for _ in range(len(triples))]
        for kidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_quints[t].append(kidx)
                
        return quad_triples, quint_triples, triple_to_quads, triple_to_quints

    def solve_n(n, time_limit, seed):
        if time.time() > deadline - 5:
            return None
            
        triples = get_triples(n)
        M = len(triples)
        if M == 0:
            return []
            
        quad_triples, quint_triples, triple_to_quads, triple_to_quints = build_incidence(n, triples)
        n_quads = len(quad_triples)
        n_quints = len(quint_triples)
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        # Use a random initialization
        colors = rng.randint(0, 2, size=M).astype(np.int8)
        
        red_counts = np.zeros(n_quads, dtype=np.int32)
        blue_counts = np.zeros(n_quints, dtype=np.int32)
        
        for t in range(M):
            if colors[t] == 0:
                for q in triple_to_quads[t]:
                    red_counts[q] += 1
            else:
                for k in triple_to_quints[t]:
                    blue_counts[k] += 1
                    
        energy = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
        
        if energy == 0:
            return colors.tolist()
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        
        # Precompute lists for speed
        # We need to pick a random triple to flip.
        
        while time.time() < start_time + time_limit:
            if energy == 0:
                return colors.tolist()
                
            # Pick random triple
            t = rng.randint(M)
            old_color = colors[t]
            
            delta = 0
            if old_color == 0:
                # Red -> Blue
                for q in triple_to_quads[t]:
                    if red_counts[q] == 4:
                        delta -= 1
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 9:
                        delta += 1
            else:
                # Blue -> Red
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 10:
                        delta -= 1
                for q in triple_to_quads[t]:
                    if red_counts[q] == 3:
                        delta += 1
                        
            # Acceptance
            if delta < 0:
                accept = True
            else:
                # Boltzmann
                if T > min_T:
                    accept = rng.rand() < np.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                # Apply flip
                if old_color == 0:
                    for q in triple_to_quads[t]:
                        red_counts[q] -= 1
                    for k in triple_to_quints[t]:
                        blue_counts[k] += 1
                    colors[t] = 1
                else:
                    for k in triple_to_quints[t]:
                        blue_counts[k] -= 1
                    for q in triple_to_quads[t]:
                        red_counts[q] += 1
                    colors[t] = 0
                energy += delta
                T *= cooling
                
        return None

    best_n = 1
    best_colors = []
    
    # Try n=34
    res = solve_n(34, 90.0, 42)
    if res is not None:
        best_n = 34
        best_colors = res
    else:
        # Try smaller n if time permits
        if time.time() < deadline - 10:
            # Try n=20, which is much easier
            res = solve_n(20, 10.0, 43)
            if res is not None:
                best_n = 20
                best_colors = res
            else:
                # Fallback to n=1
                best_n = 1
                best_colors = []
                
    return {'n': best_n, 'colors': best_colors}
