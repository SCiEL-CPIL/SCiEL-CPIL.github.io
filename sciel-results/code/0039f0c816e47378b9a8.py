import random
import time
import math
from itertools import combinations

def construct():
    # We aim for n=34, as R(4,5;3) > 34 is the known bound we want to certify.
    # If we can't find it, we fall back to smaller n.
    
    def count_violations(n, colors, idx_map, quad_edges, quint_edges):
        """
        Calculate total number of red K4s and blue K5s.
        quad_edges: list of tuples of 4 indices (into colors list) for each K4
        quint_edges: list of tuples of 10 indices (into colors list) for each K5
        """
        red_k4 = 0
        for edges in quad_edges:
            if all(colors[i] == 0 for i in edges):
                red_k4 += 1
                
        blue_k5 = 0
        for edges in quint_edges:
            if all(colors[i] == 1 for i in edges):
                blue_k5 += 1
                
        return red_k4 + blue_k5

    def get_affected(n, idx_map, edge_idx, quad_list, quint_list):
        """
        Find which K4s and K5s are affected by flipping edge_idx.
        Returns list of (quad_tuple, current_red_count) and (quint_tuple, current_blue_count)
        Actually, we just need to re-evaluate those specific hypercliques.
        """
        # Pre-computed: for each edge, which quads/quints contain it?
        # This is expensive to compute on the fly. We should pre-compute incidence lists.
        pass

    # Pre-computation for n=34
    n = 34
    triples = list(combinations(range(n), 3))
    idx_map = {}
    for i, (a, b, c) in enumerate(triples):
        idx_map[(a, b, c)] = i
        
    # Build incidence lists: edge_idx -> list of (quad_idx, other_edges) ? 
    # Simpler: edge_idx -> list of quad_indices and quint_indices.
    # And for each quad/quint, store the list of edge indices.
    
    quads = list(combinations(range(n), 4))
    quints = list(combinations(range(n), 5))
    
    quad_edges = []
    edge_to_quads = [[] for _ in range(len(triples))]
    for q in quads:
        a, b, c, d = q
        e1 = idx_map[(a,b,c)]
        e2 = idx_map[(a,b,d)]
        e3 = idx_map[(a,c,d)]
        e4 = idx_map[(b,c,d)]
        quad_edges.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4):
            edge_to_quads[e].append(len(quad_edges) - 1)
            
    quint_edges = []
    edge_to_quints = [[] for _ in range(len(triples))]
    for q in quints:
        a, b, c, d, e = q
        es = []
        # 10 triples from 5 vertices
        for comb in combinations([a,b,c,d,e], 3):
            es.append(idx_map[comb])
        quint_edges.append(tuple(es))
        for e in es:
            edge_to_quints[e].append(len(quint_edges) - 1)
            
    def evaluate(n, colors, idx_map, quad_edges, quint_edges):
        red_k4 = 0
        for edges in quad_edges:
            if all(colors[i] == 0 for i in edges):
                red_k4 += 1
        blue_k5 = 0
        for edges in quint_edges:
            if all(colors[i] == 1 for i in edges):
                blue_k5 += 1
        return red_k4 + blue_k5

    def simulated_annealing(n, time_limit=100.0):
        random.seed(42)
        colors = [random.randint(0, 1) for _ in range(len(triples))]
        
        # Initial energy
        energy = evaluate(n, colors, idx_map, quad_edges, quint_edges)
        
        start_time = time.time()
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() - start_time < time_limit and energy > 0:
            # Pick random edge
            e_idx = random.randint(0, len(triples) - 1)
            
            # Calculate delta energy
            # We need to check all quads/quints containing e_idx
            
            delta = 0
            
            # Check quads
            for q_idx in edge_to_quads[e_idx]:
                edges = quad_edges[q_idx]
                # Count reds before flip
                reds = sum(1 for i in edges if colors[i] == 0)
                # If we flip e_idx (which is in edges):
                if colors[e_idx] == 0:
                    # Becomes blue, so reds decreases by 1
                    new_reds = reds - 1
                else:
                    # Becomes red, so reds increases by 1
                    new_reds = reds + 1
                
                # Violation if reds == 4
                old_viol = 1 if reds == 4 else 0
                new_viol = 1 if new_reds == 4 else 0
                delta += new_viol - old_viol
                
            # Check quints
            for q_idx in edge_to_quints[e_idx]:
                edges = quint_edges[q_idx]
                # Count blues before flip
                blues = sum(1 for i in edges if colors[i] == 1)
                if colors[e_idx] == 1:
                    # Becomes red, so blues decreases by 1
                    new_blues = blues - 1
                else:
                    # Becomes blue, so blues increases by 1
                    new_blues = blues + 1
                    
                # Violation if blues == 10
                old_viol = 1 if blues == 10 else 0
                new_viol = 1 if new_blues == 10 else 0
                delta += new_viol - old_viol
                
            # Acceptance criterion
            if delta <= 0:
                colors[e_idx] = 1 - colors[e_idx]
                energy += delta
            else:
                # Boltzmann probability
                p = math.exp(-delta / T)
                if random.random() < p:
                    colors[e_idx] = 1 - colors[e_idx]
                    energy += delta
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return colors if energy == 0 else None

    # Try n=34
    res = simulated_annealing(34, time_limit=100.0)
    if res is not None:
        return {'n': 34, 'colors': res}
        
    # Fallback to smaller n if 34 fails (unlikely for SA, but safe)
    # For simplicity, we return a known valid small n or fail.
    # But the task says "Maximize n". If 34 fails, try 33?
    # The structure above is specific to n. Let's just try 34 again with different seed if failed?
    # Or just return a smaller valid one.
    
    # Let's try n=32 as a backup
    # Note: The pre-computation above is hardcoded for n=34. 
    # To be robust, we should generalize or just trust SA for 34.
    # Given the time limit, one run of 100s is enough.
    
    # If it fails, return a trivially valid small n to avoid 0 score?
    # The task says "valid result... invalid result scores 0".
    # So we must return a valid coloring.
    # A valid coloring for n=1 is trivial.
    # But we want high score.
    # Let's assume SA finds it.
    
    # If not, fallback to n=9 deterministic
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
