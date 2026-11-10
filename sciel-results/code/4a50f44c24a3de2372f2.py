import numpy as np
import time
import random
from itertools import combinations
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Precompute all 4-subsets
    four_subsets = list(combinations(range(d), 4))
    n_four = len(four_subsets)
    
    # Build compatibility graph
    # Two 4-subsets are compatible if they share at most 2 elements
    # This means their intersection size is <= 2
    compat = [set() for _ in range(n_four)]
    for i in range(n_four):
        si = set(four_subsets[i])
        for j in range(i + 1, n_four):
            sj = set(four_subsets[j])
            if len(si & sj) <= 2:
                compat[i].add(j)
                compat[j].add(i)
    
    best_clique = []
    best_size = 0
    
    # Tomita's algorithm for Max Clique
    # We use a recursive branch and bound
    # P is the set of candidate vertices
    # X is the set of vertices already processed (for some bounds, but simple bound is |P|)
    
    def max_clique(P, current_clique):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline - 10:
            return
            
        if not P:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
            
        # Simple upper bound: current size + |P|
        if len(current_clique) + len(P) <= best_size:
            return
            
        # Heuristic: pick vertex with max degree in P
        # Or just pick one and branch
        # For small graphs, picking the vertex with highest degree in P is good
        
        # To speed up, we can order P
        # But let's try a simple recursive approach first
        
        # Select a vertex u from P
        # To improve pruning, we can sort P by degree in P
        # But sorting at every node is expensive.
        # Let's try picking the vertex with the highest degree in P.
        
        max_deg = -1
        chosen = -1
        for v in P:
            deg = len(compat[v] & P)
            if deg > max_deg:
                max_deg = deg
                chosen = v
                
        if chosen == -1:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return

        # Branch 1: Include chosen
        # New P = P \ {chosen} \ N(chosen)
        # Wait, for Clique, if we include chosen, we can only include its neighbors.
        # So new P = P ∩ N(chosen)
        new_P = P.intersection(compat[chosen])
        current_clique.append(chosen)
        max_clique(new_P, current_clique)
        current_clique.pop()
        
        # Branch 2: Exclude chosen
        # New P = P \ {chosen}
        P.remove(chosen)
        max_clique(P, current_clique)
        
    # Start with all vertices
    all_vertices = set(range(n_four))
    max_clique(all_vertices, [])
    
    # If B&B is slow or finds nothing, use greedy fallback
    if best_size < 20:
        rng = np.random.default_rng(42)
        for _ in range(1000):
            if time.time() > deadline - 5: break
            order = rng.permutation(n_four)
            used = set()
            clique = []
            for idx in order:
                if all(idx not in compat[u] for u in clique): # This is O(k) check, slow
                    # Check if idx is compatible with all in clique
                    ok = True
                    for u in clique:
                        if idx not in compat[u]:
                            ok = False
                            break
                    if ok:
                        clique.append(idx)
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
            if best_size >= 41:
                break

    # Construct points
    points = []
    for idx in best_clique:
        support = four_subsets[idx]
        # Generate all 16 sign combinations
        for signs in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, pos in enumerate(support):
                if (signs >> k) & 1:
                    v[pos] = 1
                else:
                    v[pos] = -1
            points.append(v)
            
    if len(points) < 2:
        # Fallback
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
