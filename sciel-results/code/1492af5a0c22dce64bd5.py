import numpy as np
import itertools
import time
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s limit
    
    # D11 lattice minimal vectors: all vectors with exactly two non-zero entries, each ±1
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors
    # Each vector has norm² = 2
    
    # Generate all D11 minimal vectors
    vecs = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    
    n_vecs = len(vecs)  # 220
    
    # Build adjacency: two vectors are "compatible" (can both be in our set)
    # if their squared distance > max_norm² = 2
    # dist² = ||v||² + ||w||² - 2*v·w = 2 + 2 - 2*v·w = 4 - 2*v·w
    # Need: 4 - 2*v·w > 2  =>  v·w < 1  =>  v·w <= 0 (since integer)
    # So compatible iff dot product <= 0
    
    # Compute dot products
    vecs_arr = np.array(vecs, dtype=np.int64)
    dot_matrix = vecs_arr @ vecs_arr.T  # (220, 220)
    
    # Compatible if dot <= 0 (and not same vector, but dot=2 for same, so that's fine)
    # Actually for distinct vectors, dot can be -1, 0, or 1
    # Compatible iff dot <= 0
    compat = (dot_matrix <= 0).astype(np.int8)
    np.fill_diagonal(compat, 0)
    
    # Maximum clique in the compatibility graph
    # Use a greedy + local search approach
    
    best_clique = set()
    best_size = 0
    
    # Greedy initialization with different orderings
    def greedy_clique(order):
        clique = set()
        candidates = set(range(n_vecs))
        for node in order:
            if node in candidates:
                # Check if node is compatible with all in clique
                if all(compat[node, c] for c in clique):
                    clique.add(node)
                    # Remove incompatible candidates
                    incompatible = set(np.where(compat[node] == 0)[0]) - clique
                    candidates -= incompatible
        return clique
    
    # Local search: try to add nodes, and if stuck, do swaps
    def local_search(clique, deadline):
        clique = set(clique)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add a node
            candidates = [i for i in range(n_vecs) if i not in clique]
            for c in candidates:
                if all(compat[c, x] for x in clique):
                    clique.add(c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add one
            for remove_node in list(clique):
                if time.time() > deadline:
                    break
                remaining = clique - {remove_node}
                for add_node in range(n_vecs):
                    if add_node in remaining:
                        continue
                    if all(compat[add_node, x] for x in remaining):
                        clique.discard(remove_node)
                        clique.add(add_node)
                        improved = True
                        break
                if improved:
                    break
        return clique
    
    # Multiple random restarts
    rng = np.random.default_rng(12345)
    
    # Initial orderings
    # Sort by degree (ascending) - prefer nodes with fewer conflicts
    degrees = np.sum(1 - compat, axis=1)
    orderings = [np.argsort(degrees), np.argsort(-degrees)]
    
    # Add random orderings
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        orderings.append(rng.permutation(n_vecs))
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        
        clique = greedy_clique(order)
        clique = local_search(clique, deadline - 20)
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = set(clique)
    
    # If we have time, do more ILS on the best found
    if time.time() < deadline - 30:
        current_best = set(best_clique)
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            
            # Perturb: remove 3-5 random nodes
            if len(current_best) > 10:
                n_remove = min(5, len(current_best) // 2)
                to_remove = rng.choice(sorted(current_best), size=n_remove, replace=False)
                for r in to_remove:
                    current_best.discard(r)
            
            # Local search
            current_best = local_search(current_best, deadline - 20)
            
            if len(current_best) > best_size:
                best_size = len(current_best)
                best_clique = set(current_best)
    
    # Build the final points array
    points = [vecs_arr[i] for i in sorted(best_clique)]
    
    # Ensure at least 2 points
    if len(points) < 2:
        # Fallback: just use the first two compatible vectors
        points = [vecs_arr[0], vecs_arr[1]]
    
    points_array = np.array(points, dtype=np.int64)
    points_array = np.around(points_array).astype(np.int64)
    
    return points_array
