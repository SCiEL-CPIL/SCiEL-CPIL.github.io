import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Generate all integer vectors with norm squared = 4
    # Type 1: +/- 2 e_i (11 * 2 = 22 vectors)
    # Type 2: Four +/- 1s (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    vectors = []
    
    # Type 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # For norm-squared = 4 vectors, dist_sq[i,j] = 4 + 4 - 2*dots[i,j] = 8 - 2*dots[i,j]
    # We need dist_sq >= 4 (max norm is 2, so squared max norm is 4)
    # 8 - 2*dots[i,j] >= 4  =>  4 >= 2*dots[i,j]  =>  dots[i,j] <= 2
    # Conflict if dots[i,j] > 2
    
    conflicts_mask = dots > 2
    np.fill_diagonal(conflicts_mask, False)
    
    # Precompute neighbor lists for fast conflict checking
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        neigh = np.where(conflicts_mask[i])[0]
        neighbors[i] = set(neigh.tolist())
    
    # --- Strategy: Multiple greedy minimum-degree with random restarts ---
    # This is a structural change from simulated annealing to greedy with restarts
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_size = 0
    best_indices = []
    
    def greedy_min_degree():
        """Greedy independent set using minimum degree heuristic."""
        available = set(range(N))
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree in the remaining conflict graph
            # Degree = number of neighbors still in available
            min_deg = float('inf')
            best_v = None
            for v in available:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
        
        return selected
    
    def greedy_random():
        """Greedy independent set with random ordering."""
        available = set(range(N))
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Pick random vertex from available
            v = rng.choice(list(available))
            selected.append(v)
            available.discard(v)
            for n in neighbors[v]:
                if n in available:
                    available.discard(n)
        
        return selected
    
    def greedy_weighted():
        """Greedy with weighted scoring: prefer vertices with fewer conflicts."""
        available = set(range(N))
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Score each vertex: negative of its degree (prefer low degree)
            # Add small random noise to break ties
            scores = {}
            for v in available:
                deg = len(neighbors[v] & available)
                scores[v] = -deg + rng.random() * 0.1
            
            best_v = max(scores, key=scores.get)
            selected.append(best_v)
            available.discard(best_v)
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
        
        return selected
    
    # Run multiple restarts with different strategies
    strategies = [greedy_min_degree, greedy_random, greedy_weighted]
    
    num_restarts = 200  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Cycle through strategies
        strategy = strategies[restart % len(strategies)]
        
        selected = strategy()
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = list(selected)
            
            # If we found a large set, do local improvement
            if best_size > 200:
                # Try to add more vertices
                current_set = set(best_indices)
                improved = True
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    for v in range(N):
                        if v not in current_set:
                            if all(n not in current_set for n in neighbors[v]):
                                current_set.add(v)
                                improved = True
                    if improved:
                        best_indices = list(current_set)
                        best_size = len(best_indices)
                
                # Try to swap: remove one vertex, add two if possible
                improved = True
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    for v in list(best_indices):
                        test_set = set(best_indices)
                        test_set.discard(v)
                        # Try to add vertices that conflict with v
                        candidates = [n for n in neighbors[v] if n not in test_set]
                        added = 0
                        for c in candidates:
                            if all(n not in test_set for n in neighbors[c]):
                                test_set.add(c)
                                added += 1
                        if added > 1:
                            best_indices = list(test_set)
                            best_size = len(best_indices)
                            improved = True
                            break
    
    # Final verification and cleanup
    if best_indices:
        final_set = set(best_indices)
        # Remove any remaining conflicts
        cleaned = []
        for v in best_indices:
            if all(n not in set(cleaned) for n in neighbors[v]):
                cleaned.append(v)
        if len(cleaned) > 0:
            best_indices = cleaned
            best_size = len(cleaned)
            
    if best_size == 0:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
        
    return vectors[best_indices]

# EVOLVE-BLOCK-END
