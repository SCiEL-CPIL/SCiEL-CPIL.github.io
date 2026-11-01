import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    D = 11
    
    # 1. Generate Weight-4 Vectors (Norm² = 4)
    # Each vector has 4 non-zero entries, each ±1
    # Represent as (support_tuple, sign_tuple)
    
    v4_list = []
    v4_supports = []  # For efficient conflict detection
    v4_signs = []
    
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(combo, signs):
                v[pos] = sign
            v4_list.append(v)
            v4_supports.append(combo)
            v4_signs.append(signs)
    
    V4 = np.array(v4_list, dtype=np.int8)
    N4 = len(V4)
    
    # 2. Build Conflict Graph for Weight-4 Vectors
    # Two weight-4 vectors conflict if they share exactly 3 positions AND have same signs on those 3 positions
    # (dot product = 3 > 2)
    
    # For each triple of positions, group vectors that contain that triple
    # Within each group, vectors conflict if they have same signs on the 3 shared positions
    
    triple_to_vectors = defaultdict(list)
    for idx in range(N4):
        support = v4_supports[idx]
        signs = v4_signs[idx]
        # Generate all 4 triples from this 4-element support
        for triple_idx in range(4):
            # Remove the triple_idx-th element to get a triple
            triple = tuple(support[i] for i in range(4) if i != triple_idx)
            # Get the signs on this triple
            triple_signs = tuple(signs[i] for i in range(4) if i != triple_idx)
            key = (triple, triple_signs)
            triple_to_vectors[key].append(idx)
    
    # Build adjacency: for each key, all vectors in that group conflict with each other
    # (since they share the same 3 positions with same signs)
    
    conflicts = np.zeros((N4, N4), dtype=bool)
    
    for key, vec_indices in triple_to_vectors.items():
        if len(vec_indices) < 2:
            continue
        for i in range(len(vec_indices)):
            for j in range(i+1, len(vec_indices)):
                conflicts[vec_indices[i], vec_indices[j]] = True
                conflicts[vec_indices[j], vec_indices[i]] = True
    
    # 3. Greedy MIS with Local Improvement
    rng = np.random.default_rng(42)
    
    best_selected = None
    best_k = 0
    
    def run_mis(conflicts, deadline, rng):
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = conflicts.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random pick from candidates
            u = candidates[rng.integers(len(candidates))]
            
            selected.append(u)
            
            # Remove u and its neighbors
            neighbors = np.where(conflicts[u])[0]
            to_remove = np.concatenate(([u], neighbors))
            
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                deg[remaining_alive] -= conflicts[remaining_alive][:, to_remove].sum(axis=1)
                
        return selected
    
    def local_improve(selected, conflicts, deadline, rng):
        """Try to add more vectors or swap to improve"""
        selected_set = set(selected)
        improved = list(selected)
        
        # Try to add any non-selected vector that doesn't conflict with selected
        for idx in range(len(conflicts)):
            if time.time() > deadline:
                break
            if idx in selected_set:
                continue
            # Check if idx conflicts with any selected
            if not np.any(conflicts[idx, list(selected_set)]):
                improved.append(idx)
                selected_set.add(idx)
        
        return improved
    
    # Run multiple restarts
    max_restarts = 100
    mis_deadline = deadline - 10.0
    
    for restart in range(max_restarts):
        if time.time() > mis_deadline:
            break
        
        selected = run_mis(conflicts, mis_deadline, rng)
        
        # Local improvement: try to add more vectors
        selected = local_improve(selected, conflicts, mis_deadline, rng)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
    
    if best_selected is None:
        # Fallback: just return all weight-4 vectors (may not be valid but better than nothing)
        return V4.astype(np.int64)
    
    result = V4[best_selected].astype(np.int64)
    return result
