"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # 1. Generate Candidates: Weight-4 vectors
    # C(11, 4) * 16 = 330 * 16 = 5280
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    # Conflict if dot product > 2
    # G = cand_arr @ cand_arr.T
    # Using int8 for memory and speed
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists
    # We know degree is 56, so we can pre-allocate or just use lists
    neighbors = [[] for _ in range(n_cands)]
    # Vectorized conflict detection is faster for building
    # G is symmetric. 
    for i in range(n_cands):
        # Find j such that G[i,j] > 2 and j != i
        # np.where is fast
        js = np.where(G[i] > 2)[0]
        # Exclude self
        js = js[js != i]
        neighbors[i] = js.tolist()
        
    # 3. Improved Greedy Independent Set with local improvement
    
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_fast(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected = []
        
        while True:
            if time.time() > deadline - 2:
                break
                
            # Find active nodes
            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break
                
            # Find min degree among active
            min_deg = np.min(deg[active_nodes])
            
            # Get all active nodes with min degree
            mask = (deg == min_deg) & active
            candidates = np.where(mask)[0]
            
            # Pick randomly
            u = rng.choice(candidates)
            
            selected.append(u)
            active[u] = False
            
            # Remove neighbors
            for v in neighbors[u]:
                if active[v]:
                    active[v] = False
                    for w in neighbors[v]:
                        if active[w]:
                            deg[w] -= 1
                            
        return selected

    def run_greedy_weighted(seed_offset=0):
        """Greedy with weighted selection using boolean array."""
        if time.time() > deadline - 5:
            return None
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected = []
        
        while True:
            if time.time() > deadline - 2:
                break
                
            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break
                
            degs = deg[active_nodes]
            min_d = np.min(degs)
            
            # Consider nodes with degree <= min_d + 1
            mask = (degs <= min_d + 1)
            candidate_indices = active_nodes[mask]
            
            if len(candidate_indices) == 0:
                candidate_indices = active_nodes
                
            u = rng.choice(candidate_indices)
            
            selected.append(u)
            active[u] = False
            
            for v in neighbors[u]:
                if active[v]:
                    active[v] = False
                    for w in neighbors[v]:
                        if active[w]:
                            deg[w] -= 1
                            
        return selected
    
    def local_improve(selected, conflict_set):
        """Try to add more vectors by checking if any removed vector can be added."""
        if time.time() > deadline - 3:
            return selected, conflict_set
            
        # Build conflict set from selected
        selected_set = set(selected)
        
        # Find all vectors not in selected that don't conflict with any selected
        improved = list(selected)
        
        for i in range(n_cands):
            if time.time() > deadline - 2:
                break
            if i in selected_set:
                continue
            # Check if i conflicts with any selected
            can_add = True
            for v in neighbors[i]:
                if v in selected_set:
                    can_add = False
                    break
            if can_add:
                improved.append(i)
                selected_set.add(i)
                
        return improved, conflict_set

    # Try multiple restarts with both strategies
    
    for r in range(200):
        if time.time() > deadline - 10:
            break
        
        # Alternate between strategies
        if r % 2 == 0:
            result = run_greedy_fast(r)
        else:
            result = run_greedy_weighted(r)
            
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    # Fallback
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
