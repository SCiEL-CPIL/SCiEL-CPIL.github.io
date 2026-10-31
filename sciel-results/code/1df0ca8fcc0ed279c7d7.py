import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all weight-3 vectors
    # v is a tuple of (indices, signs)
    # Represent as a tuple of length D for dot product efficiency?
    # Or just list of (idx, sign) pairs.
    
    vectors = []
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(tuple(v))
            
    # Precompute incompatibility?
    # N = 1320.
    # adj = [set() for _ in range(N)]
    # This might be slow to build if done naively.
    # But 1320 * 1320 / 2 is ~ 800k checks.
    # Each check is dot product.
    # Let's use the structural property:
    # Incompatible if share 2 indices and same signs.
    
    # Map from (i, j, si, sj) to list of vector indices
    # This allows fast neighbor lookup.
    from collections import defaultdict
    pair_map = defaultdict(list)
    
    for idx, v in enumerate(vectors):
        # v is a tuple of 11 ints
        # Get non-zero indices
        nz = [(i, v[i]) for i in range(D) if v[i] != 0]
        # nz has 3 elements
        # For each pair in nz
        for k in range(3):
            for m in range(k+1, 3):
                i1, s1 = nz[k]
                i2, s2 = nz[m]
                # Ensure i1 < i2 for consistent key
                if i1 > i2:
                    i1, i2 = i2, i1
                    s1, s2 = s2, s1
                key = (i1, i2, s1, s2)
                pair_map[key].append(idx)
                
    # Build adjacency list
    N = len(vectors)
    adj = [set() for _ in range(N)]
    for key, indices in pair_map.items():
        # All indices in this list are mutually incompatible
        for a in indices:
            for b in indices:
                if a != b:
                    adj[a].add(b)
                    
    # Greedy + Local Search
    best_set = set()
    
    def is_valid_set(s):
        # Since we build it via adj, it's always valid if we respect adj
        pass

    def greedy_select():
        selected = set()
        available = set(range(N))
        # Randomize order for diversity
        order = list(available)
        random.shuffle(order)
        for v_idx in order:
            if time.time() > deadline:
                break
            # Check if compatible with all selected
            # Conflict if any selected is in adj[v_idx]
            # Intersection of selected and adj[v_idx]
            if not selected.intersection(adj[v_idx]):
                selected.add(v_idx)
        return selected

    def local_search(initial_set):
        current = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add a vertex
            # Find all vertices not in current and not adjacent to any in current
            # This is slow if done naively.
            # Instead, iterate through candidates
            candidates = [v for v in range(N) if v not in current]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline:
                    break
                if not current.intersection(adj[v]):
                    current.add(v)
                    added = True
                    improved = True
                    break # Restart adding after one addition to update conflicts
            
            if not added:
                # Try to swap: remove one, add two?
                # Or just remove one and see if we can add more.
                # Simple hill climb: remove random, try to add.
                if current:
                    v_remove = random.choice(list(current))
                    current.remove(v_remove)
                    improved = True # Force loop to continue trying to add
                    # After removal, we will try to add in next iteration
                    # But we need to be careful not to loop infinitely
                    # Limit removals?
                    # Let's just break and restart if stuck
                    break
                    
        return current

    best_size = 0
    best_vectors = []
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        g_set = greedy_select()
        l_set = local_search(g_set)
        
        if len(l_set) > best_size:
            best_size = len(l_set)
            best_vectors = [vectors[i] for i in l_set]
            
        # If we found a very large set, maybe stop?
        # Max possible is bounded.
        # Upper bound for kissing number in 11D is not known, but 1320 is the pool size.
        # Actually, we are limited by the pool of weight-3 vectors.
        # Can we add weight-2 or 1?
        # We established mixing is bad for weight 2.
        # Weight 1 is bad.
        # So we stick to weight 3.
        
        # Optional: Try to add weight 2 vectors to the best set?
        # If we have a set of weight-3 vectors, can we add weight-2?
        # Max norm would be max(sqrt(3), sqrt(2)) = sqrt(3).
        # Need dist(weight-2, weight-3) >= sqrt(3).
        # And dist(weight-2, weight-2) >= sqrt(3).
        # This is complex. Let's stick to weight 3 for now.
        
    # Convert best_vectors to array
    if not best_vectors:
        # Fallback to seed-like output if something went wrong
        # But we should have found something
        return np.array([], dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)
