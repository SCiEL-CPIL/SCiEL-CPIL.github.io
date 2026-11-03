import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports) # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is 3 (or 4, but i!=j so not 4)
    adj = [set() for _ in range(N_S)]
    
    # Precompute popcount for 11-bit integers? 
    # Or just use bit_count() in Python 3.10+? 
    # Assuming Python 3.10+ is available or use bin(x).count('1')
    
    def popcount(x):
        return bin(x).count('1')
        
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = popcount(masks[i] & masks[j])
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Find Max Independent Set
    best_size = 0
    best_set = set()
    
    rng = np.random.RandomState(42)
    
    def greedy_init():
        # Min-degree heuristic
        avail = list(range(N_S))
        s = set()
        degrees = [len(adj[i]) for i in avail]
        # We need to update degrees dynamically? 
        # For simplicity, use static degree or recompute.
        # With 330 nodes, recomputing is fine.
        
        # Better: Randomized Greedy
        # Sort by degree, pick lowest, remove neighbors, repeat.
        
        current_avail = list(range(N_S))
        s = set()
        
        while current_avail:
            # Find node in current_avail with min degree in current_avail
            min_deg = float('inf')
            min_node = -1
            for v in current_avail:
                deg = sum(1 for nb in adj[v] if nb in current_avail)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            # Remove min_node and its neighbors from current_avail
            to_remove = {min_node} | adj[min_node]
            current_avail = [v for v in current_avail if v not in to_remove]
            
        return s

    def local_search(init_set):
        current_set = set(init_set)
        current_size = len(current_set)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # 1. Try to add
            in_set = np.zeros(N_S, dtype=bool)
            for v in current_set:
                in_set[v] = True
                
            blocked = np.zeros(N_S, dtype=bool)
            for v in current_set:
                for nb in adj[v]:
                    blocked[nb] = True
            
            candidates = np.where(~in_set & ~blocked)[0]
            
            if len(candidates) > 0:
                # Pick random candidate
                v_add = candidates[rng.randint(len(candidates))]
                current_set.add(v_add)
                current_size += 1
                improved = True
                continue
            
            # 2. Try 1-swap
            # Remove v, add u
            # To be efficient, we can just try a few random swaps
            if len(current_set) > 0:
                v_remove = list(current_set)[rng.randint(len(current_set))]
                # Temporarily remove
                current_set.remove(v_remove)
                
                # Recompute blocked
                in_set = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    in_set[v] = True
                blocked = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    for nb in adj[v]:
                        blocked[nb] = True
                
                candidates = np.where(~in_set & ~blocked)[0]
                if len(candidates) > 0:
                    v_add = candidates[rng.randint(len(candidates))]
                    current_set.add(v_add)
                    # Size remains same, but we might escape local minimum
                    # Actually, this doesn't increase size. 
                    # We want to increase size.
                    # So we should only accept swap if it allows adding more later?
                    # Or just do random restarts.
                    pass
                else:
                    # Put it back
                    current_set.add(v_remove)

        return current_set, current_size

    # Run many restarts
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        init_set = greedy_init()
        final_set, size = local_search(init_set)
        if size > best_size:
            best_size = size
            best_set = final_set

    # 4. Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
