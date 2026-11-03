import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Precompute adjacency list and adjacency bitmask
    # i and j conflict if they share exactly 3 elements
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n  # bitmask of neighbors for each node
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    def greedy_min_degree(seed_val):
        """Greedy MIS using minimum degree heuristic"""
        rng = random.Random(seed_val)
        remaining = set(range(n))
        is_set = set()
        while remaining:
            # Find node with minimum degree in remaining subgraph
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            # Random tie-break
            v = rng.choice(candidates)
            is_set.add(v)
            remaining -= adj[v]
            remaining.discard(v)
        return is_set

    def local_search_1swap(is_set, rng):
        """Local search with 1-in 1-out swaps"""
        is_set = set(is_set)
        in_set_mask = 0
        for v in is_set:
            in_set_mask |= (1 << v)
        
        # Try to add any node that doesn't conflict
        while True:
            improved = False
            candidates = [v for v in range(n) if v not in is_set]
            rng.shuffle(candidates)
            for v in candidates:
                if not (adj[v] & is_set):
                    is_set.add(v)
                    in_set_mask |= (1 << v)
                    improved = True
            if not improved:
                break
        
        # Try swaps: remove one, add one
        for _ in range(20):  # Limited swap attempts
            improved = False
            in_list = list(is_set)
            rng.shuffle(in_list)
            out_list = [v for v in range(n) if v not in is_set]
            rng.shuffle(out_list)
            
            for out_v in in_list:
                # Try removing out_v
                new_set = is_set - {out_v}
                new_mask = in_set_mask & ~(1 << out_v)
                
                for in_v in out_list:
                    # Check if in_v can be added
                    if not (adj[in_v] & new_set):
                        is_set = new_set | {in_v}
                        in_set_mask = new_mask | (1 << in_v)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return is_set

    def simulated_annealing(is_set, rng):
        """Simulated annealing to escape local optima"""
        is_set = set(is_set)
        current_size = len(is_set)
        T = 10.0  # Initial temperature
        
        # Precompute for efficiency
        in_set_mask = 0
        for v in is_set:
            in_set_mask |= (1 << v)
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Pick a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a node
                candidates = [v for v in range(n) if v not in is_set]
                if not candidates:
                    continue
                v = rng.choice(candidates)
                if not (adj[v] & is_set):
                    is_set.add(v)
                    in_set_mask |= (1 << v)
                else:
                    # Compute penalty
                    conflicts = len(adj[v] & is_set)
                    delta = -conflicts
                    if delta < 0:
                        # Accept with probability exp(delta/T)
                        if rng.random() < np.exp(delta / T):
                            # Remove conflicting nodes and add v
                            for c in (adj[v] & is_set):
                                is_set.discard(c)
                                in_set_mask &= ~(1 << c)
                            is_set.add(v)
                            in_set_mask |= (1 << v)
            elif move_type < 0.6:
                # Remove a node
                if is_set:
                    v = rng.choice(list(is_set))
                    is_set.discard(v)
                    in_set_mask &= ~(1 << v)
            else:
                # Swap: remove one, add one
                if is_set:
                    out_v = rng.choice(list(is_set))
                    candidates = [v for v in range(n) if v not in is_set]
                    if candidates:
                        in_v = rng.choice(candidates)
                        # Check if swap is valid
                        new_set = (is_set - {out_v}) | {in_v}
                        if not (adj[in_v] & (is_set - {out_v})):
                            is_set = new_set
                            in_set_mask = (in_set_mask & ~(1 << out_v)) | (1 << in_v)
                        else:
                            # Accept with probability
                            conflicts = len(adj[in_v] & (is_set - {out_v}))
                            delta = -conflicts
                            if delta < 0:
                                if rng.random() < np.exp(delta / T):
                                    for c in (adj[in_v] & (is_set - {out_v})):
                                        is_set.discard(c)
                                        in_set_mask &= ~(1 << c)
                                    is_set.discard(out_v)
                                    is_set.add(in_v)
                                    in_set_mask = (in_set_mask & ~(1 << out_v)) | (1 << in_v)
            
            # Cool down
            T *= 0.995
            
            # Track best
            if len(is_set) > current_size:
                current_size = len(is_set)
        
        return is_set

    best_is = set()
    best_size = 0
    
    # Run multiple restarts with different strategies
    current_seed = 0
    while time.time() < deadline:
        # Alternate between strategies
        if current_seed % 3 == 0:
            is_set = greedy_min_degree(current_seed)
        elif current_seed % 3 == 1:
            is_set = greedy_min_degree(current_seed)
            is_set = local_search_1swap(is_set, random.Random(current_seed))
        else:
            is_set = greedy_min_degree(current_seed)
            is_set = local_search_1swap(is_set, random.Random(current_seed))
            is_set = simulated_annealing(is_set, random.Random(current_seed))
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            
        current_seed += 1
    
    # Final polish: try to add any remaining nodes
    final_is = set(best_is)
    while True:
        improved = False
        for v in range(n):
            if v not in final_is and not (adj[v] & final_is):
                final_is.add(v)
                improved = True
        if not improved:
            break
    
    # Build vectors
    vectors = []
    for idx in final_is:
        s = subsets[idx]
        # For each selected subset, add all 16 sign variations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
