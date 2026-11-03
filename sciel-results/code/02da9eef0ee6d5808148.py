import numpy as np
import itertools
import time
import random
import math

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
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
        
    # Precompute adjacency list
    # i and j conflict if they share exactly 3 or 4 elements
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = masks[i] & masks[j]
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)
                
    degrees = [len(adj[i]) for i in range(n)]
    
    # Greedy MIS initialization
    def greedy_mis(rng):
        available = set(range(n))
        is_set = set()
        # Sort by degree? Or random?
        # Min-degree greedy is good
        while available:
            # Pick node with min degree in available
            min_deg = float('inf')
            candidates = []
            for v in available:
                deg = len(adj[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            if not candidates:
                break
            best_node = rng.choice(candidates)
            is_set.add(best_node)
            to_remove = adj[best_node] | {best_node}
            available -= to_remove
        return is_set

    best_is = set()
    best_size = 0
    
    # Simulated Annealing
    def simulated_annealing(initial_set, rng, steps=50000):
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size_sa = current_size
        
        T = 1.0
        cooling = 0.9999
        
        for i in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Propose a move
            move_type = rng.random()
            
            if move_type < 0.4:
                # Add a node
                candidates = [v for v in range(n) if v not in current_set and not (adj[v] & current_set)]
                if not candidates:
                    continue
                v = rng.choice(candidates)
                new_set = current_set | {v}
                delta = 1
            elif move_type < 0.8:
                # Remove a node
                if not current_set:
                    continue
                v = rng.choice(list(current_set))
                new_set = current_set - {v}
                delta = -1
            else:
                # Swap
                if not current_set:
                    continue
                v_out = rng.choice(list(current_set))
                # Find v_in that is compatible with current_set - {v_out}
                trial_set = current_set - {v_out}
                candidates = [v for v in range(n) if v not in trial_set and not (adj[v] & trial_set)]
                if not candidates:
                    continue
                v_in = rng.choice(candidates)
                new_set = trial_set | {v_in}
                delta = 0 # Size doesn't change, but structure might improve for future moves?
                # Actually, if delta is 0, we should only accept if it helps?
                # For MIS, size is the only objective.
                # So swaps with delta 0 are neutral.
                # We might want to bias towards "more flexible" nodes?
                # Let's just accept with prob 0.5 to escape local minima in structure.
                # Or reject. Let's reject for now to focus on size.
                if delta == 0:
                    # Only accept if we are stuck?
                    # For now, reject neutral moves to save time
                    continue

            # Acceptance
            if delta > 0:
                current_set = new_set
                current_size += delta
                if current_size > best_size_sa:
                    best_size_sa = current_size
                    best_set = set(current_set)
            else:
                # delta < 0
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_set = new_set
                    current_size += delta
            
            T *= cooling
        
        return best_set

    # Run multiple restarts
    seed_val = 0
    while time.time() < deadline:
        rng = random.Random(seed_val)
        
        # Initialize
        is_set = greedy_mis(rng)
        
        # Improve with SA
        is_set = simulated_annealing(is_set, rng, steps=20000)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            
        seed_val += 1

    # Construct vectors
    vectors = []
    for s_idx in best_is:
        support = subsets[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
