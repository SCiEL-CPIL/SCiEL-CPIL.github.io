import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is >= 3
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter >= 3:
                adj[i].add(j)
                adj[j].add(i)
                
    adj_list = [list(s) for s in adj]
    
    # Simulated Annealing to find large Independent Set
    # State: set of selected nodes
    # Move: Add a node, Remove a node, or Swap a node
    
    rng = np.random.RandomState(42)
    
    def get_initial_set():
        # Greedy random
        s = set()
        remaining = list(range(N_S))
        rng.shuffle(remaining)
        for v in remaining:
            if all(nb not in s for nb in adj_list[v]):
                s.add(v)
        return s
    
    def energy(s):
        return -len(s) # We want to maximize size, so minimize -size
    
    best_set = get_initial_set()
    best_size = len(best_set)
    
    current_set = set(best_set)
    current_energy = -len(current_set)
    
    T = 1.0
    cooling = 0.999
    
    # Precompute in_set for fast checking
    in_set = np.zeros(N_S, dtype=bool)
    for v in current_set:
        in_set[v] = True
    
    # We need to check if a node can be added: no neighbors in set
    # blocked[v] = True if any neighbor of v is in current_set
    
    def update_blocked(set_to_add=None, set_to_remove=None):
        nonlocal in_set
        if set_to_add:
            for v in set_to_add:
                in_set[v] = True
        if set_to_remove:
            for v in set_to_remove:
                in_set[v] = False
        # Recompute blocked is expensive, so we check on the fly or maintain a count
        # For SA, we just check neighbors of the candidate node
        
    # To speed up, we maintain a count of how many neighbors each node has in the set
    neighbor_count = np.zeros(N_S, dtype=int)
    for v in current_set:
        for nb in adj_list[v]:
            neighbor_count[nb] += 1
            
    # A node v can be added if neighbor_count[v] == 0
    
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
            
        # Random move
        move_type = rng.randint(3)
        
        if move_type == 0: # Add
            # Pick a random node not in set
            v = rng.randint(N_S)
            if v in current_set:
                continue
            if neighbor_count[v] > 0:
                continue # Can't add
            
            # Add v
            current_set.add(v)
            for nb in adj_list[v]:
                neighbor_count[nb] += 1
                
            delta_e = -1
            if delta_e < 0 or rng.rand() < np.exp(-delta_e / T):
                pass # Accept
            else:
                # Reject
                current_set.remove(v)
                for nb in adj_list[v]:
                    neighbor_count[nb] -= 1
                    
        elif move_type == 1: # Remove
            if not current_set:
                continue
            v = list(current_set)[rng.randint(len(current_set))]
            current_set.remove(v)
            for nb in adj_list[v]:
                neighbor_count[nb] -= 1
                
            delta_e = 1
            if delta_e < 0 or rng.rand() < np.exp(-delta_e / T):
                pass # Accept
            else:
                # Reject
                current_set.add(v)
                for nb in adj_list[v]:
                    neighbor_count[nb] += 1
                    
        else: # Swap
            if not current_set:
                continue
            # Remove a random v from set
            v_remove = list(current_set)[rng.randint(len(current_set))]
            current_set.remove(v_remove)
            for nb in adj_list[v_remove]:
                neighbor_count[nb] -= 1
                
            # Add a random u not in set that is now valid
            # Valid if neighbor_count[u] == 0
            # We can pick a random u and check
            u = rng.randint(N_S)
            if u in current_set:
                # Revert
                current_set.add(v_remove)
                for nb in adj_list[v_remove]:
                    neighbor_count[nb] += 1
                continue
                
            if neighbor_count[u] > 0:
                # Revert
                current_set.add(v_remove)
                for nb in adj_list[v_remove]:
                    neighbor_count[nb] += 1
                continue
                
            # Perform swap
            current_set.add(u)
            for nb in adj_list[u]:
                neighbor_count[nb] += 1
                
            delta_e = 0 # Size unchanged
            # Accept with probability 1 (or explore)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Assign signs
    # For each support in best_set, assign signs such that dot products are <= 0
    vectors = []
    supports_in_set = [supports[i] for i in best_set]
    
    # We will assign signs greedily.
    # Since intersection <= 2, we can satisfy constraints locally.
    # For simplicity, we try all 16 sign combinations for each support and pick one that works with previously assigned.
    
    assigned_vectors = []
    for s_idx, support in enumerate(supports_in_set):
        valid_sign = None
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            
            # Check against all previously assigned vectors
            ok = True
            for u in assigned_vectors:
                if np.dot(v, u) > 0:
                    ok = False
                    break
            if ok:
                valid_sign = v
                break
                
        if valid_sign is None:
            # Fallback: pick random
            signs = tuple(rng.choice([1, -1]) for _ in range(4))
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            valid_sign = v
            
        assigned_vectors.append(valid_sign)
        
    if not assigned_vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return np.array(assigned_vectors, dtype=np.int8)
