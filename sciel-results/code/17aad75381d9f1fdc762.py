import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Build adjacency as bitmasks for fast conflict checking
    # adj[i] is a bitmask where bit j is set if i and j conflict (intersection size 3)
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to check if a set of nodes is an independent set
    def is_independent(node_set):
        for i in node_set:
            # Check if any other node in the set is adjacent to i
            conflict_mask = adj[i]
            # Check intersection of conflict_mask with the set of other nodes
            for j in node_set:
                if i != j and (conflict_mask >> j) & 1:
                    return False
        return True
    
    # Greedy min-degree heuristic to get initial solution
    def greedy_min_degree():
        remaining = set(range(n))
        selected = set()
        # Precompute degrees
        deg = [bin(adj[i]).count('1') for i in range(n)]
        
        while remaining:
            # Pick node with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_node = -1
            for i in remaining:
                # Count neighbors in remaining
                d = 0
                mask = adj[i]
                for j in remaining:
                    if i != j and (mask >> j) & 1:
                        d += 1
                if d < min_deg:
                    min_deg = d
                    min_node = i
            
            if min_node == -1:
                break
            selected.add(min_node)
            remaining.discard(min_node)
            # Remove all neighbors of min_node from remaining
            for j in list(remaining):
                if (adj[min_node] >> j) & 1:
                    remaining.discard(j)
        
        return selected
    
    # Local search: 1-swap (remove one, add one)
    def local_search_1swap(selected, time_limit=5):
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            selected_list = list(selected)
            for i in range(len(selected_list)):
                if time.time() > deadline - 10:
                    return selected
                # Try removing selected_list[i]
                candidate = selected - {selected_list[i]}
                # Try adding any node not in candidate
                for j in range(n):
                    if j not in candidate:
                        new_candidate = candidate | {j}
                        if is_independent(new_candidate):
                            if len(new_candidate) > len(selected):
                                selected = new_candidate
                                improved = True
                                selected_list = list(selected)
                                break
        return selected
    
    # Simulated annealing for Maximum Independent Set
    def simulated_annealing(initial_solution, time_limit=30):
        start = time.time()
        current = set(initial_solution)
        best = set(current)
        best_size = len(current)
        
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        while time.time() - start < time_limit and T > T_min:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random node
                candidates = [j for j in range(n) if j not in current]
                if candidates:
                    j = random.choice(candidates)
                    new_state = current | {j}
                    if is_independent(new_state):
                        delta = 1
                        if delta > 0 or random.random() < np.exp(-delta / T):
                            current = new_state
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(current)
            elif move_type < 0.8:
                # Remove a random node
                if current:
                    j = random.choice(list(current))
                    new_state = current - {j}
                    delta = -1
                    if delta > 0 or random.random() < np.exp(-delta / T):
                        current = new_state
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
            else:
                # Swap: remove one, add one
                if current:
                    j = random.choice(list(current))
                    candidates = [k for k in range(n) if k not in current]
                    if candidates:
                        k = random.choice(candidates)
                        new_state = (current - {j}) | {k}
                        if is_independent(new_state):
                            delta = 0  # Size doesn't change
                            if random.random() < np.exp(-delta / T):
                                current = new_state
                                if len(current) > best_size:
                                    best = set(current)
                                    best_size = len(current)
            
            T *= cooling
        
        return best
    
    # Get initial solution via greedy
    best_is = greedy_min_degree()
    best_size = len(best_is)
    
    # Improve with local search
    if time.time() < deadline - 40:
        best_is = local_search_1swap(best_is, time_limit=10)
        best_size = len(best_is)
    
    # Run simulated annealing with multiple restarts
    if time.time() < deadline - 20:
        sa_budget = min(30, deadline - 20 - time.time())
        best_is = simulated_annealing(best_is, time_limit=sa_budget)
        best_size = len(best_is)
    
    # Additional restarts if time permits
    while time.time() < deadline - 15:
        # New greedy start
        current_best = greedy_min_degree()
        if time.time() < deadline - 10:
            current_best = local_search_1swap(current_best, time_limit=3)
        if time.time() < deadline - 5:
            current_best = simulated_annealing(current_best, time_limit=5)
        
        if len(current_best) > best_size:
            best_is = current_best
            best_size = len(current_best)
    
    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
