import time
import random
import math

def _greedy_independent_set_size(adj, n):
    """
    Fast greedy heuristic for independence number.
    Returns the size of the independent set found.
    """
    available = (1 << n) - 1
    count = 0
    while available:
        # Pick vertex with minimum degree in the induced subgraph 'available'
        # to maximize the number of vertices we can keep.
        best_v = -1
        min_deg = n
        temp_avail = available
        while temp_avail:
            v_bit = temp_avail & -temp_avail
            v = v_bit.bit_length() - 1
            # Count neighbors in available
            deg = bin(adj[v] & available).count('1')
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp_avail &= temp_avail - 1
        
        if best_v == -1:
            break
            
        count += 1
        # Remove best_v and its neighbors from available
        available &= ~(adj[best_v] | (1 << best_v))
        
    return count

def _has_independent_set_10(adj, n):
    """
    Exact check for independent set of size 10.
    """
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        # Pruning: if remaining vertices + current size < 10, fail
        if bin(available_mask).count('1') < 10 - size:
            return False
            
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_avail = available_mask & ~(adj[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Branch 2: Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _random_maximal_triangle_free(n, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    adj = [0] * n
    edges_to_add = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    # We want to maximize n. Try from high to low.
    # R(3,10) <= 41, so max n is 40.
    # Known best is 39. Seed is 26. Inspiration found ~29.
    # Let's try to push for 30-35.
    
    for n in range(40, 25, -1):
        if time.time() > deadline - 2:
            break
            
        found = False
        restarts = 0
        max_restarts = 50 # Adjust based on time
        
        while restarts < max_restarts and not found:
            restarts += 1
            if time.time() > deadline - 2:
                break
                
            rng = random.Random(restarts * 12345 + n)
            adj = _random_maximal_triangle_free(n, seed=restarts * 7 + n)
            
            # Simulated Annealing / Local Search
            # Cost function: Greedy Independent Set Size
            current_alpha = _greedy_independent_set_size(adj, n)
            
            # If we already have alpha < 10, check exactly
            if current_alpha < 10:
                if not _has_independent_set_10(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Annealing parameters
            T = 1.0
            cooling = 0.995
            steps_per_restart = 2000
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 2:
                    break
                    
                # Generate a move
                # 70% chance: Add a random valid edge
                # 30% chance: Remove a random edge
                
                move_type = rng.random()
                new_adj = adj[:] # Copy for evaluation
                
                if move_type < 0.7:
                    # Try to add an edge
                    # Pick random u, v
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    
                    if not (adj[u] & (1 << v)):
                        # Check if adding (u,v) creates a triangle
                        common = adj[u] & adj[v]
                        if common == 0:
                            # Valid addition
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                        else:
                            # Invalid, skip move
                            continue
                    else:
                        # Edge exists, skip
                        continue
                else:
                    # Try to remove an edge
                    # Find a random edge
                    u = rng.randint(0, n - 1)
                    # Get neighbors of u
                    neigh_mask = adj[u]
                    if neigh_mask == 0:
                        continue
                    # Pick random bit from neigh_mask
                    v_bit = neigh_mask & -neigh_mask
                    # Wait, this picks the lowest set bit. We want random.
                    # To pick random bit:
                    # count bits, pick random index, find k-th set bit.
                    # For simplicity and speed, let's just pick a random neighbor from the list?
                    # Or use a trick:
                    # temp = neigh_mask
                    # k = rng.randint(0, bin(neigh_mask).count('1')-1)
                    # while k > 0:
                    #   temp &= temp - 1
                    #   k -= 1
                    # v_bit = temp
                    # This is O(k).
                    
                    # Simpler: Pick random v, check if edge exists.
                    v = rng.randint(0, n - 1)
                    if u != v and (adj[u] & (1 << v)):
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        continue

                # Evaluate new state
                new_alpha = _greedy_independent_set_size(new_adj, n)
                
                # Acceptance criteria
                delta = new_alpha - current_alpha
                if delta <= 0:
                    # Better or equal
                    adj = new_adj
                    current_alpha = new_alpha
                    if current_alpha < 10:
                        if not _has_independent_set_10(adj, n):
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
                else:
                    # Worse, accept with probability exp(-delta/T)
                    if rng.random() < math.exp(-delta / T):
                        adj = new_adj
                        current_alpha = new_alpha
                
                T *= cooling

    # Fallback: Return a known valid graph if we didn't find a better one
    # The seed n=26 graph is valid.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    return {'n': n_seed, 'edges': edges_seed}
