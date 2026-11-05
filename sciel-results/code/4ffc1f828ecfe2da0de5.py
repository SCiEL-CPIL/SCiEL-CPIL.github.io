import time
import random
import itertools
import numpy as np

def _paley_17():
    """Return the Paley graph on 17 vertices as a bitmask adjacency list."""
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_k4(adj):
    """Count K4s using bitmask operations for speed."""
    n = len(adj)
    count = 0
    for i in range(n - 3):
        ni = adj[i] & ~((1 << (i + 1)) - 1)
        mask_j = ni
        while mask_j:
            j_bit = mask_j & -mask_j
            j = j_bit.bit_length() - 1
            mask_j ^= j_bit
            
            nij = adj[j] & adj[i] & ~((1 << (j + 1)) - 1)
            mask_k = nij
            while mask_k:
                k_bit = mask_k & -mask_k
                k = k_bit.bit_length() - 1
                mask_k ^= k_bit
                
                common = adj[i] & adj[j] & adj[k] & ~((1 << (k + 1)) - 1)
                # Count bits in common
                count += bin(common).count('1')
    return count

def _count_ind6(adj):
    """Count independent 6-sets by checking 5-sets and their common non-neighbors."""
    n = len(adj)
    count = 0
    all_mask = (1 << n) - 1
    
    for combo in itertools.combinations(range(n), 5):
        # Check if combo is independent
        is_ind = True
        for idx, v in enumerate(combo):
            others_mask = 0
            for u in combo:
                if u != v:
                    others_mask |= (1 << u)
            if adj[v] & others_mask:
                is_ind = False
                break
        
        if is_ind:
            # Find common non-neighbor
            union_neighbors = 0
            mask5 = 0
            for v in combo:
                union_neighbors |= adj[v]
                mask5 |= (1 << v)
            
            candidates = all_mask & ~mask5 & ~union_neighbors
            count += bin(candidates).count('1')
    
    return count

def _is_valid(adj):
    return _count_k4(adj) == 0 and _count_ind6(adj) == 0

def _sa_find(n, time_budget=5.0, start_adj=None):
    """Run Simulated Annealing to find a valid graph on n vertices."""
    start_time = time.time()
    
    if start_adj is not None:
        current_adj = start_adj[:]
    else:
        # Initial random graph with moderate density
        current_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
    
    current_k4 = _count_k4(current_adj)
    current_ind6 = _count_ind6(current_adj)
    current_cost = current_k4 + current_ind6
    
    if current_cost == 0:
        return current_adj
    
    best_adj = current_adj[:]
    best_cost = current_cost
    
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_budget:
        # Pick a random edge to flip
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Flip edge
        if (current_adj[i] >> j) & 1:
            # Remove edge
            current_adj[i] &= ~(1 << j)
            current_adj[j] &= ~(1 << i)
        else:
            # Add edge
            current_adj[i] |= (1 << j)
            current_adj[j] |= (1 << i)
        
        new_k4 = _count_k4(current_adj)
        new_ind6 = _count_ind6(current_adj)
        new_cost = new_k4 + new_ind6
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = current_adj[:]
                if best_cost == 0:
                    return best_adj
        else:
            # Revert
            if (current_adj[i] >> j) & 1:
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
            else:
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
        
        T *= cooling
        if T < 0.001:
            T = 0.001

    return best_adj if best_cost == 0 else None

def _grow_graph(base_adj, target_n, time_budget=5.0):
    """Grow a valid graph to target_n by adding vertices."""
    n = len(base_adj)
    if n >= target_n:
        return base_adj[:]
    
    # Start with base graph
    adj = base_adj[:]
    
    # Add new vertices one by one
    for new_v in range(n, target_n):
        # Add vertex with random connections
        adj.append(0)
        for old_v in range(n):
            if random.random() < 0.3:
                adj[old_v] |= (1 << new_v)
                adj[new_v] |= (1 << old_v)
        n += 1
    
    # Now use SA to fix violations
    return _sa_find(target_n, time_budget, adj)

def construct():
    deadline = time.time() + 110
    
    # Start with known valid graphs
    # Paley 17 is valid
    best_n = 17
    best_adj = _paley_17()
    
    # Try to grow to larger sizes
    # Known: R(4,6) > 35 (exoo graph)
    # We'll try to find graphs up to 35
    
    for target_n in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        
        found = None
        
        # Try multiple strategies
        # Strategy 1: Grow from best_adj
        time_per_strategy = min(3.0, remaining / 3)
        
        # Try growing from current best
        adj = _grow_graph(best_adj, target_n, time_budget=time_per_strategy)
        if adj and _is_valid(adj):
            found = adj
        else:
            # Strategy 2: Fresh SA with random start
            adj = _sa_find(target_n, time_budget=time_per_strategy)
            if adj and _is_valid(adj):
                found = adj
            else:
                # Strategy 3: Try with different density
                for density in [0.2, 0.4]:
                    if time.time() > deadline - 10:
                        break
                    adj = [0] * target_n
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if random.random() < density:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                    adj = _sa_find(target_n, time_budget=time_per_strategy, start_adj=adj)
                    if adj and _is_valid(adj):
                        found = adj
                        break
        
        if found:
            best_n = target_n
            best_adj = found
        else:
            # If we fail at n, continue trying n+1 (might be easier)
            pass
    
    # Convert bitmask to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
