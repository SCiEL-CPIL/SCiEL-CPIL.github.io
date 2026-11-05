import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin

    # We know n=47 is possible (R(C4, K1,39) = 47).
    # Try 47 first.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break

        required_min_deg = n - 39

        # Try to construct for this n
        result = _robust_construct(n, required_min_deg, deadline)

        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _check_c4_bitset(adj_bits, n, u, v):
    """
    Check if adding edge (u,v) creates a C4 using bitsets.
    adj_bits[i] is an integer where bit j is set if (i,j) is an edge.
    Returns True if a C4 is created, False otherwise.
    """
    # For C4: u-x-y-v-u where x in N(u)\{v}, y in N(v)\{u}, and (x,y) in E
    # Equivalent: exists x in N(u)\{v} such that N(x) intersects N(v)\{u}
    nu = adj_bits[u] & ~(1 << v)
    nv = adj_bits[v] & ~(1 << u)
    
    # For each x in N(u)\{v}, check if N(x) intersects N(v)\{u}
    # We can do this by: for each x in nu, check if adj_bits[x] & nv != 0
    x_mask = nu
    while x_mask:
        # Get lowest set bit
        x_bit = x_mask & (-x_mask)
        x = x_bit.bit_length() - 1
        if adj_bits[x] & nv:
            return True
        x_mask ^= x_bit
    
    return False

def _greedy_fill_bitset(adj_bits, n, target, deadline):
    """
    Greedily add edges to increase min degree using bitset-based adjacency.
    Returns True if target min degree is reached.
    """
    while time.time() < deadline:
        degrees = [adj_bits[i].bit_count() for i in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        # Pick one randomly to break symmetry
        u = random.choice(min_vertices)
        
        # Get non-neighbors of u
        all_mask = (1 << n) - 1
        non_neighbors_mask = all_mask & ~adj_bits[u] & ~(1 << u)
        
        if non_neighbors_mask == 0:
            return False
            
        # Get list of non-neighbor vertices
        non_neighbors = []
        mask = non_neighbors_mask
        while mask:
            v_bit = mask & (-mask)
            v = v_bit.bit_length() - 1
            non_neighbors.append(v)
            mask ^= v_bit
        
        random.shuffle(non_neighbors)
        
        added = False
        for v in non_neighbors:
            if not _check_c4_bitset(adj_bits, n, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added = True
                break  # Restart loop to re-evaluate degrees
        
        if not added:
            # Stuck in greedy phase
            return False
            
    return False

def _local_search_swaps_bitset(adj_bits, n, target, deadline):
    """
    Local search using edge swaps to increase min degree, using bitset-based adjacency.
    Assumes adj_bits is C4-free.
    Returns True if target min degree is reached.
    """
    max_iters = 5000  # Increased from 2000
    for _ in range(max_iters):
        if time.time() > deadline:
            return False
            
        degrees = [adj_bits[i].bit_count() for i in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        u = random.choice(min_vertices)
        
        # Get non-neighbors of u
        all_mask = (1 << n) - 1
        non_neighbors_mask = all_mask & ~adj_bits[u] & ~(1 << u)
        
        if non_neighbors_mask == 0:
            return False
            
        # Get list of non-neighbor vertices
        non_neighbors = []
        mask = non_neighbors_mask
        while mask:
            v_bit = mask & (-mask)
            v = v_bit.bit_length() - 1
            non_neighbors.append(v)
            mask ^= v_bit
        
        random.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if _check_c4_bitset(adj_bits, n, u, v):
                # Try to find a swap: remove (x,y) where x in N(u)\{v}, y in N(v)\{u}, (x,y) in E
                # and such that after removing (x,y), adding (u,v) is safe
                nu = adj_bits[u] & ~(1 << v)
                nv = adj_bits[v] & ~(1 << u)
                
                # Find all blocking edges (x,y) where x in nu, y in nv, (x,y) in E
                blockers = []
                x_mask = nu
                while x_mask:
                    x_bit = x_mask & (-x_mask)
                    x = x_bit.bit_length() - 1
                    common = adj_bits[x] & nv
                    if common:
                        # Get all y in common
                        y_mask = common
                        while y_mask:
                            y_bit = y_mask & (-y_mask)
                            y = y_bit.bit_length() - 1
                            blockers.append((x, y))
                            y_mask ^= y_bit
                    x_mask ^= x_bit
                
                for x, y in blockers:
                    # Tentatively remove (x,y) and add (u,v)
                    adj_bits[x] &= ~(1 << y)
                    adj_bits[y] &= ~(1 << x)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    
                    # Check if the graph is still C4-free
                    # The only new C4s must involve (u,v)
                    # So check if adding (u,v) to graph-without-(x,y) creates a C4
                    # We need to check: does there exist x' in N(u)\{v} and y' in N(v)\{u} with (x',y') in E?
                    # Note: (x,y) is no longer in E, so it won't block.
                    nu_new = adj_bits[u] & ~(1 << v)
                    nv_new = adj_bits[v] & ~(1 << u)
                    
                    safe = True
                    x_prime_mask = nu_new
                    while x_prime_mask:
                        xp_bit = x_prime_mask & (-x_prime_mask)
                        xp = xp_bit.bit_length() - 1
                        if adj_bits[xp] & nv_new:
                            safe = False
                            break
                        x_prime_mask ^= xp_bit
                    
                    if safe:
                        # Keep the swap
                        improved = True
                        break
                    else:
                        # Revert
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        adj_bits[x] |= (1 << y)
                        adj_bits[y] |= (1 << x)
                        continue
                        
                if improved:
                    break
            else:
                # No C4, just add
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                improved = True
                break
                
        if not improved:
            break
            
    return False

def _robust_construct(n, target, deadline):
    """
    Combines greedy and local search with multiple restarts using bitset-based adjacency.
    """
    # Try several random greedy starts
    for attempt in range(150):  # Increased from 100
        if time.time() > deadline:
            break
            
        adj_bits = [0] * n
        
        # 1. Greedy Phase
        if _greedy_fill_bitset(adj_bits, n, target, deadline):
            return (n, _to_edges_bitset(adj_bits, n))
            
        # 2. Local Search Phase
        if _local_search_swaps_bitset(adj_bits, n, target, deadline):
            return (n, _to_edges_bitset(adj_bits, n))
            
    return None

def _to_edges_bitset(adj_bits, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_bits[u] & (1 << v):
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
