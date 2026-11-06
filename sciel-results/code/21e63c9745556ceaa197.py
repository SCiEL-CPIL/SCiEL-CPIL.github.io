"""Combined program: greedy C4-free construction with bitmask speed, polarity seed, and edge-swap local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_c4_free_bitmask(adj, n):
    """Check if the bitmask adjacency list represents a C4-free graph."""
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if common.bit_count() >= 2:
                return False
    return True

def _polarity_graph(n):
    """
    Construct a polarity graph for n = q^2 where q is prime.
    Vertices are (a,b) in F_q^2, edge if a*b' - a'*b = 1 (mod q).
    """
    if n < 4:
        return None
    q = int(round(n ** 0.5))
    if q * q != n:
        return None
    
    # Check if q is prime
    if q < 2:
        return None
    is_prime = True
    for i in range(2, int(q**0.5) + 1):
        if q % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None

    adj = [0] * n

    def idx(a, b):
        return a * q + b

    for a in range(q):
        for b in range(q):
            u = idx(a, b)
            for c in range(q):
                for d in range(q):
                    if c == a and d == b:
                        continue
                    v = idx(c, d)
                    if v > u:  # Only add each edge once
                        val = (a * d - b * c) % q
                        if val == 1:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
    return adj

def _construct_c4_free(n, min_deg_required, time_limit, seed=None):
    """
    Construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses polarity graph seed + bitmask-based local search.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    # Try polarity graph first for n that are prime squares
    pp_adj = _polarity_graph(n)
    if pp_adj is not None:
        degrees = [adj.bit_count() for adj in pp_adj]
        if min(degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if pp_adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    mask = (1 << n) - 1
    
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize adjacency and degrees
        adj = [0] * n
        degrees = [0] * n
        
        # Phase 1: Greedy construction with randomization
        # Process vertices in random order
        order = list(range(n))
        rng.shuffle(order)
        
        for u in order:
            # Find all non-neighbors of u
            non_neighbors = (~adj[u]) & mask
            non_neighbors &= ~(1 << u)
            
            nn_list = []
            tmp = non_neighbors
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                nn_list.append(idx)
                tmp ^= lsb
            
            rng.shuffle(nn_list)
            
            # Add edges to u that don't create C4
            for v in nn_list:
                # Check if adding (u,v) creates a C4
                # C4 exists if u and v share >= 2 common neighbors
                common = adj[u] & adj[v]
                if common.bit_count() < 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Phase 2: Local search with edge swaps to improve minimum degree
        # We try to add edges to low-degree vertices, and if stuck, remove edges
        # from high-degree vertices to create space.
        
        max_local_steps = 20000
        stuck_count = 0
        
        for step in range(max_local_steps):
            if time.time() - start_time > time_limit:
                break
            
            min_deg = min(degrees)
            if min_deg >= min_deg_required:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # Find vertex with minimum degree
            min_deg_vertex = degrees.index(min_deg)
            
            # Get non-neighbors of min_deg_vertex
            non_neighbors = (~adj[min_deg_vertex]) & mask
            non_neighbors &= ~(1 << min_deg_vertex)
            
            if non_neighbors == 0:
                break
            
            nn_list = []
            tmp = non_neighbors
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                nn_list.append(idx)
                tmp ^= lsb
            
            rng.shuffle(nn_list)
            
            found = False
            for v in nn_list:
                # Check if adding (min_deg_vertex, v) creates a C4
                common = adj[min_deg_vertex] & adj[v]
                if common.bit_count() < 2:
                    adj[min_deg_vertex] |= (1 << v)
                    adj[v] |= (1 << min_deg_vertex)
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    found = True
                    stuck_count = 0
                    break
            
            if not found:
                # Stuck - try edge removal to open up possibilities
                stuck_count += 1
                
                if stuck_count > 500:
                    # Reset with a fresh attempt
                    break
                
                # Find edges that can be removed without dropping min degree below threshold
                # We want to remove an edge (a,b) such that degrees[a] > min_deg_required and degrees[b] > min_deg_required
                # And ideally, removing it helps the min_deg_vertex.
                
                # Strategy: Pick a random high-degree vertex 'h' and remove an edge (h, x)
                # where x is a neighbor of min_deg_vertex's neighbor? No, simpler:
                # Just remove a random edge between two vertices that are not min_deg_vertex
                # and have degree > min_deg_required.
                
                candidates = []
                for u in range(n):
                    if u == min_deg_vertex:
                        continue
                    if degrees[u] <= min_deg_required:
                        continue
                    # Get neighbors of u
                    neighbors = adj[u]
                    tmp_nb = neighbors
                    while tmp_nb:
                        lsb = tmp_nb & -tmp_nb
                        idx = lsb.bit_length() - 1
                        tmp_nb ^= lsb
                        if idx == u:
                            continue
                        if degrees[idx] > min_deg_required:
                            candidates.append((u, idx))
                
                if candidates:
                    u, v = rng.choice(candidates)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    degrees[u] -= 1
                    degrees[v] -= 1
                else:
                    # If no candidates, we might be stuck in a local minimum where
                    # everyone is at or below min_deg_required.
                    # This shouldn't happen if min_deg < min_deg_required, but just in case.
                    break
        
        # Check again after local search
        min_deg = min(degrees)
        if min_deg >= min_deg_required:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin
    
    # Try from largest n down with better time allocation
    # 47: min_deg 8. 46: min_deg 7. 45: min_deg 6.
    # 46 is the target.
    
    time_budgets = {
        47: 20.0,
        46: 50.0,
        45: 25.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
    
    # Fallback: return a valid small graph
    # For n=39, min_deg 0 is always possible (empty graph)
    # But let's return something slightly better if possible, or just 39 empty.
    # Actually, for n=39, min_deg 0.
    # Let's just return n=39 with no edges to be safe and fast.
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
