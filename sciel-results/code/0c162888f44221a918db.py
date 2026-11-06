"""Triangle-free 4-chromatic graphs on 16 vertices via local search with swaps."""
# EVOLVE-BLOCK-START
import random
import time

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        if adj[u] & (1 << u):
            return False
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            # If u and v are connected, they cannot share a common neighbor
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj):
    """
    Check if the graph defined by adj bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    """
    n = len(adj)
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    colors = [-1] * n
    
    def backtrack(v_idx):
        if v_idx == n:
            return True
        
        v = order[v_idx]
        # Check colors of neighbors
        mask = 0
        neighbors = adj[v]
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            neighbors ^= lb
            if colors[u] != -1:
                mask |= (1 << colors[u])
        
        available = ~mask & 0b111
        if available == 0:
            return False
        
        for c in range(3):
            if available & (1 << c):
                colors[v] = c
                if backtrack(v_idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return backtrack(0)

def _local_search(seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    # Initialize with Grotzsch
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)
    
    best_count = len(edge_set)
    best_adj = adj[:]
    
    failed_attempts = 0
    
    while time.time() < deadline:
        # Try to add a random edge that doesn't create a triangle
        added = False
        # Iterate through all non-edges? Or random?
        # Random is faster to sample, but might miss good ones.
        # Let's try a few random non-edges.
        
        # To ensure we explore well, we can keep a list of non-edges, but it's expensive to update.
        # Instead, we just pick random pairs.
        
        for _ in range(50): # Try 50 random additions per loop
            if time.time() > deadline:
                break
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edge_set:
                continue
            
            # Check triangle-free
            if adj[u] & adj[v] == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add(e)
                added = True
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_adj = adj[:]
                failed_attempts = 0
                break # Restart the 50 loop to pick another random edge
        
        if not added:
            failed_attempts += 1
            
            # If stuck, try a swap
            if failed_attempts > 20:
                # Save state
                saved_adj = adj[:]
                saved_edge_set = set(edge_set)
                
                # Remove a random edge
                if not edge_set:
                    break
                e_out = rng.choice(list(edge_set))
                u_out, v_out = e_out
                adj[u_out] &= ~(1 << v_out)
                adj[v_out] &= ~(1 << u_out)
                edge_set.remove(e_out)
                
                # Try to add a different edge
                swap_success = False
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    u_in = rng.randrange(n)
                    v_in = rng.randrange(n)
                    if u_in == v_in:
                        continue
                    e_in = (min(u_in, v_in), max(u_in, v_in))
                    if e_in in edge_set:
                        continue
                    
                    # Check triangle-free
                    if adj[u_in] & adj[v_in] == 0:
                        # Tentatively add
                        adj[u_in] |= (1 << v_in)
                        adj[v_in] |= (1 << u_in)
                        edge_set.add(e_in)
                        
                        # Check 3-colorability
                        if not _is_3_colorable(adj):
                            swap_success = True
                            if len(edge_set) > best_count:
                                best_count = len(edge_set)
                                best_adj = adj[:]
                            break
                        else:
                            # Revert
                            adj[u_in] &= ~(1 << v_in)
                            adj[v_in] &= ~(1 << u_in)
                            edge_set.remove(e_in)
                
                if not swap_success:
                    # Revert removal
                    adj[u_out] |= (1 << v_out)
                    adj[v_out] |= (1 << u_out)
                    edge_set.add(e_out)
                
                failed_attempts = 0
                
            # If stuck for too long, kick
            if failed_attempts > 100:
                # Remove 5 random edges
                kick_edges = rng.sample(list(edge_set), k=min(5, len(edge_set)))
                for e in kick_edges:
                    u, v = e
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_set.remove(e)
                
                # Greedily add edges back
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    u = rng.randrange(n)
                    v = rng.randrange(n)
                    if u == v:
                        continue
                    e = (min(u, v), max(u, v))
                    if e in edge_set:
                        continue
                    if adj[u] & adj[v] == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add(e)
                        if len(edge_set) > best_count:
                            best_count = len(edge_set)
                            best_adj = adj[:]
                
                failed_attempts = 0

    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    
    return result_edges, len(result_edges)

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges. Uses local search with swaps.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Run multiple instances of local search with different seeds
    seed = 0
    while time.time() < deadline - 10.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        if best_count >= 50: # Heuristic stop
            break

    # Final validation
    if best_edges:
        if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)):
            grotzsch = _build_grotzsch_edges()
            best_edges = list(grotzsch)
    else:
        best_edges = _build_grotzsch_edges()
    
    return best_edges
# EVOLVE-BLOCK-END
