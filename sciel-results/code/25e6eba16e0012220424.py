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
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj):
    """
    Check if the graph defined by adj bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    Uses MRV heuristic with forward checking for fast pruning.
    """
    n = len(adj)
    colors = [-1] * n
    
    def backtrack():
        # Find uncolored vertex with fewest available colors (MRV)
        best_v = -1
        best_avail = 4
        for v in range(n):
            if colors[v] != -1:
                continue
            mask = 0
            neighbors = adj[v]
            while neighbors:
                lb = neighbors & (-neighbors)
                u = lb.bit_length() - 1
                neighbors ^= lb
                if colors[u] != -1:
                    mask |= (1 << colors[u])
            avail = (~mask) & 0b111
            cnt = bin(avail).count('1')
            if cnt < best_avail:
                best_avail = cnt
                best_v = v
                if cnt <= 1:
                    break
        if best_v == -1:
            return True  # All colored
        if best_avail == 0:
            return False  # No color available
        
        v = best_v
        mask = 0
        neighbors = adj[v]
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            neighbors ^= lb
            if colors[u] != -1:
                mask |= (1 << colors[u])
        avail = (~mask) & 0b111
        
        for c in range(3):
            if avail & (1 << c):
                colors[v] = c
                if backtrack():
                    return True
                colors[v] = -1
        return False
    
    return backtrack()

def _greedy_fill(adj, edge_set, deadline):
    """Try to add any valid non-edge repeatedly until none remain or time runs out."""
    n = 16
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    return
                if (u, v) not in edge_set and adj[u] & adj[v] == 0:
                    # Check 3-colorability before adding
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    if _is_3_colorable(adj):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove((u, v))
                    else:
                        improved = True
                        break
            if improved:
                break

def _local_search(seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    # Initialize with Grotzsch
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)
    
    # Greedy fill to get a good starting point
    _greedy_fill(adj, edge_set, deadline)
    
    best_count = len(edge_set)
    best_adj = adj[:]
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    failed_moves = 0
    kick_count = 0
    
    while time.time() < deadline:
        # Try add-one move
        added = False
        # Pick a random non-edge
        non_edge_candidates = [e for e in all_edges if e not in edge_set]
        if non_edge_candidates:
            u, v = rng.choice(non_edge_candidates)
            if adj[u] & adj[v] == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                if not _is_3_colorable(adj):
                    added = True
                    if len(edge_set) > best_count:
                        best_count = len(edge_set)
                        best_adj = adj[:]
                    failed_moves = 0
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_set.remove((u, v))
        
        if not added:
            failed_moves += 1
            
            # Kick perturbation every 80 failed moves
            if failed_moves >= 80:
                failed_moves = 0
                kick_count += 1
                
                saved_adj = adj[:]
                saved_edge_set = set(edge_set)
                
                # Remove 2-3 random edges
                k = min(2 + (kick_count % 2), len(edge_set))
                if k > 0:
                    kick_edges = rng.sample(list(edge_set), k=k)
                    for e_out in kick_edges:
                        u_out, v_out = e_out
                        adj[u_out] &= ~(1 << v_out)
                        adj[v_out] &= ~(1 << u_out)
                        edge_set.remove(e_out)
                
                # Greedy refill
                _greedy_fill(adj, edge_set, deadline)
                
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_adj = adj[:]
                else:
                    adj = saved_adj
                    edge_set = saved_edge_set
            
            # Try remove-one-add-two move
            elif len(edge_set) > 2:
                e_out = rng.choice(list(edge_set))
                u_out, v_out = e_out
                adj[u_out] &= ~(1 << v_out)
                adj[v_out] &= ~(1 << u_out)
                edge_set.remove(e_out)
                
                swap_success = False
                # Try up to 30 random pairs to add back
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    u = rng.randrange(n)
                    v = rng.randrange(n)
                    if u == v:
                        continue
                    e_in = (min(u, v), max(u, v))
                    if e_in in edge_set:
                        continue
                    if adj[u] & adj[v] != 0:
                        continue
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add(e_in)
                    
                    if not _is_3_colorable(adj):
                        swap_success = True
                        if len(edge_set) > best_count:
                            best_count = len(edge_set)
                            best_adj = adj[:]
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove(e_in)
                
                if not swap_success:
                    # Restore removed edge
                    adj[u_out] |= (1 << v_out)
                    adj[v_out] |= (1 << u_out)
                    edge_set.add(e_out)
        
        # Periodic greedy fill to pick up any available edges
        if failed_moves % 20 == 0 and failed_moves > 0:
            _greedy_fill(adj, edge_set, deadline)
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_adj = adj[:]
    
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
    deadline = start_time + 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Run multiple instances of local search with different seeds
    seed = 0
    while time.time() < deadline - 5.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        if best_count >= 55:
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
