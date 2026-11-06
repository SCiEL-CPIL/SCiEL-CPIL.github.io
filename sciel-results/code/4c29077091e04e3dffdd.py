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
    """
    n = len(adj)
    colors = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        mask = 0
        # Check neighbors
        neighbors = adj[v]
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            neighbors ^= lb
            if colors[u] != -1:
                mask |= (1 << colors[u])
        
        available = ~mask & 0b111
        
        for c in range(3):
            if available & (1 << c):
                colors[v] = c
                if backtrack(v + 1):
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
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    non_edges = [e for e in all_edges if e not in edge_set]
    
    best_count = len(edge_set)
    best_adj = adj[:]
    
    while time.time() < deadline:
        # 1. Try to add a random non-edge
        added = False
        if non_edges:
            # Shuffle non_edges occasionally or pick random index
            idx = rng.randrange(len(non_edges))
            u, v = non_edges[idx]
            # Check triangle-free
            if adj[u] & adj[v] == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                non_edges.pop(idx)
                added = True
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_adj = adj[:]
        
        # 2. If no edge added, try a swap (remove one, add one)
        if not added:
            # Pick a random edge to remove
            if not edge_set:
                break
            e_out = rng.choice(list(edge_set))
            u_out, v_out = e_out
            
            # Remove it
            adj[u_out] &= ~(1 << v_out)
            adj[v_out] &= ~(1 << u_out)
            edge_set.remove(e_out)
            
            # Try to add a random non-edge that was previously blocked or just random
            # We need to find a valid non-edge in the new state
            # Note: e_out is now a non-edge
            # We can try e_out itself? No, that's just reverting.
            # Try other non-edges
            
            # Collect candidate non-edges
            # To be efficient, we can just pick a few random ones from the current non_edges list
            # But non_edges list is stale (it doesn't include e_out, and might include edges that became valid)
            # Let's just pick a random pair from all possible non-edges
            
            # A simple heuristic: Try a few random swaps
            swap_success = False
            for _ in range(50): # Try 50 random swaps
                if time.time() > deadline:
                    break
                # Pick a random non-edge from the full set (excluding existing edges)
                # To avoid O(N^2) check, we can maintain a list of non-edges
                # But maintaining it is complex.
                # Instead, pick a random pair (u, v) not in edge_set
                
                # Generate random non-edge
                while True:
                    u = rng.randrange(n)
                    v = rng.randrange(n)
                    if u == v:
                        continue
                    e_in = (min(u, v), max(u, v))
                    if e_in not in edge_set:
                        break
                
                # Check if e_in is valid in current adj
                if adj[u] & adj[v] == 0:
                    # Check if graph is still 4-chromatic
                    # Temporarily add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add(e_in)
                    
                    if not _is_3_colorable(adj):
                        # Valid swap
                        swap_success = True
                        if len(edge_set) > best_count:
                            best_count = len(edge_set)
                            best_adj = adj[:]
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove(e_in)
            
            # If no swap succeeded, revert the removal of e_out
            if not swap_success:
                adj[u_out] |= (1 << v_out)
                adj[v_out] |= (1 << u_out)
                edge_set.add(e_out)
        
        # If we are stuck for a long time, restart with a perturbation?
        # For now, just keep trying.
        
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
        # If we find a very dense graph, we might want to stop early?
        # Max possible is 64. If we reach 50+, it's good.
        if best_count >= 55:
            break

    # Final validation
    if best_edges:
        if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)):
            # Fallback
            grotzsch = _build_grotzsch_edges()
            best_edges = list(grotzsch)
    else:
        best_edges = _build_grotzsch_edges()
    
    return best_edges
# EVOLVE-BLOCK-END
