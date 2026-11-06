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

def _greedy_add_edges(adj, edge_set, non_edges, deadline, rng):
    """
    Greedily add as many valid non-edges as possible.
    Returns the number of edges added.
    """
    added_count = 0
    # Repeatedly try to add edges until no more can be added or time runs out
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Try all non-edges in random order
        indices = list(range(len(non_edges)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            u, v = non_edges[idx]
            if adj[u] & adj[v] == 0:
                # Add this edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                non_edges.pop(idx)
                added_count += 1
                progress = True
                # After adding an edge, some other non-edges may become invalid
                # So we need to restart the scan
                break
    return added_count

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
    
    # First, greedily add as many edges as possible
    _greedy_add_edges(adj, edge_set, non_edges, deadline, rng)
    if len(edge_set) > best_count:
        best_count = len(edge_set)
        best_adj = adj[:]
    
    # Now do local search with swaps
    stuck_count = 0
    while time.time() < deadline:
        # Try to greedily add more edges
        added = _greedy_add_edges(adj, edge_set, non_edges, deadline, rng)
        if added > 0:
            stuck_count = 0
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_adj = adj[:]
            continue
        
        # No edges could be added, so try a swap
        stuck_count += 1
        if stuck_count > 100:
            # Perturb: remove a random edge and restart greedy
            if not edge_set:
                break
            e_out = rng.choice(list(edge_set))
            u_out, v_out = e_out
            adj[u_out] &= ~(1 << v_out)
            adj[v_out] &= ~(1 << u_out)
            edge_set.remove(e_out)
            non_edges.append(e_out)
            stuck_count = 0
            continue
        
        # Try a swap: remove one edge, add one
        if not edge_set:
            break
        e_out = rng.choice(list(edge_set))
        u_out, v_out = e_out
        
        # Remove it
        adj[u_out] &= ~(1 << v_out)
        adj[v_out] &= ~(1 << u_out)
        edge_set.remove(e_out)
        non_edges.append(e_out)
        
        # Try to add a random non-edge
        swap_success = False
        for _ in range(20):
            if time.time() > deadline:
                break
            if not non_edges:
                break
            idx = rng.randrange(len(non_edges))
            u, v = non_edges[idx]
            
            # Check if this edge is valid (triangle-free)
            if adj[u] & adj[v] == 0:
                # Temporarily add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                non_edges.pop(idx)
                
                # Check if still 4-chromatic
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
                    edge_set.remove((u, v))
                    non_edges.append((u, v))
        
        # If no swap succeeded, revert the removal of e_out
        if not swap_success:
            adj[u_out] |= (1 << v_out)
            adj[v_out] |= (1 << u_out)
            edge_set.add(e_out)
            # Remove e_out from non_edges if present
            try:
                non_edges.remove(e_out)
            except ValueError:
                pass
    
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

    # Final validation: Must be triangle-free AND NOT 3-colorable
    if best_edges:
        adj = _make_bitmask_adj(best_edges)
        if _is_valid_triangle_free(adj) and not _is_3_colorable(adj):
            return best_edges
        else:
            # If the best found is invalid, fall back to a known valid graph
            # Grötzsch graph is 4-chromatic and triangle-free, but only 11 vertices.
            # We need 16. We can just return the Grötzsch graph padded with isolated vertices?
            # No, the task implies a graph on 16 vertices. Isolated vertices are allowed.
            # The Grötzsch graph on 0..10 plus 11..15 isolated is a valid triangle-free 4-chromatic graph on 16 vertices.
            grotzsch = _build_grotzsch_edges()
            return grotzsch
    else:
        # Fallback
        return _build_grotzsch_edges()
# EVOLVE-BLOCK-END
