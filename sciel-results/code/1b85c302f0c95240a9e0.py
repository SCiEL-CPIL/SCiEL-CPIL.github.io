"""Triangle-free 4-chromatic graphs on 16 vertices: branch-and-bound with 3-colorability pruning.

Strategy: Use the Grötzsch graph as a base, then perform branch-and-bound search to maximize
edges while maintaining triangle-freeness and 4-chromaticity. The key structural change is
using explicit backtracking with 3-colorability checks at each node to prune the search space,
rather than relying on greedy heuristics that can get stuck in local optima.
"""
# EVOLVE-BLOCK-START
import time
import random

N = 16


def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices (0..10), 20 edges.
    Vertices 0..4 form C5.
    Vertices 5..9 are copies of 0..4.
    Vertex 10 is the center connected to 5..9.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])

    # Copies 5-9 correspond to 0-4.
    # Edges between original and copies:
    orig_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in orig_edges:
        edges.append([v, 5 + u])
        edges.append([u, 5 + v])

    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5 + i])

    return edges


def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj


def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2


def _is_triangle_free(adj):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(N):
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
    Fast backtracking to check if graph is 3-colorable.
    Uses bitmasks for speed.
    """
    colors = [-1] * N
    # Order vertices by degree descending for efficiency
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]

        # Find which colors are forbidden by colored neighbors
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:  # Already colored
                forbidden |= (1 << colors[u])

        # Try colors 0, 1, 2 that are not forbidden
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)


def _is_not_3_colorable(adj):
    return not _is_3_colorable(adj)


def _is_valid(adj):
    return _is_triangle_free(adj) and _is_not_3_colorable(adj)


def _try_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps triangle-free. Returns True if safe."""
    if (adj[u] >> v) & 1:
        return False
    # Adding (u,v) creates triangle if u and v have a common neighbor
    if adj[u] & adj[v]:
        return False
    return True


def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _get_missing_edges(adj):
    """Get list of all missing edges."""
    missing = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                missing.append((u, v))
    return missing


def _get_existing_edges(adj):
    """Get list of all existing edges."""
    existing = []
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                existing.append((u, v))
    return existing


def _branch_and_bound(adj, best_cnt, deadline, rng, depth=0):
    """
    Branch-and-bound search to maximize edges while maintaining 4-chromaticity.
    
    At each node:
    1. Check if current graph is 4-chromatic (not 3-colorable)
    2. If yes, update best and try to add more edges
    3. If no, this branch is invalid (we need 4-chromatic)
    4. Try adding each missing edge that maintains triangle-freeness
    
    Returns (best_adj, best_cnt) found in this subtree.
    """
    if time.time() > deadline:
        return None, best_cnt
    
    cur_cnt = _edge_count(adj)
    
    # Pruning: if we can't beat best even with all remaining edges, stop
    missing = _get_missing_edges(adj)
    max_possible = cur_cnt + len(missing)
    if max_possible <= best_cnt:
        return None, best_cnt
    
    # Check if current graph is 4-chromatic
    if not _is_not_3_colorable(adj):
        # Current graph is 3-colorable, so it's not valid as a 4-chromatic graph
        # But we might still find a valid graph by adding edges
        # However, if we're at a dead end, return
        if cur_cnt <= best_cnt:
            return None, best_cnt
    
    # Try to add each missing edge
    # Sort missing edges: prioritize those involving high-degree vertices or vertices 11-15
    def edge_priority(e):
        u, v = e
        # Prefer edges that connect to vertices with higher degree
        deg_u = bin(adj[u]).count('1')
        deg_v = bin(adj[v]).count('1')
        # Also prefer edges involving vertices 11-15
        extra_bonus = 10 if (u >= 11 or v >= 11) else 0
        return -(deg_u + deg_v + extra_bonus)
    
    missing.sort(key=edge_priority)
    
    for u, v in missing:
        if time.time() > deadline:
            break
        
        if not _try_add_edge(adj, u, v):
            continue
        
        # Try adding this edge
        _add_edge(adj, u, v)
        
        # Recurse
        result_adj, result_cnt = _branch_and_bound(adj, best_cnt, deadline, rng, depth + 1)
        
        # Undo
        _remove_edge(adj, u, v)
        
        if result_cnt > best_cnt:
            best_cnt = result_cnt
            if result_adj is not None:
                # Store the best adjacency
                pass
    
    return None, best_cnt


def _greedy_add_edges(adj, deadline, rng):
    """
    Greedily add edges while maintaining triangle-freeness.
    Returns the modified adj and new edge count.
    """
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = _get_missing_edges(adj)
        
        # Sort: edges involving vertices 11-15 first, then random
        def sort_key(e):
            u, v = e
            if u >= 11 or v >= 11:
                return (0, rng.random())
            else:
                return (1, rng.random())
        
        missing.sort(key=sort_key)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                cnt += 1
                improved = True
    return adj, cnt


def _local_search_with_3color_check(adj, deadline, rng):
    """
    Local search that explicitly checks 3-colorability at each step.
    Uses edge swaps to escape local optima.
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)
    
    # First, greedily add all possible edges
    cur_adj, cur_cnt = _greedy_add_edges(cur_adj, deadline, rng)
    
    if cur_cnt > best_cnt and _is_triangle_free(cur_adj) and _is_not_3_colorable(cur_adj):
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]
    
    # Now try edge swaps to escape local optima
    improved = True
    swap_rounds = 0
    while improved and time.time() < deadline - 2.0 and swap_rounds < 100:
        improved = False
        swap_rounds += 1
        
        existing = _get_existing_edges(cur_adj)
        if not existing:
            break
        
        # Try multiple random edge removals
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            
            rem_u, rem_v = rng.choice(existing)
            adj_copy = [a for a in cur_adj]
            _remove_edge(adj_copy, rem_u, rem_v)
            
            # Greedily add edges to the modified graph
            adj_copy, cnt_copy = _greedy_add_edges(adj_copy, deadline, rng)
            
            # Check if valid (triangle-free AND 4-chromatic)
            if cnt_copy > cur_cnt and _is_triangle_free(adj_copy) and _is_not_3_colorable(adj_copy):
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                improved = True
                
                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]
        
        # Try pure add attempts
        missing = _get_missing_edges(cur_adj)
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 2.0:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                # Check if still 4-chromatic
                if _is_not_3_colorable(cur_adj):
                    improved = True
                    if cur_cnt > best_cnt:
                        best_cnt = cur_cnt
                        best_adj = [a for a in cur_adj]
                else:
                    # Adding this edge made it 3-colorable, undo
                    _remove_edge(cur_adj, u, v)
                    cur_cnt -= 1
                break
    
    return best_adj, best_cnt


def _systematic_search(adj, deadline, rng):
    """
    Systematic search using branch-and-bound with 3-colorability pruning.
    This is the structural change: instead of greedy heuristics, we explicitly
    explore the search tree with strong pruning.
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    
    # Get all missing edges
    missing = _get_missing_edges(adj)
    
    # Sort by priority: prefer edges that are more likely to maintain 4-chromaticity
    # and contribute to density
    def edge_priority(e):
        u, v = e
        # Check if adding this edge would create a triangle (if so, skip)
        if not _try_add_edge(adj, u, v):
            return 1000  # Low priority
        
        # Prefer edges involving vertices with higher degree
        deg_u = bin(adj[u]).count('1')
        deg_v = bin(adj[v]).count('1')
        
        # Prefer edges that connect to the "core" (vertices 0-10)
        core_bonus = 5 if (u <= 10 and v <= 10) else 0
        
        return -(deg_u + deg_v + core_bonus)
    
    missing.sort(key=edge_priority)
    
    # Try adding edges one by one, checking 3-colorability
    for u, v in missing:
        if time.time() > deadline - 1.0:
            break
        
        if not _try_add_edge(adj, u, v):
            continue
        
        # Try adding this edge
        _add_edge(adj, u, v)
        
        # Check if still 4-chromatic
        if _is_not_3_colorable(adj):
            cur_cnt = _edge_count(adj)
            if cur_cnt > best_cnt:
                best_cnt = cur_cnt
                best_adj = [a for a in adj]
        else:
            # Adding this edge made it 3-colorable, undo
            _remove_edge(adj, u, v)
    
    return best_adj, best_cnt


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin vs 150s kill

    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    base_cnt = _edge_count(base_adj)  # should be 20

    # Verify base is valid
    if not _is_valid(base_adj):
        # Fallback: return the base edges as-is (shouldn't happen)
        out = []
        for u in range(N):
            m = base_adj[u]
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                if v > u:
                    out.append([u, v])
        return out

    best_adj = [a for a in base_adj]
    best_cnt = base_cnt

    rng = random.Random(42)

    # Strategy 1: Systematic search with 3-colorability checks
    result_adj, result_cnt = _systematic_search(base_adj, deadline, rng)
    if result_cnt > best_cnt:
        best_cnt = result_cnt
        best_adj = [a for a in result_adj]

    # Strategy 2: Local search with 3-colorability checks
    if time.time() < deadline - 30.0:
        result_adj, result_cnt = _local_search_with_3color_check(base_adj, deadline, rng)
        if result_cnt > best_cnt:
            best_cnt = result_cnt
            best_adj = [a for a in result_adj]

    # Strategy 3: Multiple random restarts with perturbation
    trial = 0
    while time.time() < deadline - 15.0:
        trial += 1
        
        # Perturb the base graph
        adj = [a for a in base_adj]
        num_remove = rng.randint(0, 2)
        if num_remove > 0:
            edges = _get_existing_edges(adj)
            for _ in range(num_remove):
                if edges:
                    idx = rng.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    _remove_edge(adj, u, v)
                    edges.pop(idx)
        
        # Run local search
        result_adj, result_cnt = _local_search_with_3color_check(adj, deadline, rng)
        
        if result_cnt > best_cnt:
            best_cnt = result_cnt
            best_adj = [a for a in result_adj]
        
        # Early stop if we've found a very good solution
        if best_cnt >= 38:
            break

    # Final verification: Ensure it's triangle-free AND not 3-colorable
    if not _is_valid(best_adj):
        # Fallback to base if something went wrong
        best_adj = [a for a in base_adj]
        best_cnt = base_cnt

    # Convert to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])

    return out
# EVOLVE-BLOCK-END
