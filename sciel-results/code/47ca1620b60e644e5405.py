import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # safety margin before 150s kill

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break

        required_min_deg = n - 39

        # Try many random restarts with bitmask-based greedy
        max_attempts = 500 if n <= 44 else 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            result = _greedy_bitmask(n, required_min_deg, deadline, attempt)
            if result is not None:
                return {'n': n, 'edges': result}

        # If greedy didn't find it, try local search from best partial solution
        result = _local_search_bitmask(n, required_min_deg, deadline)
        if result is not None:
            return {'n': n, 'edges': result}

    # Fallback: n=39 with empty graph (min degree 0 >= 39-39=0)
    return {'n': 39, 'edges': []}


def _check_c4_bitmask(adj_masks, u, v):
    """
    Check if adding edge (u,v) creates a C4 using bitmask operations.
    A C4 exists if there's a path u-x-y-v of length 3.
    Equivalent: exists x in N(u)\\{v} such that (N(x) & N(v) & ~(1<<u) & ~(1<<v)) != 0
    Wait, more precisely: exists x in N(u)\\{v}, y in N(x)\\{u,v} such that y in N(v).
    This means: for some x in N(u)\\{v}, (N(x) & N(v) & ~(1<<u) & ~(1<<v)) != 0
    """
    mask_u = adj_masks[u]
    mask_v = adj_masks[v]
    
    # Candidates for x: neighbors of u, excluding v
    candidates_x = mask_u & ~(1 << v)
    
    # For each x in candidates_x, check if N(x) intersects N(v) excluding u and v
    # We can do this in bulk: for all x in candidates_x, we need (N(x) & N(v) & ~(1<<u) & ~(1<<v)) != 0
    # This is equivalent to: OR over all x in candidates_x of N(x), then intersect with N(v) & ~(1<<u) & ~(1<<v)
    # But that's not quite right - we need existence of a specific x-y pair.
    
    # Actually, the correct check is:
    # There exists x in N(u)\\{v} and y in N(x)\\{u,v} such that y in N(v)
    # This is: (OR_{x in N(u)\\{v}} N(x)) & N(v) & ~(1<<u) & ~(1<<v) != 0
    # But we need to be careful: y must be in N(x) for the SAME x that's in N(u)
    
    # Let me think again. Adding edge (u,v) creates a C4 iff there exists a path of length 3 from u to v.
    # Path: u - x - y - v where x != v, y != u, x != y
    # x must be neighbor of u (x != v)
    # y must be neighbor of x (y != u, y != v)  
    # y must be neighbor of v
    
    # So: exists x in N(u)\\{v} such that exists y in (N(x) & N(v) & ~(1<<u) & ~(1<<v))
    # Which is: exists x in N(u)\\{v} such that (N(x) & N(v) & ~(1<<u) & ~(1<<v)) != 0
    
    # Efficient way: 
    # Let S = N(v) & ~(1<<u) & ~(1<<v)  (potential y values)
    # For each x in N(u)\\{v}, check if N(x) & S != 0
    
    S = mask_v & ~(1 << u) & ~(1 << v)
    if S == 0:
        return False
    
    candidates_x = mask_u & ~(1 << v)
    while candidates_x:
        # Get lowest set bit
        x_bit = candidates_x & (-candidates_x)
        x = x_bit.bit_length() - 1
        if adj_masks[x] & S:
            return True
        candidates_x ^= x_bit
    
    return False


def _greedy_bitmask(n, required_min_deg, deadline, seed):
    """
    Greedy edge addition using bitmask representation.
    """
    rng = random.Random(seed)
    
    # adj_masks[i] is a bitmask of neighbors of vertex i
    adj_masks = [0] * n
    
    # Start with empty graph, add edges greedily
    # Priority: edges that help low-degree vertices
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We'll use a priority-based approach: repeatedly pick the edge that
    # most helps the minimum degree vertex, if it doesn't create a C4
    
    degrees = [0] * n
    
    # Shuffle edges initially for diversity
    rng.shuffle(all_edges)
    
    # Process edges in shuffled order, but with a twist:
    # First pass: add all edges that don't create C4
    # Then iterate to boost min degree
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not (adj_masks[u] >> v) & 1 and not (adj_masks[v] >> u) & 1:
            if not _check_c4_bitmask(adj_masks, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
    
    # Now try to boost min degree by adding more edges
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        min_deg = min(degrees)
        if min_deg >= required_min_deg:
            break
        
        # Find vertices with min degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        rng.shuffle(min_vertices)
        
        added = False
        for u in min_vertices:
            if time.time() > deadline:
                break
            
            # Try to add edges from u to non-neighbors
            candidates = [v for v in range(n) if v != u and not ((adj_masks[u] >> v) & 1)]
            # Sort by degree (prefer lower degree neighbors? or higher?)
            # Prefer adding to vertices that will also benefit
            candidates.sort(key=lambda v: degrees[v])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_bitmask(adj_masks, u, v):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            break
    
    # Check if we met the requirement
    if min(degrees) >= required_min_deg:
        return _extract_edges_bitmask(adj_masks, n)
    
    return None


def _local_search_bitmask(n, required_min_deg, deadline):
    """
    Local search: start from a greedy solution, try edge swaps to improve min degree.
    """
    rng = random.Random(12345)
    
    # First, get a greedy solution
    best_adj = None
    best_min_deg = -1
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        adj_masks = [0] * n
        degrees = [0] * n
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.Random(seed).shuffle(all_edges)
        
        for u, v in all_edges:
            if not (adj_masks[u] >> v) & 1:
                if not _check_c4_bitmask(adj_masks, u, v):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        min_deg = min(degrees)
        if min_deg > best_min_deg:
            best_min_deg = min_deg
            best_adj = adj_masks[:]
    
    if best_adj is None:
        return None
    
    # Now do local search: try to add edges to boost min degree
    adj_masks = best_adj[:]
    degrees = [bin(adj_masks[i]).count('1') for i in range(n)]
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        min_deg = min(degrees)
        if min_deg >= required_min_deg:
            return _extract_edges_bitmask(adj_masks, n)
        
        # Find min-degree vertices
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        rng.shuffle(min_vertices)
        
        added = False
        for u in min_vertices:
            if time.time() > deadline:
                break
            
            candidates = [v for v in range(n) if v != u and not ((adj_masks[u] >> v) & 1)]
            rng.shuffle(candidates)
            
            for v in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_bitmask(adj_masks, u, v):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            # Try removing an edge and re-adding to free up C4 constraints
            # This is complex; for now, just break
            break
    
    if min(degrees) >= required_min_deg:
        return _extract_edges_bitmask(adj_masks, n)
    
    return None


def _extract_edges_bitmask(adj_masks, n):
    """Extract edge list from bitmask adjacency."""
    edges = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if (mask >> v) & 1:
                edges.append([u, v])
    return edges


# EVOLVE-BLOCK-END
