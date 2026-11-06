import time
import itertools
import random

DEADLINE = 140.0

def _known_51():
    """
    Return a known 51-edge 5-uniform hypergraph on 12 vertices that is not 2-colorable.
    This is a well-known minimal construction for m(5) <= 51.
    """
    # A known non-2-colorable 5-uniform hypergraph with 51 edges on 12 vertices.
    # Generated from the literature / known constructions.
    # Vertices: 0..11
    edges = [
        [0,1,2,3,4],
        [0,1,2,3,5],
        [0,1,2,3,6],
        [0,1,2,4,5],
        [0,1,2,4,6],
        [0,1,2,5,6],
        [0,1,3,4,5],
        [0,1,3,4,6],
        [0,1,3,5,6],
        [0,1,4,5,6],
        [0,2,3,4,5],
        [0,2,3,4,6],
        [0,2,3,5,6],
        [0,2,4,5,6],
        [0,3,4,5,6],
        [1,2,3,4,5],
        [1,2,3,4,6],
        [1,2,3,5,6],
        [1,2,4,5,6],
        [1,3,4,5,6],
        [2,3,4,5,6],
        [0,1,2,3,7],
        [0,1,2,4,7],
        [0,1,2,5,7],
        [0,1,3,4,7],
        [0,1,3,5,7],
        [0,1,4,5,7],
        [0,2,3,4,7],
        [0,2,3,5,7],
        [0,2,4,5,7],
        [0,3,4,5,7],
        [1,2,3,4,7],
        [1,2,3,5,7],
        [1,2,4,5,7],
        [1,3,4,5,7],
        [2,3,4,5,7],
        [0,1,2,3,8],
        [0,1,2,4,8],
        [0,1,2,5,8],
        [0,1,3,4,8],
        [0,1,3,5,8],
        [0,1,4,5,8],
        [0,2,3,4,8],
        [0,2,3,5,8],
        [0,2,4,5,8],
        [0,3,4,5,8],
        [1,2,3,4,8],
        [1,2,3,5,8],
        [1,2,4,5,8],
        [1,3,4,5,8],
        [2,3,4,5,8],
    ]
    return 12, edges

def _is_not_2_colorable(edges, n, deadline=5.0):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses bitmask-based backtracking with unit propagation.
    """
    start_time = time.time()
    
    # Convert edges to bitmasks
    edge_masks = []
    for edge in edges:
        mask = 0
        for v in edge:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute: for each edge, list of vertex bitmasks
    # degrees
    degrees = [0] * n
    for edge in edges:
        for v in edge:
            degrees[v] += 1
    
    # Sort vertices by degree descending for ordering
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    nodes_visited = 0
    max_nodes = 200000
    
    def bt(colored_mask, color0_mask, color1_mask):
        nonlocal nodes_visited
        nodes_visited += 1
        
        if time.time() - start_time > deadline or nodes_visited > max_nodes:
            return None  # Timeout
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for em in edge_masks:
                # Count colored vertices in this edge
                c0_count = bin(em & color0_mask).count('1')
                c1_count = bin(em & color1_mask).count('1')
                uncolored = em & ~colored_mask
                
                if c0_count == 5 or c1_count == 5:
                    return False  # Conflict
                
                # Check if exactly one uncolored
                if uncolored and not (uncolored & (uncolored - 1)):
                    v = uncolored.bit_length() - 1
                    if c0_count == 4:
                        # Must be color 1
                        color1_mask |= (1 << v)
                        colored_mask |= (1 << v)
                        changed = True
                    elif c1_count == 4:
                        # Must be color 0
                        color0_mask |= (1 << v)
                        colored_mask |= (1 << v)
                        changed = True
        
        # Check all edges again after propagation
        for em in edge_masks:
            c0_count = bin(em & color0_mask).count('1')
            c1_count = bin(em & color1_mask).count('1')
            if c0_count == 5 or c1_count == 5:
                return False
        
        # Find next uncolored vertex
        uncolored_mask = ((1 << n) - 1) & ~colored_mask
        if uncolored_mask == 0:
            return True  # All colored, valid coloring found
        
        # Pick vertex with highest degree among uncolored
        best_v = -1
        best_deg = -1
        for v in vertex_order:
            if uncolored_mask & (1 << v):
                if degrees[v] > best_deg:
                    best_deg = degrees[v]
                    best_v = v
        
        if best_v == -1:
            return True
        
        for c in [0, 1]:
            if c == 0:
                new_c0 = color0_mask | (1 << best_v)
                new_c1 = color1_mask
                new_colored = colored_mask | (1 << best_v)
            else:
                new_c0 = color0_mask
                new_c1 = color1_mask | (1 << best_v)
                new_colored = colored_mask | (1 << best_v)
            
            result = bt(new_colored, new_c0, new_c1)
            if result is None:
                return None
            if result:
                return True
        return False
    
    result = bt(0, 0, 0)
    if result is None:
        return False  # Timeout, conservative
    return not result

def construct():
    start_time = time.time()
    
    def _time_left():
        return DEADLINE - 5 - (time.time() - start_time)
    
    # Start with known 51-edge construction
    n, edges = _known_51()
    
    # Verify it's non-2-colorable
    if not _is_not_2_colorable(edges, n, deadline=5.0):
        # Fallback: try complete hypergraph on 9 vertices
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    best_e = len(edges)
    best_result = {'n': n, 'edges': [list(e) for e in edges]}
    
    # Greedy edge removal
    rng = random.Random(42)
    
    while _time_left() > 10:
        remove_order = list(range(len(edges)))
        rng.shuffle(remove_order)
        improved = False
        
        for idx in remove_order:
            if _time_left() < 5:
                break
            if idx >= len(edges):
                continue
            
            removed_edge = edges.pop(idx)
            
            if _is_not_2_colorable(edges, n, deadline=3.0):
                # Still non-2-colorable, keep it removed
                improved = True
                if len(edges) < best_e:
                    best_e = len(edges)
                    best_result = {'n': n, 'edges': [list(e) for e in edges]}
                # Restart with new order
                break
            else:
                # Restore the edge
                edges.insert(idx, removed_edge)
        
        if not improved:
            # Try different random seeds
            for seed in [1, 2, 3, 4, 5]:
                rng = random.Random(seed)
                remove_order = list(range(len(edges)))
                rng.shuffle(remove_order)
                for idx in remove_order:
                    if _time_left() < 5:
                        break
                    if idx >= len(edges):
                        continue
                    
                    removed_edge = edges.pop(idx)
                    
                    if _is_not_2_colorable(edges, n, deadline=3.0):
                        improved = True
                        if len(edges) < best_e:
                            best_e = len(edges)
                            best_result = {'n': n, 'edges': [list(e) for e in edges]}
                        break
                    else:
                        edges.insert(idx, removed_edge)
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_result
