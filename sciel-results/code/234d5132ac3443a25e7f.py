"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _is_regular_induced(adj_mask, n, subset):
    """Check if the induced subgraph on 'subset' is regular.
    adj_mask: list of int bitmasks, adj_mask[i] has bit j set if i-j edge exists.
    subset: tuple of vertex indices.
    """
    k = len(subset)
    if k < 6:
        return False
    # Compute degree of each vertex in the subset
    degrees = []
    for v in subset:
        deg = 0
        mask = adj_mask[v]
        # Count bits in mask that are also in subset
        sub_mask = 0
        for u in subset:
            sub_mask |= (1 << u)
        common = mask & sub_mask
        deg = bin(common).count('1')
        degrees.append(deg)
    # Regular means all degrees equal
    return all(d == degrees[0] for d in degrees)

def _check_graph_valid(n, adj_mask):
    """Check if no induced subgraph on 6+ vertices is regular.
    Returns True if valid."""
    # Check all subsets of size 6 through n
    # For efficiency, only check size 6 first; if none are regular, check larger
    # Actually, we need to check all sizes >= 6
    # But if a subset of size k is regular, it may not contain a regular subset of size 6
    # So we must check all sizes
    # Optimization: for size k, a regular induced subgraph has each vertex with degree (k-1)/2 if k is odd,
    # or k/2 - 1 or k/2 if... wait, regular means all same degree, not necessarily (k-1)/2
    # Let's just check all subsets of size 6, 7, ..., n
    
    vertices = list(range(n))
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            if _is_regular_induced(adj_mask, n, subset):
                return False
    return True

def _check_graph_valid_fast(n, adj_mask):
    """Faster check using precomputed subset masks."""
    vertices = list(range(n))
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            k = len(subset)
            # Build sub_mask once
            sub_mask = 0
            for u in subset:
                sub_mask |= (1 << u)
            # Check each vertex's degree within subset
            first_deg = None
            regular = True
            for v in subset:
                deg = bin(adj_mask[v] & sub_mask).count('1')
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    regular = False
                    break
            if regular:
                return False
    return True

def _try_build_graph(n, num_tries=1000, seed=42):
    """Try to build a valid graph on n vertices via random edge assignment."""
    rng = random.Random(seed)
    for attempt in range(num_tries):
        # Random graph with probability ~0.5
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        if _check_graph_valid_fast(n, adj_mask):
            # Extract edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    return None

def _try_greedy_build(n, seed=42):
    """Greedy approach: start with empty graph, add edges that don't create regular induced subgraphs."""
    rng = random.Random(seed)
    adj_mask = [0] * n
    # Try to add random edges one by one
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(possible_edges)
    for (i, j) in possible_edges:
        # Tentatively add edge
        adj_mask[i] |= (1 << j)
        adj_mask[j] |= (1 << i)
        # Check validity
        if _check_graph_valid_fast(n, adj_mask):
            pass  # Keep the edge
        else:
            # Remove the edge
            adj_mask[i] &= ~(1 << j)
            adj_mask[j] &= ~(1 << i)
    # Check final validity
    if _check_graph_valid_fast(n, adj_mask):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave 10s safety margin
    
    best_result = {'n': 5, 'edges': []}
    
    # Try increasing n from 6 upward
    for n in range(6, 23):
        if time.time() > deadline - 5:
            break
        
        # Try random construction
        result = _try_build_graph(n, num_tries=50, seed=n*1000)
        if result is not None:
            best_result = result
            continue  # Try larger n
        
        # Try greedy construction
        if time.time() > deadline - 10:
            break
        result = _try_greedy_build(n, seed=n*2000)
        if result is not None:
            best_result = result
            continue  # Try larger n
    
    return best_result
# EVOLVE-BLOCK-END
