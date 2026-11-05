"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave 35s safety margin before 150s kill
    
    best_result = {'n': 5, 'edges': []}
    
    # Try increasing n from 21 downward (21 is the target, 20 is known)
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random constructions with different seeds
        result = None
        num_attempts = 200 if n <= 20 else 50
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = n * 10000 + attempt * 137
            result = _try_random_graph(n, seed)
            if result is not None:
                break
        
        if result is not None:
            best_result = result
            if n == 21:
                break  # Found a 21-vertex graph, that's the maximum
            continue
        
        # Try local search approach
        if time.time() > deadline - 20:
            break
        result = _try_local_search(n, seed=n*5000, max_iter=2000)
        if result is not None:
            best_result = result
            if n == 21:
                break
            continue
    
    return best_result

def _count_bits(x):
    return x.bit_count()

def _is_regular_induced(adj_mask, subset):
    """Check if the induced subgraph on 'subset' is regular.
    adj_mask: list of int bitmasks, adj_mask[i] has bit j set if i-j edge exists.
    subset: tuple of vertex indices.
    """
    k = len(subset)
    if k < 6:
        return False
    
    # Build sub_mask once
    sub_mask = 0
    for u in subset:
        sub_mask |= (1 << u)
    
    # Check each vertex's degree within subset
    first_deg = None
    for v in subset:
        deg = _count_bits(adj_mask[v] & sub_mask)
        if first_deg is None:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _check_graph_valid(n, adj_mask):
    """Check if no induced subgraph on 6+ vertices is regular.
    Returns True if valid (no regular induced subgraph on 6+ vertices)."""
    vertices = list(range(n))
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            if _is_regular_induced(adj_mask, subset):
                return False
    return True

def _find_violation(n, adj_mask):
    """Find a violating subset (regular induced subgraph on 6+ vertices).
    Returns (subset, k) if found, (None, None) if none."""
    vertices = list(range(n))
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            if _is_regular_induced(adj_mask, subset):
                return (subset, k)
    return (None, None)

def _try_random_graph(n, seed=42):
    """Try to build a valid graph on n vertices via random edge assignment."""
    rng = random.Random(seed)
    # Start with empty graph, add edges randomly
    adj_mask = [0] * n
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(possible_edges)
    
    # Add edges with probability ~0.5
    for (i, j) in possible_edges:
        if rng.random() < 0.5:
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    
    if _check_graph_valid(n, adj_mask):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def _try_local_search(n, seed=42, max_iter=5000):
    """Local search: start with random graph, fix violations by flipping edges."""
    rng = random.Random(seed)
    adj_mask = [0] * n
    
    # Start with random graph
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    for it in range(max_iter):
        if time.time() > start_time + 110:
            break
        
        subset, k = _find_violation(n, adj_mask)
        if subset is None:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Pick a random vertex from the violating subset and flip one of its edges
        v = rng.choice(subset)
        # Pick a random other vertex to flip edge with
        u = rng.choice([x for x in subset if x != v])
        
        # Flip the edge (v, u)
        adj_mask[v] ^= (1 << u)
        adj_mask[u] ^= (1 << v)
    
    return None

# EVOLVE-BLOCK-END
