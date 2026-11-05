"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _popcount(x):
    """Count set bits in integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def _is_regular_subset(adj_bits, subset_mask, n):
    """Check if the induced subgraph on subset_mask is regular."""
    # Count vertices in subset
    size = _popcount(subset_mask)
    if size < 6:
        return False
    
    # Get degrees of each vertex in the subset
    degrees = []
    temp = subset_mask
    while temp:
        v = (temp & -temp).bit_length() - 1
        temp &= temp - 1
        # Count neighbors of v within subset
        deg = _popcount(adj_bits[v] & subset_mask)
        degrees.append(deg)
    
    # Check if all degrees are equal
    if not degrees:
        return False
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _has_regular_subset(adj_bits, n):
    """Check if any 6+ vertex subset induces a regular subgraph."""
    # Check all subsets of size >= 6
    # Use bitmasks for efficiency
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset(adj_bits, mask, n):
                return True
    return False

def _adj_to_edges(adj_bits, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_bits[u] & (1 << v):
                edges.append([u, v])
    return edges

def _construct_deterministic_20():
    """Try to construct a valid 20-vertex graph using a deterministic approach."""
    n = 20
    # Start with a base structure and try to build up
    # Use a greedy approach: start empty, add edges one by one
    # while maintaining the property
    
    # First, try a known construction: complement of a specific graph
    # or use a random seed with deterministic verification
    
    # Try multiple random seeds deterministically
    for seed in range(200):
        random.seed(seed)
        adj = [0] * n
        # Generate random graph with moderate density
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if not _has_regular_subset(adj, n):
            return {'n': n, 'edges': _adj_to_edges(adj, n)}
    
    # If no random graph works, try local search from a base
    # Start with a cycle or path
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    # Try to add edges while maintaining validity
    for attempt in range(50):
        new_adj = adj[:]
        # Try adding a random edge
        for _ in range(100):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not (new_adj[u] & (1 << v)):
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not _has_regular_subset(new_adj, n):
                    adj = new_adj
                    break
                else:
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
    
    if not _has_regular_subset(adj, n):
        return {'n': n, 'edges': _adj_to_edges(adj, n)}
    
    # Fallback: return a smaller valid graph
    # Try n=19
    for seed in range(100):
        random.seed(seed + 1000)
        adj = [0] * 19
        for i in range(19):
            for j in range(i + 1, 19):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if not _has_regular_subset(adj, 19):
            return {'n': 19, 'edges': _adj_to_edges(adj, 19)}
    
    # Last resort: empty graph on 5 vertices
    return {'n': 5, 'edges': []}

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try to construct a 20-vertex graph
    result = _construct_deterministic_20()
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Verify the result
        n = result['n']
        if n >= 6:
            # Rebuild adjacency and verify
            adj = [0] * n
            for u, v in result['edges']:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            if _has_regular_subset(adj, n):
                # Invalid, return fallback
                return {'n': 5, 'edges': []}
    
    return result
# EVOLVE-BLOCK-END
