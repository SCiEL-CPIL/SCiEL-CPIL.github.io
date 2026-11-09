"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    # Strategy: Greedy sequential vertex addition with bitmask verification
    # Then local search to try to extend
    
    best_graph = None
    best_n = 0

    # Try greedy construction for n=20, then try to extend to 21
    for target_n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds for greedy construction
        for seed in range(200):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed * 7919 + target_n * 104729)
            edges = _greedy_construct(target_n, rng, deadline - 15)
            if edges is not None:
                if target_n > best_n:
                    best_graph = edges
                    best_n = target_n
                break
        
        if best_n >= target_n:
            break
    
    # If we have a valid graph, try local search to extend it
    if best_graph is not None and best_n < 21 and time.time() < deadline - 15:
        extended = _local_search_extend(best_graph, best_n, deadline - 5)
        if extended is not None:
            best_graph = extended
            best_n = best_n + 1
    
    if best_graph is not None:
        return {'n': best_n, 'edges': best_graph}
    
    # Fallback: try random graphs
    for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed + target_n * 1000)
            edges = _random_graph(target_n, rng)
            if _is_valid_bitmask(edges, target_n):
                if target_n > best_n:
                    best_graph = edges
                    best_n = target_n
                break
        if best_n >= target_n:
            break
    
    if best_graph is not None:
        return {'n': best_n, 'edges': best_graph}
    
    # Last resort
    return {'n': 5, 'edges': []}


def _greedy_construct(n, rng, deadline):
    """
    Build a graph vertex by vertex. For each new vertex, try all possible
    edge patterns to existing vertices and pick one that maintains validity.
    Returns edge list or None if stuck.
    """
    adj = [0] * n  # adj[i] is bitmask of neighbors of vertex i
    edges = []
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Try random edge patterns for vertex v to vertices 0..v-1
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Generate random edges from v to previous vertices
            candidate_edges = {}
            for u in range(v):
                if rng.random() < 0.5:
                    candidate_edges[u] = True
            
            # Apply candidate edges
            new_adj = adj[:]
            for u, exists in candidate_edges.items():
                if exists:
                    new_adj[v] |= (1 << u)
                    new_adj[u] |= (1 << v)
                else:
                    new_adj[v] &= ~(1 << u)
                    new_adj[u] &= ~(1 << v)
            
            # Check validity on the partial graph (vertices 0..v)
            if _is_valid_bitmask_adj(new_adj, v + 1):
                adj = new_adj
                found = True
                break
        
        if not found:
            return None
        
        # Record edges
        for u in range(v):
            if adj[v] & (1 << u):
                edges.append([u, v])
    
    return edges


def _is_valid_bitmask_adj(adj, n):
    """
    Check if the graph on vertices 0..n-1 (given by adj bitmasks) has no
    regular induced subgraph on 6+ vertices.
    Uses bitmask operations for speed.
    """
    # Check all subsets of size >= 6
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_bitmask(adj, subset, size):
                return False
    return True


def _is_regular_bitmask(adj, subset, size):
    """Check if induced subgraph on subset is regular using bitmasks."""
    degs = []
    for v in subset:
        # Count neighbors of v that are also in subset
        deg = 0
        for u in subset:
            if u != v and (adj[v] & (1 << u)):
                deg += 1
        degs.append(deg)
    # Regular if all degrees equal
    return all(d == degs[0] for d in degs)


def _is_valid_bitmask(edges, n):
    """Convert edges to adj bitmasks and check validity."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return _is_valid_bitmask_adj(adj, n)


def _find_violation_bitmask(adj, n):
    """Find a regular induced subgraph of size >= 6, or return None."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_bitmask(adj, subset, size):
                return list(subset)
    return None


def _local_search_extend(edges, n, deadline):
    """
    Try to extend a valid n-vertex graph to n+1 vertices by adding a new vertex
    and finding appropriate edges via local search.
    Returns new edge list or None.
    """
    n_new = n + 1
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    rng = random.Random(42)
    
    for attempt in range(500):
        if time.time() > deadline:
            return None
        
        # Start with random edges from new vertex to old vertices
        new_adj = adj[:] + [0]
        for i in range(n):
            if rng.random() < 0.5:
                new_adj[n] |= (1 << i)
                new_adj[i] |= (1 << n)
        
        # Local search: flip edges involving new vertex to fix violations
        improved = True
        while improved and time.time() < deadline:
            improved = False
            viol = _find_violation_bitmask(new_adj, n_new)
            if viol is None:
                # Valid! Convert to edge list
                new_edges = []
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        if new_adj[i] & (1 << j):
                            new_edges.append([i, j])
                return new_edges
            
            # Find a violation involving the new vertex
            if n in viol:
                others = [v for v in viol if v != n]
                if others:
                    v = rng.choice(others)
                    # Flip edge between n and v
                    if new_adj[n] & (1 << v):
                        new_adj[n] &= ~(1 << v)
                        new_adj[v] &= ~(1 << n)
                    else:
                        new_adj[n] |= (1 << v)
                        new_adj[v] |= (1 << n)
                    improved = True
            else:
                # Violation doesn't involve new vertex - can't fix by flipping
                # Try restarting with different random edges
                break
    
    return None


def _random_graph(n, rng):
    """Generate a random graph, return edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append([i, j])
    return edges
