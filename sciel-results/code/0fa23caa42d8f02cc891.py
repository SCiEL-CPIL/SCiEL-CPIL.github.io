"""Improved program using threshold graph initialization and optimized search."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _is_regular_subset(adj, n, subset):
    """Check if induced subgraph on subset is regular."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
    degrees = []
    for v in subset:
        d = (adj[v] & mask).bit_count()
        degrees.append(d)
    return len(set(degrees)) == 1

def _count_reg_subgraphs_fast(adj, n, k):
    """Count regular induced subgraphs of size k using precomputed subsets."""
    count = 0
    vertices = range(n)
    for subset in itertools.combinations(vertices, k):
        if _is_regular_subset(adj, n, subset):
            count += 1
    return count

def _has_violation_full(adj, n):
    """Check if any induced subgraph of size >= 6 is regular."""
    for k in range(6, n + 1):
        if _count_reg_subgraphs_fast(adj, n, k) > 0:
            return True
    return False

def _generate_threshold_graph(n):
    """
    Generate a random threshold graph.
    Threshold graphs are constructed by adding vertices one by one,
    each either as an isolated vertex or a universal vertex.
    These graphs are structurally irregular and unlikely to contain
    regular induced subgraphs of size >= 6.
    """
    adj = [0] * n
    # Randomly decide the type of each vertex (0: isolated, 1: universal)
    # The order of addition is 0 to n-1.
    # When adding vertex i, it connects to all previous vertices if it is universal,
    # or to none if it is isolated.
    for i in range(n):
        if i == 0:
            continue
        if random.random() < 0.5:
            # Universal: connect to all previous vertices
            for j in range(i):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Else: Isolated, no edges to previous vertices
    return adj

def _find_best_edge_flip(adj, n, violating_subsets, deadline):
    """
    Find the edge flip that resolves the most violating 6-subsets.
    """
    if not violating_subsets:
        return None
    
    edge_score = {}
    
    for subset in violating_subsets:
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                key = (min(u,v), max(u,v))
                edge_score[key] = edge_score.get(key, 0) + 1
    
    if not edge_score:
        return None
    
    max_score = max(edge_score.values())
    candidates = [k for k, v in edge_score.items() if v == max_score]
    return random.choice(candidates)

def _local_search(adj, n, deadline, max_iter=1000):
    """
    Local search to fix violations.
    Returns True if a valid graph is found.
    """
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Find all violating 6-subsets
        violating = []
        vertices = range(n)
        for subset in itertools.combinations(vertices, 6):
            if _is_regular_subset(adj, n, subset):
                violating.append(subset)
        
        if not violating:
            if not _has_violation_full(adj, n):
                return True
            else:
                # Try random flip to escape
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
        
        best_edge = _find_best_edge_flip(adj, n, violating, deadline)
        if best_edge is None:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
        else:
            u, v = best_edge
        
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 5
    best_edges = []
    
    # Try n=21 first, then 20, 19...
    # Threshold graphs are very likely to be valid for n=20, and maybe 21.
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        # More restarts for higher n
        max_restarts = 50 if n <= 20 else 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with threshold graph
            adj = _generate_threshold_graph(n)
            
            # Quick check: is it already valid?
            if not _has_violation_full(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                return {'n': best_n, 'edges': best_edges}
            
            # Use local search to fix violations
            if _local_search(adj, n, deadline):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                return {'n': best_n, 'edges': best_edges}
            
            # If local search fails, try a few random perturbations + local search
            # This helps escape local minima in the threshold graph landscape
            if time.time() < deadline - 2:
                for _ in range(5):
                    if time.time() > deadline - 2:
                        break
                    # Randomly flip a few edges to perturb the threshold structure
                    for _ in range(3):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u == v: continue
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    
                    if _local_search(adj, n, deadline, max_iter=200):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        return {'n': best_n, 'edges': best_edges}

    # Fallback
    if best_n == 5:
        for n in [15, 14, 13]:
            if time.time() > deadline - 2:
                break
            adj = _generate_threshold_graph(n)
            if not _has_violation_full(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
