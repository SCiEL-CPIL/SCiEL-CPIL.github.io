"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def _is_regular_subgraph(adj_bits, mask):
    """Check if the induced subgraph on vertices in mask is regular."""
    n = int(bin(mask).count('1'))
    if n < 6:
        return False
    # For each vertex in mask, count degree within mask
    degrees = []
    for v in range(22):
        if mask & (1 << v):
            deg = bin(adj_bits[v] & mask).count('1') - 1  # exclude self
            degrees.append(deg)
    # Check if all degrees are equal
    return all(d == degrees[0] for d in degrees)

def _has_regular_6plus(adj_bits, n):
    """Check if any subset of 6+ vertices induces a regular subgraph."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subgraph(adj_bits, mask):
                return True
    return False

def _build_adj_bits(n, edges):
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _verify_graph(n, edges):
    """Verify that no subset of 6+ vertices induces a regular subgraph."""
    adj_bits = _build_adj_bits(n, edges)
    # First check all 6-subsets (most common violations)
    for subset in itertools.combinations(range(n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_regular_subgraph(adj_bits, mask):
            return False
    # Check larger subsets
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subgraph(adj_bits, mask):
                return False
    return True

def _generate_random_graph(n):
    """Generate a random graph on n vertices."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                edges.append([i, j])
    return edges

def _local_search(n, max_iter=50000, time_limit=100):
    """Local search to find a graph with no regular induced subgraph of size 6+."""
    start_time = time.time()
    
    best_n = 0
    best_edges = []
    
    for target_n in range(21, 19, -1):  # Try 21, 20
        adj_bits = [0] * target_n
        edges = []
        
        # Initialize with random graph
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    edges.append([i, j])
        
        # Local search: flip edges to eliminate regular 6-subgraphs
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Find a violating 6-subset
            found_violation = False
            for subset in itertools.combinations(range(target_n), 6):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if _is_regular_subgraph(adj_bits, mask):
                    found_violation = True
                    # Try to fix by flipping a random edge within or near the subset
                    # Pick a random edge to flip
                    u, v = random.sample(range(target_n), 2)
                    if adj_bits[u] & (1 << v):
                        # Remove edge
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        edges = [[a, b] for a, b in edges if not (a == u and b == v) and not (a == v and b == u)]
                    else:
                        # Add edge
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        edges.append([u, v])
                    break
            
            if not found_violation:
                # Check if larger subsets are also OK
                ok = True
                for size in range(7, target_n + 1):
                    for subset in itertools.combinations(range(target_n), size):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        if _is_regular_subgraph(adj_bits, mask):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_n = target_n
                    best_edges = edges
                    return best_n, best_edges
                # If only 6-subsets were checked and passed but larger ones fail,
                # we need to fix those too
                # For simplicity, continue searching
        
        # If we didn't find a valid graph for this target_n, continue to next
    
    return best_n, best_edges

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy 1: Try local search for n=21, then n=20
    random.seed(42)
    
    best_n = 0
    best_edges = []
    
    for target_n in [21, 20]:
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            n_found, edges_found = _local_search(target_n, max_iter=20000, time_limit=deadline - time.time())
            if n_found >= target_n:
                best_n = n_found
                best_edges = edges_found
                # Verify before returning
                if _verify_graph(best_n, best_edges):
                    return {'n': best_n, 'edges': best_edges}
        
        if best_n >= target_n and _verify_graph(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: use a known 20-vertex construction or return best found
    if best_n > 0 and _verify_graph(best_n, best_edges):
        return {'n': best_n, 'edges': best_edges}
    
    # Final fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
