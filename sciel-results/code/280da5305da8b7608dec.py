"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # 80s safety margin
    
    n = 14
    m = 35
    
    # Use powers of 2: values[i] = 2^i
    # Sums: 2^i + 2^j, all distinct for different (i,j) pairs -> m distinct sums
    # Products: 2^(i+j), collisions when i+j == k+l
    # Choose edges to minimize distinct (i+j) values
    values = [2**i for i in range(n)]
    
    # For each possible sum s = i+j, count how many valid edges (i<j, i+j=s)
    # We want to pick m edges using the fewest distinct s values
    # Group all possible edges by their sum s
    edge_by_sum = {}
    for i in range(n):
        for j in range(i+1, n):
            s = i + j
            if s not in edge_by_sum:
                edge_by_sum[s] = []
            edge_by_sum[s].append((i, j))
    
    # Sort sum groups by size (descending) to pick most edges per sum value
    sum_groups = sorted(edge_by_sum.items(), key=lambda x: -len(x[1]))
    
    edges = []
    for s, group in sum_groups:
        if len(edges) >= m:
            break
        for (i, j) in group:
            if len(edges) >= m:
                break
            edges.append([i, j])
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Ensure exactly m edges
    if len(edges) < m:
        # Fill with remaining edges
        all_edges = list(combinations(range(n), 2))
        existing = set(tuple(e) for e in edges)
        for e in all_edges:
            if len(edges) >= m:
                break
            if e not in existing:
                edges.append([e[0], e[1]])
    
    # Validate
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
