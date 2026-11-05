import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 140  # 140s out of 150s budget
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies and seeds
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If not found with fast method, try local search
        result = _local_search_fast(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
    
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _check_c4_bitset(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4 using bitset adjacency.
    adj is a list of integers where bit j of adj[i] is 1 if (i, j) is an edge.
    """
    nu = adj[u]
    nv = adj[v]
    
    x_candidates = nu
    while x_candidates:
        x = (x_candidates & -x_candidates).bit_length() - 1
        x_candidates &= x_candidates - 1
        
        # Check if there is y in N(v) such that x ~ y and y != x
        # Common neighbors of x and v: common = adj[x] & nv
        # We need common to have a bit set that is not x.
        if (adj[x] & nv) & ~(1 << x):
            return True
            
    return False


def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Targeted greedy construction: always try to increase the degree of the vertex with the minimum degree.
    """
    random.seed(seed)
    
    adj = [0] * n
    degrees = [0] * n
    
    max_steps = n * n * 10 # Safety limit
    steps = 0
    
    while min(degrees) < required_min_deg:
        if time.time() > deadline:
            return None
        steps += 1
        if steps > max_steps:
            return None
            
        # Find vertex with minimum degree
        min_deg = min(degrees)
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        random.shuffle(min_vertices)
        
        added = False
        for u in min_vertices:
            # Try to find a v to connect to u
            # Candidates: v not in N(u)
            candidates = [v for v in range(n) if v != u and not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            
            for v in candidates:
                if _check_c4_bitset(adj, u, v):
                    continue
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break
                
            if added:
                break
                
        if not added:
            # Could not add any edge to any min-degree vertex
            return None
            
    # Success
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return (n, edges)

def _local_search_fast(n, required_min_deg, deadline):
    """
    Local search: Start with a random valid C4-free graph, then try to improve min degree.
    """
    for seed in range(50):
        if time.time() > deadline:
            break
            
        random.seed(seed + 10000)
        
        # Initialize with a random subset of edges that is C4-free
        adj = [0] * n
        degrees = [0] * n
        
        # Add random edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if _check_c4_bitset(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            
        # Check if current graph satisfies min degree
        if min(degrees) >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return (n, edges)
            
    return None

# EVOLVE-BLOCK-END
