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
    deadline = time.time() + 145  # 145s out of 150s budget
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies with bitset-based C4 check
        for attempt in range(30):
            if time.time() > deadline:
                break
            result = _try_construct_greedy(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # Local search with edge swaps
        result = _local_search(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
    
    # Fallback: return a valid smaller graph
    return {'n': 39, 'edges': []}


def _check_c4_bitset(nbrs, u, v):
    """
    Check if adding edge (u,v) creates a C4 using bitsets.
    nbrs is a list of integers where nbrs[i] is a bitmask of neighbors of i.
    """
    # A C4 is created iff there exists x != u,v such that:
    # x is adjacent to u, and x has a common neighbor y with v (y != u, y != v)
    # i.e., (N(x) & N(v)) has a bit set other than u and v, for some x in N(u)
    
    nu = nbrs[u]
    nv = nbrs[v]
    
    # For each neighbor x of u (excluding v), check if N(x) & N(v) has bits other than u,v
    # We can compute: for all x in N(u), does (N(x) & N(v)) & ~(1<<u | 1<<v | 1<<x) != 0?
    
    # Iterate over neighbors of u
    x = 0
    while nu:
        if nu & 1:
            if x != v:
                # Check if N(x) & N(v) has any bit other than u, v, x
                common = nbrs[x] & nv
                # Mask out u, v, x
                mask = ~( (1 << u) | (1 << v) | (1 << x) )
                if common & mask:
                    return True
        nu >>= 1
        x += 1
    
    return False


def _try_construct_greedy(n, required_min_deg, deadline, seed):
    """
    Greedy construction with bitset-based C4 checking.
    """
    random.seed(seed)
    
    # nbrs[i] is a bitmask of neighbors of vertex i
    nbrs = [0] * n
    
    # Precompute all edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # First pass: add all edges that don't create C4
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not (nbrs[u] & (1 << v)) and not _check_c4_bitset(nbrs, u, v):
            nbrs[u] |= (1 << v)
            nbrs[v] |= (1 << u)
    
    # Compute degrees
    degrees = [bin(nbrs[i]).count('1') for i in range(n)]
    
    # Second pass: try to increase min degree by adding more edges
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        min_deg = min(degrees)
        if min_deg >= required_min_deg:
            break
        
        # Find vertices with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        random.shuffle(min_vertices)
        
        added = False
        for u in min_vertices:
            if time.time() > deadline:
                break
            
            # Try adding edges from u to other vertices
            candidates = [v for v in range(n) if v != u and not (nbrs[u] & (1 << v))]
            # Sort by degree (prefer lower degree vertices)
            candidates.sort(key=lambda v: degrees[v])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_bitset(nbrs, u, v):
                    nbrs[u] |= (1 << v)
                    nbrs[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            break
    
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if nbrs[u] & (1 << v):
                    edges.append([u, v])
        return (n, edges)
    return None


def _local_search(n, required_min_deg, deadline):
    """
    Local search: start with a greedy solution, then try edge swaps to improve min degree.
    """
    # Get an initial solution
    best_nbrs = None
    best_min_deg = -1
    
    for seed in range(15):
        if time.time() > deadline:
            break
        result = _try_construct_greedy(n, required_min_deg, deadline, seed)
        if result is not None:
            n_found, edges_found = result
            # Reconstruct nbrs
            nbrs = [0] * n
            for u, v in edges_found:
                nbrs[u] |= (1 << v)
                nbrs[v] |= (1 << u)
            degrees = [bin(nbrs[i]).count('1') for i in range(n)]
            min_deg = min(degrees)
            if min_deg > best_min_deg:
                best_min_deg = min_deg
                best_nbrs = nbrs[:]
            if min_deg >= required_min_deg:
                return (n, edges_found)
    
    if best_nbrs is None:
        return None
    
    nbrs = best_nbrs[:]
    degrees = [bin(nbrs[i]).count('1') for i in range(n)]
    
    # Local search: try edge swaps
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        min_deg = min(degrees)
        if min_deg >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if nbrs[u] & (1 << v):
                        edges.append([u, v])
            return (n, edges)
        
        # Find a vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        u = random.choice(min_vertices)
        
        # Try to add an edge from u to a vertex with low degree
        candidates = [v for v in range(n) if v != u and not (nbrs[u] & (1 << v))]
        candidates.sort(key=lambda v: degrees[v])
        
        found_swap = False
        for v in candidates[:15]:  # Try top 15 candidates
            if time.time() > deadline:
                break
            if not _check_c4_bitset(nbrs, u, v):
                nbrs[u] |= (1 << v)
                nbrs[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                found_swap = True
                break
        
        if not found_swap:
            # Try removing an edge from a high-degree vertex and adding it elsewhere
            high_vertices = [i for i in range(n) if degrees[i] == max(degrees)]
            w = random.choice(high_vertices)
            # Find a neighbor of w to remove
            neighbors = []
            for x in range(n):
                if nbrs[w] & (1 << x):
                    neighbors.append(x)
            if neighbors:
                z = random.choice(neighbors)
                # Try to add edge (w, some other vertex) instead
                other_candidates = [x for x in range(n) if x != w and x != z and not (nbrs[w] & (1 << x))]
                other_candidates.sort(key=lambda x: degrees[x])
                for x in other_candidates[:8]:
                    if time.time() > deadline:
                        break
                    if not _check_c4_bitset(nbrs, w, x):
                        # Remove (w,z), add (w,x)
                        nbrs[w] &= ~(1 << z)
                        nbrs[z] &= ~(1 << w)
                        nbrs[w] |= (1 << x)
                        nbrs[x] |= (1 << w)
                        degrees[z] -= 1
                        degrees[x] += 1
                        found_swap = True
                        break
    
    # Check final result
    degrees = [bin(nbrs[i]).count('1') for i in range(n)]
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if nbrs[u] & (1 << v):
                    edges.append([u, v])
        return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
