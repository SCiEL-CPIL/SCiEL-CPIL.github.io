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
    deadline = time.time() + 148  # 148s out of 150s budget
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with bitset-based C4 check
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            result = _local_search_bitset(n, required_min_deg, deadline, seed=attempt)
            if result is not None:
                n_found, edges_found = result
                if n_found > best_n:
                    best_n = n_found
                    best_edges = edges_found
                return {'n': n_found, 'edges': edges_found}
        
        # Try greedy construction with bitsets
        for attempt in range(10):
            if time.time() > deadline:
                break
            result = _greedy_bitset(n, required_min_deg, deadline, seed=attempt)
            if result is not None:
                n_found, edges_found = result
                if n_found > best_n:
                    best_n = n_found
                    best_edges = edges_found
                return {'n': n_found, 'edges': edges_found}
    
    return {'n': best_n, 'edges': best_edges}


def _c4_check_bitset(adj_bits, n, u, v):
    """
    Check if adding edge (u,v) creates a C4 using bitsets.
    adj_bits[i] is a Python int representing the adjacency of vertex i.
    A C4 is created iff there exist x, y such that:
      u~x, x~y, y~v, x != v, y != u, x != y
    Equivalently: exists x in N(u)\\{v} such that N(x) ∩ (N(v)\\{u}) != empty
    where the intersection excludes x itself (but x can't be in N(v)\\{u} if x~v and x~u... 
    actually x could be in both N(u) and N(v)).
    
    More carefully: we need x in N(u), x != v, and y in N(v), y != u, y != x, x~y.
    So: for x in N(u)\\{v}: check if (adj_bits[x] & adj_bits[v]) has any bit set except possibly x and u.
    Actually: (adj_bits[x] & adj_bits[v]) & ~(1 << x) & ~(1 << u) != 0
    But also need y != x which is handled by clearing bit x.
    And y != u is handled by clearing bit u.
    Also need y != v? No, y is in N(v) so y != v automatically (no self-loops).
    And x != v is handled by x in N(u)\\{v}.
    
    Simplified: exists x in N(u)\\{v} such that (adj_bits[x] & adj_bits[v] & ~(1<<x) & ~(1<<u)) != 0
    """
    nu = adj_bits[u] & ~(1 << v)  # N(u) \\ {v}
    if nu == 0:
        return False
    mask = ~(1 << u)  # exclude u from y candidates
    # For each x in nu, check if adj_bits[x] & adj_bits[v] & mask has any bit != x
    # We can iterate through set bits of nu
    temp = nu
    while temp:
        # Get lowest set bit
        x_bit = temp & (-temp)
        x = x_bit.bit_length() - 1
        temp ^= x_bit
        # Check if adj_bits[x] & adj_bits[v] has any bit set other than x and u
        common = adj_bits[x] & adj_bits[v] & mask
        if common & ~(x_bit):  # any common neighbor that's not x
            return True
    return False


def _greedy_bitset(n, required_min_deg, deadline, seed=0):
    """Greedy edge addition with bitset-based C4 check."""
    rng = random.Random(seed)
    adj_bits = [0] * n  # adj_bits[i] has bit j set if i~j
    
    # Build list of all edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_pairs)
    
    # Greedy: try to add edges, prioritizing low-degree vertices
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Compute degrees
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        
        if min(degrees) >= required_min_deg:
            break
        
        # Find lowest degree vertex
        min_deg = min(degrees)
        low_verts = [i for i in range(n) if degrees[i] == min_deg]
        rng.shuffle(low_verts)
        
        improved = False
        for u in low_verts:
            if time.time() > deadline:
                break
            
            # Try to add edges from u to non-neighbors, sorted by their degree
            non_neighbors = [v for v in range(n) if v != u and not (adj_bits[u] >> v) & 1]
            # Sort by degree (prefer connecting to low-degree vertices)
            non_neighbors.sort(key=lambda x: degrees[x])
            
            for v in non_neighbors:
                if v <= u:
                    continue
                if not _c4_check_bitset(adj_bits, n, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return (n, edges)
    return None


def _local_search_bitset(n, required_min_deg, deadline, seed=0):
    """
    Local search with bitset C4 check: start with random seed, then iteratively
    add edges to boost minimum degree, and try edge swaps if stuck.
    """
    rng = random.Random(seed)
    adj_bits = [0] * n
    
    # Initialize with random sparse graph
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_pairs)
    
    # Add random edges until C4-free constraint is tight
    for (u, v) in all_pairs:
        if time.time() > deadline:
            break
        if not _c4_check_bitset(adj_bits, n, u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
    
    # Local search phase
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        if min(degrees) >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return (n, edges)
        
        # Find lowest degree vertices
        min_deg = min(degrees)
        low_verts = [i for i in range(n) if degrees[i] <= min_deg + 1]
        rng.shuffle(low_verts)
        
        improved = False
        for u in low_verts:
            if time.time() > deadline:
                break
            
            # Try to add edges from u
            non_neighbors = [v for v in range(n) if v != u and not (adj_bits[u] >> v) & 1]
            non_neighbors.sort(key=lambda x: degrees[x])
            
            for v in non_neighbors:
                if v <= u:
                    continue
                if not _c4_check_bitset(adj_bits, n, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try edge swaps: remove an edge incident to a high-degree vertex,
            # then try to add an edge incident to a low-degree vertex
            high_verts = [i for i in range(n) if degrees[i] >= max(degrees) - 1]
            rng.shuffle(high_verts)
            
            swap_improved = False
            for h in high_verts:
                if time.time() > deadline:
                    break
                
                neighbors = [v for v in range(n) if v != h and (adj_bits[h] >> v) & 1]
                rng.shuffle(neighbors)
                
                for v in neighbors:
                    if v <= h:
                        continue
                    # Try removing edge (h, v)
                    adj_bits[h] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << h)
                    
                    # Now try to add edges to boost low-degree vertices
                    new_degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
                    new_min_deg = min(new_degrees)
                    new_low_verts = [i for i in range(n) if new_degrees[i] <= new_min_deg + 1]
                    rng.shuffle(new_low_verts)
                    
                    added = False
                    for lu in new_low_verts:
                        if time.time() > deadline:
                            break
                        non_neighbors = [x for x in range(n) if x != lu and not (adj_bits[lu] >> x) & 1]
                        non_neighbors.sort(key=lambda x: new_degrees[x])
                        
                        for lv in non_neighbors:
                            if lv <= lu:
                                continue
                            if not _c4_check_bitset(adj_bits, n, lu, lv):
                                adj_bits[lu] |= (1 << lv)
                                adj_bits[lv] |= (1 << lu)
                                added = True
                                break
                        if added:
                            break
                    
                    if added:
                        swap_improved = True
                        break
                    else:
                        # Restore the removed edge
                        adj_bits[h] |= (1 << v)
                        adj_bits[v] |= (1 << h)
                
                if swap_improved:
                    break
            
            if not swap_improved:
                break
    
    # Final check
    degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return (n, edges)
    return None

# EVOLVE-BLOCK-END
