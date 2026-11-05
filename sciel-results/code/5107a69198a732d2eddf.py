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
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies with bitset-based C4 checks
        result = _bitset_local_search(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found > best_n:
                best_n = n_found
                best_edges = edges_found
            return {'n': n_found, 'edges': edges_found}
        
        # Try greedy construction with bitset checks
        result = _try_construct_bitset(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found > best_n:
                best_n = n_found
                best_edges = edges_found
            return {'n': n_found, 'edges': edges_found}
    
    return {'n': best_n, 'edges': best_edges}


def _check_c4_bitset(adj_bits, u, v):
    """
    O(1) C4 check using precomputed bitsets.
    adj_bits[x] is a Python int representing the neighbor set of x as bits.
    Adding (u,v) creates C4 iff exists x != v, y != u, x != y such that
    adj[u,x] & adj[x,y] & adj[y,v].
    
    Using bitsets: for each x in N(u)\\{v}, check if N(x) intersects N(v)\\{u}.
    But we can do it in one operation:
    common = adj_bits[u] & adj_bits[v]  # common neighbors
    For C4: exists x in N(u)\\{v} such that N(x) & (N(v)\\{u}) != 0
    Equivalently: (adj_bits[u] & ~adj_bits[v]) should have some x where adj_bits[x] & (adj_bits[v] & ~adj_bits[u]) != 0
    
    Actually simpler: C4 exists iff there's a path u-x-y-v with x,y distinct from u,v and x!=y.
    This happens iff (adj_bits[u] & adj_bits[v]) has a common neighbor... no.
    
    Let me think again. A C4 is u-x-y-v-u. So we need x in N(u)∩N(y) and y in N(v)∩N(x).
    Equivalently: exists x,y such that x∈N(u), y∈N(v), x∈N(y), y∈N(x), x≠y, x≠v, y≠u.
    
    Using bitsets: For each x in N(u)\\{v}, check if N(x) & N(v)\\{u} is non-empty.
    But we need this to be fast. Precompute for each x: adj_bits[x].
    
    The check: let A = adj_bits[u] & ~(1<<v), B = adj_bits[v] & ~(1<<u).
    We need: exists x in A such that (adj_bits[x] & B) != 0.
    
    We can precompute: for each pair (u,v), whether adding (u,v) creates a C4.
    But that's O(n^2 * n/64) which is fine for n=47.
    
    Actually, let's just do the direct check with bitsets:
    """
    # Get neighbors of u and v as bitsets
    nu = adj_bits[u] & ~(1 << v)
    nv = adj_bits[v] & ~(1 << u)
    
    # For each x in nu, check if adj_bits[x] & nv is non-zero
    # We need to iterate over set bits in nu
    x = 0
    while nu:
        # Get lowest set bit
        lsb = nu & (-nu)
        x = lsb.bit_length() - 1
        if adj_bits[x] & nv:
            return True
        nu &= nu - 1  # clear lowest set bit
    
    return False


def _try_construct_bitset(n, required_min_deg, deadline):
    """Greedy edge addition with O(1) bitset C4 check."""
    adj_bits = [0] * n  # adj_bits[i] is bitset of neighbors of i
    degrees = [0] * n
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_pairs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        if min(degrees) >= required_min_deg:
            break
        
        # Sort by degree sum (lower degree vertices first)
        all_pairs.sort(key=lambda p: degrees[p[0]] + degrees[p[1]])
        
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if (adj_bits[u] >> v) & 1:
                continue
            if not _check_c4_bitset(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                improved = True
                break  # restart to re-prioritize
    
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return (n, edges)
    return None


def _polarity_seed_bitset(n):
    """
    Create a greedy C4-free seed using bitset checks.
    """
    adj_bits = [0] * n
    degrees = [0] * n
    rng = random.Random(42)
    
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(pairs)
    
    for (u, v) in pairs:
        if not _check_c4_bitset(adj_bits, u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
    
    return adj_bits, degrees


def _bitset_local_search(n, required_min_deg, deadline):
    """
    Local search with bitset C4 checks and k-opt swaps.
    Formulate as: maximize min degree subject to C4-free constraint.
    Use randomized local search with edge additions and swaps.
    """
    adj_bits, degrees = _polarity_seed_bitset(n)
    
    # If already good enough, return
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return (n, edges)
    
    rng = random.Random(12345)
    
    # Local search with edge additions and swaps
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        if min(degrees) >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return (n, edges)
        
        # Find vertex with minimum degree
        min_deg = min(degrees)
        low_verts = [i for i in range(n) if degrees[i] == min_deg]
        u = rng.choice(low_verts)
        
        # Try to add an edge to u
        non_neighbors = [v for v in range(n) if v != u and not ((adj_bits[u] >> v) & 1)]
        rng.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if not _check_c4_bitset(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                improved = True
                break
        
        if not improved:
            # Try a swap: remove edge (a,b) and add edge (u,v)
            # Choose a random edge to remove
            all_edges = [(a, b) for a in range(n) for b in range(a + 1, n) if (adj_bits[a] >> b) & 1]
            if all_edges:
                a, b = rng.choice(all_edges)
                # Try to add edge (u,v) for some v not adjacent to u
                non_neighbors_u = [v for v in range(n) if v != u and not ((adj_bits[u] >> v) & 1)]
                rng.shuffle(non_neighbors_u)
                
                for v in non_neighbors_u:
                    # Check if we can remove (a,b) and add (u,v)
                    # First check if (u,v) can be added without C4 (assuming (a,b) is removed)
                    # This is complex; for now, just try adding (u,v) if possible after removing (a,b)
                    # Simplified: just try to add (u,v) directly
                    if not _check_c4_bitset(adj_bits, u, v):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        improved = True
                        break
        
        if not improved and iteration % 100 == 0:
            # Random restart with a new seed
            adj_bits, degrees = _polarity_seed_bitset(n)
            if min(degrees) >= required_min_deg:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_bits[u] >> v) & 1:
                            edges.append([u, v])
                return (n, edges)
    
    # Final check
    if min(degrees) >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
