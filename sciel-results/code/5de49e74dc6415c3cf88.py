"""Construct a C4-free graph with high minimum degree using structured seeding + fast 2-opt swaps."""
import time
import random
from itertools import combinations

def _is_c4_free(n, adj):
    """Check C4-free using adjacency bitsets."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
    return True

def _min_degree(adj, n):
    return min(adj[i].bit_count() for i in range(n))

def _forbidden_pairs(n, adj):
    """Build forbidden matrix: forbid[u][v] bitset of w such that adding (u,v) creates C4 with w in common neighborhood.
    Actually: adding edge (u,v) creates C4 iff there exist x,y with x~u, y~v, x~y, x!=v, y!=u.
    Equivalent: common_neighbors(u,v) >= 2 is NOT the right check for adding an edge.
    Correct: adding (u,v) creates C4 iff there exist x in N(u)\\{v} and y in N(v)\\{u} with x~y and x!=y.
    But simpler: adding (u,v) creates C4 iff |N(u) ∩ N(v)| >= 2 AND (u,v) not already edge... 
    Wait no. C4 is u-x-v-y-u. If we add (u,v), a C4 exists if there are x,y such that u-x-v-y-u forms cycle.
    That means x~u, x~v, y~u, y~v, x~y, x!=y. So we need two common neighbors x,y of u and v that are adjacent to each other.
    
    For the swap: remove (a,b), add (c,d). The new edge (c,d) must not create C4 in the graph after removing (a,b).
    This is complex to precompute. Instead, use fast direct check.
    """
    pass

def _creates_c4_fast(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Fast bitset version.
    C4 u-x-v-y-u requires x in N(u), x in N(v), y in N(u), y in N(v), x~y, x!=y.
    So we need two distinct common neighbors of u and v that are adjacent.
    """
    common = adj[u] & adj[v]
    # Need to check if any two vertices in common are adjacent
    c = common
    while c:
        bit = c & (-c)
        w = bit.bit_length() - 1
        c ^= bit
        # Check if w is adjacent to any other vertex in common
        other = common & ~bit
        if other & adj[w]:
            return True
    return False

def _creates_c4_after_removal(adj, n, u, v, a, b):
    """Check if adding (u,v) creates C4 after removing edge (a,b).
    We simulate: temporarily remove (a,b), check C4 for (u,v), then restore.
    For speed, we handle the case where a or b is a common neighbor of u,v.
    """
    # If neither a nor b is a neighbor of both u and v, the removal doesn't affect common neighbors
    # Actually removal of (a,b) affects adj[a] and adj[b]
    # Let's do it properly but fast:
    # Temporarily clear bits
    adj[a] &= ~(1 << b)
    adj[b] &= ~(1 << a)
    result = _creates_c4_fast(adj, n, u, v)
    # Restore
    adj[a] |= (1 << b)
    adj[b] |= (1 << a)
    return result

def _paley_seed(n):
    """Create a Paley-type seed graph for C4-free dense core.
    For n = p^k where p ≡ 1 mod 4, use quadratic residues.
    For general n, use a heuristic: connect i,j if (j-i)^2 is a quadratic residue mod n (approximate).
    Since we need C4-free, we'll start sparse and add edges.
    Actually, let's just start with a random sparse seed and build up.
    """
    adj = [0] * n
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found_valid = False
        for seed in range(300):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            # Phase 1: Greedy edge addition with fast C4 check
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            all_edges = list(combinations(range(target_n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if not _creates_c4_fast(adj, target_n, u, v):
                    edges.append([u, v])
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            # Check if we already meet the minimum degree requirement
            if min(deg) >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges[:]
                    found_valid = True
                    break
            
            # Phase 2: 2-opt edge swaps to improve minimum degree
            if time.time() > deadline:
                break
            
            edges_set = set()
            for u, v in edges:
                edges_set.add((min(u, v), max(u, v)))
            
            improved = True
            max_swaps = 5000
            swap_count = 0
            
            while improved and time.time() < deadline and swap_count < max_swaps:
                improved = False
                swap_count += 1
                
                min_deg = min(deg)
                if min_deg >= min_deg_needed:
                    break
                
                # Find vertices with minimum degree
                low_deg_vertices = [i for i in range(target_n) if deg[i] == min_deg]
                random.shuffle(low_deg_vertices)
                
                for vertex in low_deg_vertices:
                    if time.time() > deadline:
                        break
                    
                    # Try to find a non-neighbor to connect to, possibly via a swap
                    non_neighbors = [v for v in range(target_n) 
                                   if v != vertex and not (adj[vertex] >> v) & 1]
                    random.shuffle(non_neighbors)
                    
                    for candidate in non_neighbors:
                        if time.time() > deadline:
                            break
                        
                        # First try direct addition
                        if not _creates_c4_fast(adj, target_n, vertex, candidate):
                            edges.append([vertex, candidate])
                            edges_set.add((min(vertex, candidate), max(vertex, candidate)))
                            adj[vertex] |= 1 << candidate
                            adj[candidate] |= 1 << vertex
                            deg[vertex] += 1
                            deg[candidate] += 1
                            improved = True
                            break
                        
                        # Try 2-opt: remove an edge (a,b) where a is neighbor of vertex or candidate
                        # to free up C4 constraints, then add (vertex, candidate)
                        # This is expensive; only try a few candidates
                        if candidate == non_neighbors[0]:  # Only try swap for first candidate to save time
                            # Try removing edges incident to vertex's neighbors
                            neighbors_v = [x for x in range(target_n) if (adj[vertex] >> x) & 1]
                            random.shuffle(neighbors_v)
                            for a in neighbors_v[:5]:  # Limit attempts
                                if time.time() > deadline:
                                    break
                                for b in range(target_n):
                                    if b == a or b == vertex or b == candidate:
                                        continue
                                    if not (adj[a] >> b) & 1:
                                        continue
                                    # Check if removing (a,b) and adding (vertex,candidate) is valid
                                    # and improves min degree
                                    if deg[a] <= min_deg or deg[b] <= min_deg:
                                        # Removing would worsen min degree, skip
                                        continue
                                    
                                    # Simulate
                                    adj[a] &= ~(1 << b)
                                    adj[b] &= ~(1 << a)
                                    deg[a] -= 1
                                    deg[b] -= 1
                                    
                                    if not _creates_c4_fast(adj, target_n, vertex, candidate):
                                        # Valid swap
                                        edges.append([vertex, candidate])
                                        edges_set.add((min(vertex, candidate), max(vertex, candidate)))
                                        edges_set.discard((min(a, b), max(a, b)))
                                        adj[vertex] |= 1 << candidate
                                        adj[candidate] |= 1 << vertex
                                        deg[vertex] += 1
                                        deg[candidate] += 1
                                        improved = True
                                    else:
                                        # Restore
                                        adj[a] |= (1 << b)
                                        adj[b] |= (1 << a)
                                        deg[a] += 1
                                        deg[b] += 1
                                        continue
                                    break
                                if improved:
                                    break
                            if improved:
                                break
                    
                    if improved:
                        break
            
            # Final check
            if min(deg) >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges[:]
                    found_valid = True
                    break
        
        if found_valid:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
