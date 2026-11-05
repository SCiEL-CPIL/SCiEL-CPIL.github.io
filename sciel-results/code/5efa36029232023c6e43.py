"""Construct a C4-free graph with no independent set of size 11, maximizing n.

Uses a block design / finite geometry approach: construct a graph where
vertices are points and adjacency is defined via a combinatorial structure
that inherently guarantees C4-freeness (any two vertices share at most one
common neighbor).

Strategy: Use a greedy randomized search to build a C4-free graph with
large n and independence number < 11. Start from the seed (10 triangles)
and try to add vertices/edges while maintaining constraints.
"""

import random
import time

def _is_c4_free(adj, n):
    """Check if adjacency list represents a C4-free graph."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) > 1:
                return False
    return True

def _can_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) preserves C4-freeness."""
    if u == v:
        return False
    if v in adj[u]:
        return False
    # For C4-freeness: for any w in adj[u], w must not be in adj[v] with
    # another common neighbor. More precisely, after adding (u,v), u and v
    # will have |adj[u] ∩ adj[v]| + 0 common neighbors... wait.
    # After adding edge (u,v): u's neighbors are adj[u] ∪ {v}, v's neighbors are adj[v] ∪ {u}.
    # We need to check that no pair of vertices has >1 common neighbor.
    # The only pairs affected are pairs involving u or v.
    # For pair (u, v): common neighbors = adj[u] ∩ adj[v]. Must be ≤ 1.
    if len(adj[u] & adj[v]) > 1:
        return False
    # For pair (u, w) where w in adj[v]: u's neighbors include v, w's neighbors include v.
    # So v is a common neighbor of u and w. Need to check that u and w don't have
    # another common neighbor.
    for w in adj[v]:
        if w == u:
            continue
        # u and w: common neighbors include v (since v~u and v~w).
        # Check if they have another common neighbor.
        common_uw = adj[u] & adj[w]
        if v in common_uw:
            common_uw.discard(v)
        if len(common_uw) > 0:
            return False
    # For pair (v, w) where w in adj[u]: symmetric
    for w in adj[u]:
        if w == v:
            continue
        common_vw = adj[v] & adj[w]
        if u in common_vw:
            common_vw.discard(u)
        if len(common_vw) > 0:
            return False
    return True

def _has_independent_set(adj, n, k):
    """Check if graph has an independent set of size k using backtracking."""
    # Use a more efficient approach with bitmasks for small n
    if n > 45:
        return True  # Assume yes for very large n (conservative)
    
    # Build complement adjacency: non_adj[u] = set of vertices NOT adjacent to u (and != u)
    non_adj = [set(range(n)) - {u} - adj[u] for u in range(n)]
    
    best = [0]
    
    def backtrack(candidates, size):
        if size >= k:
            return True
        if not candidates:
            return False
        # Pruning
        if size + len(candidates) < k:
            return False
        # Greedy: pick a vertex from candidates
        # Use a heuristic: pick vertex with smallest degree in candidate subgraph
        # For simplicity, pick the first
        v = next(iter(candidates))
        # Include v
        new_candidates = candidates & non_adj[v]
        new_candidates.discard(v)
        if backtrack(new_candidates, size + 1):
            return True
        # Exclude v
        new_candidates2 = candidates - {v}
        if backtrack(new_candidates2, size):
            return True
        return False
    
    return backtrack(set(range(n)), 0)

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    
    def seed_construction():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Try to extend the seed construction by adding vertices and edges
    # Use randomized search with time limit
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    best_result = seed_construction()
    
    # Try different n values from high to low
    for target_n in range(39, 30, -1):
        if time.time() - start_time > time_limit:
            break
        
        success = False
        for trial in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with 10 triangles (30 vertices)
            adj = [set() for _ in range(target_n)]
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                adj[a].add(b); adj[b].add(a)
                adj[a].add(c); adj[c].add(a)
                adj[b].add(c); adj[c].add(b)
            
            # Add edges between triangles and to new vertices
            # Randomly try to add edges
            random.seed(trial * 1000 + target_n)
            
            # Generate all possible edges not yet present
            possible_edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if v not in adj[u]:
                        possible_edges.append((u, v))
            
            random.shuffle(possible_edges)
            
            # Greedily add edges
            for u, v in possible_edges:
                if time.time() - start_time > time_limit:
                    break
                if _can_add_edge(adj, target_n, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check C4-free
            if not _is_c4_free(adj, target_n):
                continue
            
            # Check independence number
            if _has_independent_set(adj, target_n, 11):
                continue
            
            # Valid!
            edges = []
            for u in range(target_n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            best_result = {'n': target_n, 'edges': edges}
            success = True
            break
        
        if success:
            break
    
    if time.time() - start_time > time_limit:
        return best_result
    
    return best_result
