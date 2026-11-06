"""Improved program for dense K3,3-free graphs on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k33_violations(adj):
    """Count triples with >=3 common neighbors (K3,3 violations)."""
    n = adj.shape[0]
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            ab_common = adj[a] & adj[b]
            for c in range(b+1, n):
                if ab_common[c]:
                    continue
                common = ab_common & adj[c]
                common[a] = False
                common[b] = False
                common[c] = False
                s = np.sum(common)
                if s >= 3:
                    count += 1
    return count

def _has_k33(adj):
    """Check if graph contains K3,3."""
    return _count_k33_violations(adj) > 0

def _try_add_edge(adj, u, v):
    """Try adding edge (u,v); return True if still K3,3-free."""
    if adj[u, v]:
        return True
    adj[u, v] = True
    adj[v, u] = True
    if _has_k33(adj):
        adj[u, v] = False
        adj[v, u] = False
        return False
    return True

def _greedy_fill(adj, deadline):
    """Greedy: try adding all missing edges in random order."""
    n = adj.shape[0]
    missing = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u, v]:
                missing.append((u, v))
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline:
            break
        _try_add_edge(adj, u, v)

def _local_search(adj, deadline, max_iter=50):
    """Local search: remove edges from violating triples, then re-add."""
    n = adj.shape[0]
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        violations = _count_k33_violations(adj)
        if violations == 0:
            break
        # Find a violating triple and remove one edge
        found = False
        for a in range(n):
            if time.time() > deadline:
                return adj
            for b in range(a+1, n):
                ab_common = adj[a] & adj[b]
                for c in range(b+1, n):
                    if ab_common[c]:
                        continue
                    common = ab_common & adj[c]
                    common[a] = False
                    common[b] = False
                    common[c] = False
                    if np.sum(common) >= 3:
                        # Remove a random edge incident to one of a,b,c
                        candidates = []
                        for x in [a, b, c]:
                            for y in range(n):
                                if y != x and adj[x, y]:
                                    candidates.append((min(x,y), max(x,y)))
                        if candidates:
                            u, v = random.choice(candidates)
                            adj[u, v] = False
                            adj[v, u] = False
                            found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        # After removal, try to re-add edges greedily
        _greedy_fill(adj, deadline)
    return adj

def _build_edges_from_adj(adj):
    """Convert adjacency matrix to edge list."""
    n = adj.shape[0]
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin for 150s limit
    
    n = 20
    best_adj = np.zeros((n, n), dtype=bool)
    best_edges = 0
    
    # Seed with K2,18
    for a in range(2):
        for b in range(2, n):
            best_adj[a, b] = True
            best_adj[b, a] = True
    
    # Multiple random restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Create a copy with some random modifications
        adj = best_adj.copy()
        
        # Randomly remove a few edges to explore
        if restart > 0 and random.random() < 0.5:
            current_edges = _build_edges_from_adj(adj)
            for _ in range(random.randint(1, 3)):
                if current_edges:
                    u, v = random.choice(current_edges)
                    adj[u, v] = False
                    adj[v, u] = False
        
        # Greedy fill
        _greedy_fill(adj, deadline)
        
        # Local search
        adj = _local_search(adj, deadline, max_iter=10)
        
        # Check if better
        edge_count = np.sum(adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
    
    return _build_edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
