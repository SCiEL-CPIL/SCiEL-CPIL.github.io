"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _build_adj(edges, n=40):
    """Build adjacency list from edge set."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_max_book(adj, n=40):
    """Compute the maximum book size (max triangles containing any edge)."""
    max_book = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common_count = len(adj[u] & adj[v])
                if common_count > max_book:
                    max_book = common_count
    return max_book

def _affected_edges(edge, adj, n=40):
    """Return set of edges whose book size may change when `edge` is added/removed."""
    u, v = edge
    affected = set()
    # Edges incident to u or v
    for w in adj[u]:
        if w != v:
            affected.add((min(u, w), max(u, w)))
    for w in adj[v]:
        if w != u:
            affected.add((min(v, w), max(v, w)))
    # The edge itself
    affected.add((min(u, v), max(u, v)))
    return affected

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0  # Safety margin
    
    random.seed(42)
    
    # Initial: K20,20 + one intra-partition edge
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    edges_set.add((0, 1))
    
    adj = _build_adj(edges_set, n)
    current_max = _compute_max_book(adj, n)
    best_edges_set = set(edges_set)
    best_max = current_max
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges_set:
            break
        edge_to_remove = random.choice(list(edges_set))
        u_rm, v_rm = edge_to_remove
        
        # Generate candidate new edges - focus on edges that might reduce book
        found_improvement = False
        candidates_tried = 0
        max_candidates = 20
        
        while candidates_tried < max_candidates and not found_improvement:
            candidates_tried += 1
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            new_edge = (min(a, b), max(a, b))
            
            if new_edge == edge_to_remove:
                continue
            if new_edge in edges_set:
                continue
            
            # Tentative swap
            edges_set.discard(edge_to_remove)
            edges_set.add(new_edge)
            adj[u_rm].discard(v_rm)
            adj[v_rm].discard(u_rm)
            na, nb = new_edge
            adj[na].add(nb)
            adj[nb].add(na)
            
            # Only check affected edges for max book
            affected = _affected_edges(new_edge, adj, n) | _affected_edges(edge_to_remove, adj, n)
            new_max = 0
            for e in affected:
                eu, ev = e
                if eu < ev and ev in adj[eu]:
                    common_count = len(adj[eu] & adj[ev])
                    if common_count > new_max:
                        new_max = common_count
            
            if new_max < best_max:
                best_max = new_max
                best_edges_set = set(edges_set)
                found_improvement = True
            else:
                # Revert
                edges_set.discard(new_edge)
                edges_set.add(edge_to_remove)
                adj[na].discard(nb)
                adj[nb].discard(na)
                adj[u_rm].add(v_rm)
                adj[v_rm].add(u_rm)
        
        if found_improvement:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck for a while, try a restart with perturbation
        if no_improve_count >= max_no_improve:
            no_improve_count = 0
            # Perturb: do several random swaps
            for _ in range(5):
                if not edges_set:
                    break
                e_rm = random.choice(list(edges_set))
                for _ in range(10):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b:
                        continue
                    ne = (min(a, b), max(a, b))
                    if ne != e_rm and ne not in edges_set:
                        edges_set.discard(e_rm)
                        edges_set.add(ne)
                        break
            
            adj = _build_adj(edges_set, n)
            current_max = _compute_max_book(adj, n)
            if current_max < best_max:
                best_max = current_max
                best_edges_set = set(edges_set)
    
    # Convert to list of lists as per contract hint [u, v]
    return [list(e) for e in best_edges_set]
# EVOLVE-BLOCK-END
