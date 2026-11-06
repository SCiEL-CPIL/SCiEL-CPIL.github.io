"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_max_book(adj_set, n=40):
    """Compute the maximum book size (max triangles containing any edge)."""
    neighbors = [set() for _ in range(n)]
    for u, v in adj_set:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    max_book = 0
    for (u, v) in adj_set:
        # Count common neighbors of u and v
        common_count = len(neighbors[u] & neighbors[v])
        if common_count > max_book:
            max_book = common_count
    return max_book

def _greedy_construct(n=40, target_edges=401, deadline=None):
    """Greedy construction: add edges one by one, choosing the edge that minimizes max book."""
    if deadline is None:
        deadline = time.time() + 110.0
    
    edges_set = set()
    neighbors = [set() for _ in range(n)]
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Greedily add edges
    while len(edges_set) < target_edges and time.time() < deadline:
        best_edge = None
        best_max = float('inf')
        
        # Sample a subset of candidate edges to keep it fast
        if len(all_edges) > 500:
            candidates = random.sample(all_edges, 500)
        else:
            candidates = all_edges[:]
        
        for (u, v) in candidates:
            if (u, v) in edges_set:
                continue
            
            # Calculate the book size if we add this edge
            # The book for edge (u,v) = number of common neighbors of u and v
            common = len(neighbors[u] & neighbors[v])
            
            # But adding (u,v) also affects the book sizes of edges (u,w) and (v,w)
            # for all w that is neighbor of both u and v
            # Actually, adding (u,v) creates triangles (u,v,w) for each common neighbor w
            # This increases the book of edge (u,w) by 1 and edge (v,w) by 1
            # We need to check if any of these would exceed current max
            
            # For simplicity, just use the direct book of (u,v) as a heuristic
            if common < best_max:
                best_max = common
                best_edge = (u, v)
        
        if best_edge is None:
            # Fallback: pick any edge not in set
            for e in all_edges:
                if e not in edges_set:
                    best_edge = e
                    break
        
        if best_edge is not None:
            u, v = best_edge
            edges_set.add(best_edge)
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        # Remove added edge from candidates
        if best_edge in all_edges:
            all_edges.remove(best_edge)
    
    return edges_set

def _local_search(edges_set, n=40, deadline=None):
    """Local search to improve the graph."""
    if deadline is None:
        deadline = time.time() + 110.0
    
    best_edges_set = set(edges_set)
    best_max = _compute_max_book(best_edges_set, n)
    current_max = best_max
    
    random.seed(42)
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 1000:
            # Restart with random perturbation
            for _ in range(10):
                if edges_set:
                    e = random.choice(list(edges_set))
                    edges_set.remove(e)
                    # Add a random new edge
                    while True:
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b:
                            continue
                        ne = (min(a,b), max(a,b))
                        if ne not in edges_set:
                            edges_set.add(ne)
                            break
            no_improve_count = 0
            continue
        
        # Pick a random edge to remove
        if not edges_set:
            break
        edge_to_remove = random.choice(list(edges_set))
        
        # Try several random swaps
        found_improvement = False
        for _ in range(10):
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
            edges_set.remove(edge_to_remove)
            edges_set.add(new_edge)
            
            new_max = _compute_max_book(edges_set, n)
            
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges_set = set(edges_set)
                found_improvement = True
                no_improve_count = 0
            else:
                # Revert
                edges_set.remove(new_edge)
                edges_set.add(edge_to_remove)
            
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve_count += 1
    
    return best_edges_set

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 105.0  # Safety margin
    
    random.seed(42)
    
    # Phase 1: Greedy construction
    edges_set = _greedy_construct(n, target_edges, deadline)
    
    # Phase 2: Local search to improve
    best_edges_set = _local_search(edges_set, n, deadline)
    
    # Convert to list of lists as per contract hint [u, v]
    return [list(e) for e in best_edges_set]
# EVOLVE-BLOCK-END
