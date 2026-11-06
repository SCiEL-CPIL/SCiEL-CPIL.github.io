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

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110.0  # Safety margin
    
    # Initial: K20,20 + one edge
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    edges_set.add((0, 1))
    
    current_max = _compute_max_book(edges_set, n)
    best_edges_set = set(edges_set)
    best_max = current_max
    
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges_set:
            break
        edge_to_remove = random.choice(list(edges_set))
        
        # Generate candidate new edges
        found_improvement = False
        # Try a few random swaps
        for _ in range(5):
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
            
            if new_max < best_max:
                best_max = new_max
                best_edges_set = set(edges_set)
                found_improvement = True
            else:
                # Revert
                edges_set.remove(new_edge)
                edges_set.add(edge_to_remove)
            
            if found_improvement:
                break
        
        # If no improvement in a while, we could restart, but simple hill climbing works okay.
    
    # Convert to list of lists as per contract hint [u, v]
    return [list(e) for e in best_edges_set]
# EVOLVE-BLOCK-END
