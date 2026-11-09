"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _book_sizes_from_adj(adj):
    """
    adj: numpy bool array of shape (n, n)
    Returns array book of shape (n, n) where book[i,j] is the number of triangles containing edge (i,j).
    Efficient via matrix multiplication for small n.
    """
    n = adj.shape[0]
    # A^2[i,j] = number of common neighbors of i and j
    # If adj[i,j] is true, then book[i,j] = A^2[i,j]
    # We only care about entries where adj is True.
    A2 = adj.astype(np.int64) @ adj.astype(np.int64)
    book = np.zeros((n, n), dtype=np.int64)
    book[adj] = A2[adj]
    return book

def _max_book(book):
    return int(np.max(book))

def _get_max_edges(book, top_k=10):
    """Return list of edges (i,j) with highest book sizes."""
    n = book.shape[0]
    # Create a copy to ignore diagonal
    tmp = book.copy()
    np.fill_diagonal(tmp, 0)
    # Get flattened indices
    flat = tmp.flatten()
    # Get top k indices
    # We only want upper triangle to avoid duplicates and ensure valid edges
    # But np.argpartition works on the whole array.
    # Let's just get top k from the whole array, then filter for i<j
    k = min(top_k, n*n)
    indices = np.argpartition(flat, -k)[-k:]
    edges = []
    for idx in indices:
        i, j = np.unravel_index(idx, tmp.shape)
        if i < j and book[i, j] > 0:
            edges.append((i, j))
    # Sort by book size descending
    edges.sort(key=lambda x: -book[x[0], x[1]])
    return edges

def construct():
    # Deadline guard: 120s limit, use 110s for search
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    m = 401
    
    # 1. Generate baseline: K20,20 + one edge
    # Bipartition: 0-19 and 20-39
    # Edges: all (i, j) with i in 0-19, j in 20-39, plus (0,1)
    
    # Representation: set of frozensets or list of tuples?
    # For speed in checking existence, a set of (min, max) tuples is good.
    # Or a bitset? 780 bits fits in 12 64-bit ints. But Python set is likely faster for random access.
    # Let's use a set of (u, v) with u < v.
    
    def make_edge(u, v):
        return (min(u, v), max(u, v))
        
    edges = set()
    # Add bipartite edges
    for u in range(20):
        for v in range(20, 40):
            edges.add(make_edge(u, v))
    # Add one extra edge
    edges.add((0, 1))
    
    # Ensure size is exactly m
    if len(edges) != m:
        # This shouldn't happen with the logic above (400 + 1 = 401)
        pass
        
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
        
    book = _book_sizes_from_adj(adj)
    best_max_book = _max_book(book)
    best_edges = list(edges)
    
    # Helper to update book incrementally?
    # For n=40, full recalculation is ~40^3 = 64000 ops for A2, plus overhead.
    # NumPy matmul is very fast. Let's benchmark roughly.
    # If it's < 1ms, we can do thousands of steps.
    
    # Precompute all possible non-edges?
    # Total possible edges = 40*39/2 = 780.
    # Current edges = 401.
    # Non-edges = 780 - 401 = 379.
    
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
            
    # State
    current_edges = edges
    current_adj = adj
    current_book = book
    current_max = best_max_book
    
    # Random moves
    random.seed(42)
    
    steps = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Identify "hot" edges
        # If we can reduce the max book size, great.
        # Strategy:
        # 1. Pick a random edge to remove. Bias towards edges with high book size.
        # 2. Pick a random non-edge to add.
        # 3. Check if new max book size is lower.
        
        # To be efficient, we can try to remove an edge that contributes to the max book size.
        # Let's pick a random edge from the current set.
        # Or pick from the top book size edges.
        
        if current_max <= 5:
            # We are in a good range, try to squeeze more
            pass
            
        # Select edge to remove
        # 50% chance: random edge
        # 50% chance: edge with high book size
        if random.random() < 0.5:
            remove_edge = random.choice(list(current_edges))
        else:
            top_edges = _get_max_edges(current_book, top_k=5)
            if top_edges:
                remove_edge = random.choice(top_edges)
            else:
                remove_edge = random.choice(list(current_edges))
                
        # Select edge to add
        # We need a non-edge.
        # Random sampling from non-edges:
        # We can maintain a list of non-edges? Or just sample until we find one.
        # 379 non-edges out of 780. Probability of hitting a non-edge is ~0.48.
        # So on average ~2-3 tries.
        
        add_edge = None
        for _ in range(10):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                e = make_edge(u, v)
                if e not in current_edges:
                    add_edge = e
                    break
                    
        if add_edge is None:
            # Fallback: scan all possible edges (unlikely to be needed)
            for e in all_edges:
                if e not in current_edges:
                    add_edge = e
                    break
        if add_edge is None:
            continue # Should not happen
            
        # Perform move
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        # Update adjacency
        u1, v1 = remove_edge
        current_adj[u1, v1] = False
        current_adj[v1, u1] = False
        
        u2, v2 = add_edge
        current_adj[u2, v2] = True
        current_adj[v2, u2] = True
        
        # Recalculate book sizes
        # Optimization: only recalculate if we think it might improve?
        # No, we need the exact value to decide.
        new_book = _book_sizes_from_adj(current_adj)
        new_max = _max_book(new_book)
        
        if new_max < current_max:
            # Improvement!
            current_max = new_max
            current_book = new_book
            best_max_book = current_max
            best_edges = list(current_edges)
            stuck_counter = 0
        else:
            # No improvement.
            # Should we accept it to explore?
            # If we always reject, we are stuck in local minima.
            # If we accept equal, we can wander.
            # Let's accept with a small probability if equal, to allow plateaus.
            # Or just revert.
            # Given the tight time, let's revert if not better, to stay tight.
            # But wait, if we revert, we are doing hill climbing.
            # Let's try "accept if better or equal with prob 0.1"
            if new_max == current_max and random.random() < 0.1:
                # Accept sideways move
                current_book = new_book
                # current_max stays same
                stuck_counter += 1
            else:
                # Revert
                current_edges.remove(add_edge)
                current_edges.add(remove_edge)
                
                current_adj[u2, v2] = False
                current_adj[v2, u2] = False
                
                current_adj[u1, v1] = True
                current_adj[v1, u1] = True
                
                # Book reverts to old state
                # No need to recalc, we have old_book
                stuck_counter += 1
                
        # If stuck too long, maybe restart from a different baseline?
        # Or just keep going.
        if stuck_counter > 5000:
            stuck_counter = 0
            # Could implement a restart logic here if needed
            pass

    # Return best found
    # Convert set of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
