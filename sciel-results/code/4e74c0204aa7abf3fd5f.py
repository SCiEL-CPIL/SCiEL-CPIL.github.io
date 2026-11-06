import random
import time
import sys

def _max_independent_set_size_bitmask(adj_masks, n):
    """
    Find the size of the maximum independent set using bitmasks and branch and bound.
    adj_masks: list of integers, where adj_masks[i] is the bitmask of neighbors of i.
    n: number of vertices.
    """
    best = [0]
    full_mask = (1 << n) - 1
    
    # Memoization is not ideal for large n, but pruning is key.
    # We use a recursive function that takes the available vertices mask.
    
    def backtrack(available_mask, current_size):
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pruning: even if we take all remaining vertices, we can't beat best
        # Note: This is a weak bound, but helps.
        # Stronger bound: upper bound on independent set size in available_mask.
        # We can use a greedy heuristic to get a lower bound on the remaining potential?
        # Or just rely on the recursion depth.
        
        # If current_size + popcount(available_mask) <= best[0], return
        # popcount is fast in Python 3.8+ via int.bit_count()
        if current_size + available_mask.bit_count() <= best[0]:
            return
            
        # Pick a vertex with high degree in the induced subgraph?
        # Or just pick the lowest set bit for simplicity and speed.
        # Picking a vertex with high degree in the remaining graph is better for pruning.
        
        # Find vertex with max degree in available_mask
        # This is O(n^2) per node, might be slow.
        # Let's stick to lowest bit for now, or maybe a fixed order.
        
        v = (available_mask & -available_mask).bit_length() - 1
        # v is the index of the lowest set bit
        
        # Option 1: Exclude v
        backtrack(available_mask & ~(1 << v), current_size)
        
        # Option 2: Include v
        # Remove v and its neighbors from available_mask
        neighbors = adj_masks[v]
        new_available = available_mask & ~(1 << v) & ~neighbors
        backtrack(new_available, current_size + 1)

    backtrack(full_mask, 0)
    return best[0]

def _is_triangle_free_masks(adj_masks, n):
    """Check if graph defined by adj_masks is triangle-free."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj_masks[u] & (1 << v):
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _generate_random_tf_graph(n, seed):
    """
    Generate a random triangle-free graph.
    Strategy: Start with random edges, remove those that create triangles.
    """
    rng = random.Random(seed)
    adj_masks = [0] * n
    
    # Randomly permute edges to break symmetry
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u, v) creates a triangle
        # Triangle exists if there is w such that (u,w) and (v,w) are in graph
        # i.e., adj_masks[u] & adj_masks[v] is not 0
        if adj_masks[u] & adj_masks[v] == 0:
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
    return adj_masks

def _addable_edges(adj_masks, n):
    """
    Return list of edges (u, v) that can be added without creating a triangle.
    """
    addable = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_masks[u] & (1 << v)):
                # Check if adding creates triangle
                if adj_masks[u] & adj_masks[v] == 0:
                    addable.append((u, v))
    return addable

def _try_improve(adj_masks, n, deadline, max_iterations=1000):
    """
    Try to reduce alpha by adding edges.
    Returns (success, adj_masks)
    """
    alpha = _max_independent_set_size_bitmask(adj_masks, n)
    if alpha < 10:
        return True, adj_masks
        
    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        addable = _addable_edges(adj_masks, n)
        if not addable:
            break
        
        # Pick a random addable edge
        u, v = random.choice(addable)
        
        # Add it
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
        # Check alpha
        new_alpha = _max_independent_set_size_bitmask(adj_masks, n)
        if new_alpha < 10:
            return True, adj_masks
            
        # If alpha didn't decrease or increased, we might want to backtrack?
        # For simplicity, we just keep going. If we get stuck, we restart.
        # A better strategy: if new_alpha >= alpha, undo the move and try another.
        # But checking alpha is expensive. Let's assume adding edges generally helps.
        
    return False, adj_masks

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_result = None
    best_n = 0
    
    # Try n from 39 down to 20
    # We know n=26 is possible (Andrasfai).
    # We want to maximize n.
    
    for n in range(39, 19, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline:
                break
                
            adj_masks = _generate_random_tf_graph(n, seed)
            
            # Quick check alpha
            alpha = _max_independent_set_size_bitmask(adj_masks, n)
            
            if alpha < 10:
                # Found a valid graph
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_masks[u] & (1 << v):
                            edges.append([u, v])
                best_result = {'n': n, 'edges': edges}
                best_n = n
                found = True
                break
            else:
                # Try to improve
                success, adj_masks = _try_improve(adj_masks, n, deadline, max_iterations=20)
                if success:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj_masks[u] & (1 << v):
                                edges.append([u, v])
                    best_result = {'n': n, 'edges': edges}
                    best_n = n
                    found = True
                    break
                    
        if found:
            # If we found a valid graph for this n, we can try larger n?
            # The loop goes 39 down to 20.
            # If we found one for 39, we return immediately.
            # If we found one for 38, we continue to 37? No, we want max n.
            # So if we find one for n, we should return it immediately?
            # Yes, because we are iterating downwards.
            return best_result
        else:
            # If we didn't find one for this n, we continue to smaller n.
            # But we should keep the best result found so far (which is None or smaller).
            pass
            
    # If no graph found for n > 20, try to find one for n=26 (Andrasfai) as fallback
    if best_result is None:
        # Construct Andrasfai graph And(9) is not standard, And(7) is n=21.
        # Let's just return a small valid graph if we must.
        # C5 blowup by 4 is n=20, alpha=8.
        n = 20
        edges = []
        # C5 vertices 0..4. Blowup by 4: vertices 0..19.
        # Group i: 4*i, 4*i+1, 4*i+2, 4*i+3.
        # Edges between group i and group j if i,j adjacent in C5.
        # C5 edges: (0,1), (1,2), (2,3), (3,4), (4,0).
        c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u_g, v_g in c5_edges:
            for u in range(4*u_g, 4*u_g+4):
                for v in range(4*v_g, 4*v_g+4):
                    edges.append([u, v])
        best_result = {'n': n, 'edges': edges}
        
    return best_result
