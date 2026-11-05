import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
            
    num_k4s = len(k4s)

    def is_valid(coloring):
        for k4_idx in range(num_k4s):
            e_indices = k4_edges[k4_idx]
            color_set = set(coloring[e] for e in e_indices)
            if len(color_set) < 5:
                return False
        return True

    def count_colors(coloring):
        return len(set(coloring))

    # Try modular constructions
    best_coloring = None
    best_count = 18  # Higher than 17
    
    for p in range(13, 18):
        if time.time() > deadline - 10:
            break
            
        # Try sum
        coloring_sum = [(i + j) % p for i in range(n) for j in range(i + 1, n)]
        if is_valid(coloring_sum):
            c = count_colors(coloring_sum)
            if c < best_count:
                best_count = c
                best_coloring = coloring_sum[:]
                
        # Try diff
        coloring_diff = [(i - j) % p for i in range(n) for j in range(i + 1, n)]
        if is_valid(coloring_diff):
            c = count_colors(coloring_diff)
            if c < best_count:
                best_count = c
                best_coloring = coloring_diff[:]
                
        # Try product? (i*j) % p might be 0 often. Skip.
        
    if best_coloring is not None and best_count <= 13:
        return best_coloring

    # If we found a good modular solution (14-16), we might still try to improve with local search
    # But if we found 17, we definitely want to try to reduce it.
    
    # Start with best known valid coloring
    if best_coloring is None:
        # Fallback to 17
        best_coloring = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        best_count = 17
    
    # Local search to reduce colors
    # Strategy: Try to eliminate one color at a time.
    # We maintain a list of available colors.
    # Initially, colors are 0..best_count-1.
    
    current_coloring = best_coloring[:]
    current_colors = set(current_coloring)
    available_colors = list(current_colors)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        if not available_colors:
            break
            
        # Pick a color to remove
        # Prefer colors with fewer edges? Or random?
        # Let's pick a random color to remove
        target_color = rng.choice(available_colors)
        
        # Find all edges with this color
        edges_to_recolor = [e for e in range(num_edges) if current_coloring[e] == target_color]
        
        if not edges_to_recolor:
            # Color not used, remove from available
            available_colors.remove(target_color)
            continue
            
        # Try to reassign each edge to a different color
        # We do this sequentially. If we fail, we revert.
        original_assignments = {}
        success = True
        
        for e in edges_to_recolor:
            original_assignments[e] = current_coloring[e]
            assigned = False
            
            # Try other colors
            other_colors = [c for c in available_colors if c != target_color]
            rng.shuffle(other_colors)
            
            for c in other_colors:
                # Tentatively assign
                current_coloring[e] = c
                
                # Check validity for all K4s containing e
                valid = True
                for k4_idx in edge_to_k4s[e]:
                    e_indices = k4_edges[k4_idx]
                    color_set = set(current_coloring[ei] for ei in e_indices)
                    if len(color_set) < 5:
                        valid = False
                        break
                
                if valid:
                    assigned = True
                    break
                
            if not assigned:
                # Revert changes for this edge and previous ones in this attempt?
                # It's easier to just revert all and try next target color
                for e2 in edges_to_recolor:
                    current_coloring[e2] = original_assignments.get(e2, current_coloring[e2])
                success = False
                break
                
        if success:
            # Successfully removed target_color
            available_colors.remove(target_color)
            # Update current_colors if needed, but available_colors tracks it
            # We can also try to remove more colors immediately
            # The loop will handle it
    
    final_count = len(set(current_coloring))
    
    if final_count < best_count:
        return current_coloring
    else:
        return best_coloring
