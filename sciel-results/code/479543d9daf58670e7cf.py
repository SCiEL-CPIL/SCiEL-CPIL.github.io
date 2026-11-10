import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    q = 5
    # Build PG(2, q) points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31
    target_edges = 91
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph base edges
    base_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.add((i, j))
    
    # Precompute all possible non-base edges (candidates to add)
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    candidate_edges = list(all_pairs - base_edges)
    random.shuffle(candidate_edges)
    
    # We need to add (target_edges - len(base_edges)) edges
    num_to_add = target_edges - len(base_edges)
    
    # Initialize with base edges
    edges = set(base_edges)
    
    # Precompute neighbor sets for fast C4 counting
    def get_neighbors(edges_set):
        neighbors = [set() for _ in range(n)]
        for u, v in edges_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors
    
    def count_total_c4s(edges_set):
        """Count total number of C4 subgraphs in the graph."""
        neighbors = get_neighbors(edges_set)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbors of i and j
                common = neighbors[i] & neighbors[j]
                count += len(common) * (len(common) - 1) // 2
        return count
    
    # Initial C4 count
    current_c4 = count_total_c4s(edges)
    best_c4 = current_c4
    best_edges = set(edges)
    
    # Simulated annealing to select which candidate edges to add
    # State: which subset of candidate_edges is included
    # We need exactly num_to_add edges from candidate_edges
    
    if num_to_add > 0:
        # Start with first num_to_add candidates
        selected_indices = list(range(num_to_add))
        
        # Precompute edge list for current state
        def get_current_edges(selected_indices):
            e = set(base_edges)
            for idx in selected_indices:
                e.add(candidate_edges[idx])
            return e
        
        current_edges = get_current_edges(selected_indices)
        current_c4 = count_total_c4s(current_edges)
        best_c4 = current_c4
        best_edges = set(current_edges)
        best_selected = set(selected_indices)
        
        # Simulated annealing
        temperature = 100.0
        cooling_rate = 0.999
        min_temperature = 0.1
        max_iterations = 50000
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if temperature < min_temperature:
                break
            
            # Choose a random edge to remove and a random edge to add
            if len(selected_indices) == 0:
                break
                
            remove_idx = random.randint(0, len(selected_indices) - 1)
            removed_edge = selected_indices[remove_idx]
            
            # Choose a candidate edge not currently selected
            available = [i for i in range(len(candidate_edges)) if i not in selected_indices]
            if not available:
                break
            add_idx = random.choice(available)
            
            # Update selected_indices
            new_selected = selected_indices.copy()
            new_selected[remove_idx] = add_idx
            new_selected_set = set(new_selected)
            
            # Compute new C4 count
            new_edges = get_current_edges(new_selected)
            new_c4 = count_total_c4s(new_edges)
            
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                selected_indices = new_selected
                current_edges = new_edges
                current_c4 = new_c4
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = set(current_edges)
                    best_selected = set(selected_indices)
            
            temperature *= cooling_rate
    
    # Use the best edges found
    edges = best_edges
    
    # If we don't have exactly target_edges, fix it greedily
    while len(edges) < target_edges:
        if time.time() > deadline:
            break
        # Add random candidate edge
        added = False
        for edge in candidate_edges:
            if edge not in edges:
                edges.add(edge)
                added = True
                break
        if not added:
            break
    
    # If we have too many edges (shouldn't happen), remove randomly
    while len(edges) > target_edges:
        if time.time() > deadline:
            break
        edge_to_remove = random.choice(list(edges))
        edges.remove(edge_to_remove)
    
    # Final verification and cleanup
    result = [[u, v] for u, v in edges]
    
    # Ensure exactly 91 edges
    if len(result) != target_edges:
        # Fallback: use base edges + first needed candidates
        edges = set(base_edges)
        for edge in candidate_edges:
            if len(edges) >= target_edges:
                break
            if edge not in edges:
                edges.add(edge)
        result = [[u, v] for u, v in edges]
    
    return result
