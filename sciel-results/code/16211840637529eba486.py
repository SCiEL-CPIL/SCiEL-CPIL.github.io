import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
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
    
    # Efficient C4 counting using neighbor sets
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
    
    # Incremental C4 count update when adding/removing an edge
    def compute_c4_delta_add(u, v, neighbors, edges_set):
        """Compute the change in C4 count when adding edge (u, v)."""
        # A C4 containing (u,v) has the form u-a-b-v or u-b-a-v
        # where a is a neighbor of v, b is a neighbor of u, a != b
        # and (a,b) is an edge
        count = 0
        for a in neighbors[v]:
            if a == u:
                continue
            for b in neighbors[u]:
                if b == v or b == a:
                    continue
                if (min(a, b), max(a, b)) in edges_set:
                    count += 1
        return count
    
    def compute_c4_delta_remove(u, v, neighbors, edges_set):
        """Compute the change in C4 count when removing edge (u, v)."""
        # Same as add, but we're removing
        count = 0
        for a in neighbors[v]:
            if a == u:
                continue
            for b in neighbors[u]:
                if b == v or b == a:
                    continue
                if (min(a, b), max(a, b)) in edges_set:
                    count += 1
        return count
    
    # Start with base edges
    current_edges = set(base_edges)
    current_neighbors = get_neighbors(current_edges)
    current_c4 = count_total_c4s(current_edges)
    best_c4 = current_c4
    best_edges = set(current_edges)
    
    # We need to add num_to_add edges from candidate_edges
    if num_to_add > 0:
        # Greedy initialization: add edges that minimize incremental C4 increase
        selected_edges = set()
        
        for _ in range(num_to_add):
            if time.time() > deadline:
                break
            
            best_edge = None
            best_delta = float('inf')
            
            # Evaluate each candidate edge
            for edge in candidate_edges:
                if edge in current_edges or edge in selected_edges:
                    continue
                
                u, v = edge
                delta = compute_c4_delta_add(u, v, current_neighbors, current_edges)
                
                if delta < best_delta:
                    best_delta = delta
                    best_edge = edge
            
            if best_edge is None:
                break
            
            # Add the best edge
            current_edges.add(best_edge)
            selected_edges.add(best_edge)
            u, v = best_edge
            current_neighbors[u].add(v)
            current_neighbors[v].add(u)
            current_c4 += best_delta
        
        # Now perform local search / simulated annealing
        # State: which candidate edges are selected (exactly num_to_add of them)
        selected_set = set(selected_edges)
        
        temperature = 50.0
        cooling_rate = 0.9995
        min_temperature = 0.01
        max_iterations = 30000
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if temperature < min_temperature:
                break
            
            # Choose a random selected edge to remove
            if not selected_set:
                break
            
            remove_edge = random.choice(list(selected_set))
            
            # Choose a random non-selected candidate edge to add
            available = [e for e in candidate_edges if e not in selected_set]
            if not available:
                break
            add_edge = random.choice(available)
            
            # Compute delta for removing remove_edge
            ru, rv = remove_edge
            delta_remove = compute_c4_delta_remove(ru, rv, current_neighbors, current_edges)
            
            # Compute delta for adding add_edge
            au, av = add_edge
            delta_add = compute_c4_delta_add(au, av, current_neighbors, current_edges)
            
            # Note: when removing then adding, the deltas might interact
            # For simplicity, approximate as sum
            delta = delta_remove + delta_add
            
            # Simulated annealing acceptance
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                # Accept the move
                current_edges.remove(remove_edge)
                current_edges.add(add_edge)
                selected_set.remove(remove_edge)
                selected_set.add(add_edge)
                
                # Update neighbors
                current_neighbors[ru].remove(rv)
                current_neighbors[rv].remove(ru)
                current_neighbors[au].add(av)
                current_neighbors[av].add(au)
                
                current_c4 += delta
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = set(current_edges)
            
            temperature *= cooling_rate
        
        # Final polish: try to improve by swapping edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, e1 in enumerate(candidate_edges):
                if e1 not in selected_set:
                    continue
                for j in range(i + 1, len(candidate_edges)):
                    e2 = candidate_edges[j]
                    if e2 in selected_set:
                        continue
                    
                    # Try swapping e1 (remove) with e2 (add)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    delta_remove = compute_c4_delta_remove(u1, v1, current_neighbors, current_edges)
                    delta_add = compute_c4_delta_add(u2, v2, current_neighbors, current_edges)
                    delta = delta_remove + delta_add
                    
                    if delta < 0:
                        # Accept improvement
                        current_edges.remove(e1)
                        current_edges.add(e2)
                        selected_set.remove(e1)
                        selected_set.add(e2)
                        
                        current_neighbors[u1].remove(v1)
                        current_neighbors[v1].remove(u1)
                        current_neighbors[u2].add(v2)
                        current_neighbors[v2].add(u2)
                        
                        current_c4 += delta
                        improved = True
                        
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_edges = set(current_edges)
    
    # Use the best edges found
    edges = best_edges
    
    # If we don't have exactly target_edges, fix it greedily
    while len(edges) < target_edges:
        if time.time() > deadline:
            break
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
