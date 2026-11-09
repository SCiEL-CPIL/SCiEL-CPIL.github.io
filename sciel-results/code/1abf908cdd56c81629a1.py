import random
import itertools
import time

# Precompute Q3 properties
# We determined that a graph on 8 vertices is Q3 iff it is 3-regular and bipartite.

def is_q3_fast(vertices, adj_bits):
    """
    Check if the induced subgraph on 'vertices' (tuple of 8) is Q3.
    Q3 is 3-regular and bipartite.
    """
    # Map global vertices to local 0-7 for easier bit manipulation if needed, 
    # but we can just use the global adj_bits.
    
    # 1. Check 3-regular
    for i in range(8):
        u = vertices[i]
        deg = 0
        for j in range(8):
            if i != j:
                v = vertices[j]
                if (adj_bits[u] >> v) & 1:
                    deg += 1
        if deg != 3:
            return False
    
    # 2. Check Bipartite
    # Colors: -1 unvisited, 0, 1
    colors = [-1] * 8
    # Start from vertex 0
    colors[0] = 0
    queue = [0]
    
    while queue:
        u_local = queue.pop()
        u_global = vertices[u_local]
        u_color = colors[u_local]
        
        # Iterate over neighbors in the induced subgraph
        # We need to find which local vertices are neighbors of u_global
        # adj_bits[u_global] has bits for all global neighbors.
        # We need to check which of vertices[j] are in adj_bits[u_global].
        
        # To make this fast, we can precompute local adjacency masks for the subset?
        # Or just loop.
        for v_local in range(8):
            if u_local == v_local:
                continue
            v_global = vertices[v_local]
            if (adj_bits[u_global] >> v_global) & 1:
                # Edge exists
                if colors[v_local] == -1:
                    colors[v_local] = 1 - u_color
                    queue.append(v_local)
                elif colors[v_local] == u_color:
                    # Conflict, not bipartite
                    return False
    return True

def construct():
    n = 16
    total_edges = n * (n-1) // 2  # 120
    
    # Precompute all 8-vertex subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # Map edge (u,v) to list of subset indices that contain u and v
    # This allows fast update of q3_count
    edge_to_subsets = {}
    for i, subset in enumerate(all_subsets):
        # Add all pairs in this subset
        for a in range(8):
            for b in range(a+1, 8):
                u, v = subset[a], subset[b]
                key = (min(u,v), max(u,v))
                if key not in edge_to_subsets:
                    edge_to_subsets[key] = []
                edge_to_subsets[key].append(i)
    
    # Initialize random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    def count_q3s():
        count = 0
        for subset in all_subsets:
            if is_q3_fast(subset, adj_bits):
                count += 1
        return count
    
    start_time = time.time()
    time_limit = 118
    
    # Initial state
    current_q3s = count_q3s()
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    current_cost = -current_edges + 10.0 * current_q3s
    
    best_edges = current_edges
    best_q3s = current_q3s
    best_adj_bits = adj_bits[:]
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        key = (min(u,v), max(u,v))
        
        # Calculate delta in q3s
        # Only subsets containing this edge can change status
        delta_q3s = 0
        affected_subsets = edge_to_subsets.get(key, [])
        
        # We need to check the state BEFORE and AFTER flip for these subsets
        # Since we only flip one edge, only subsets containing this edge are affected.
        
        # Check current state for affected subsets
        current_q3_count_affected = 0
        for idx in affected_subsets:
            if is_q3_fast(all_subsets[idx], adj_bits):
                current_q3_count_affected += 1
        
        # Flip edge
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        # Check new state for affected subsets
        new_q3_count_affected = 0
        for idx in affected_subsets:
            if is_q3_fast(all_subsets[idx], adj_bits):
                new_q3_count_affected += 1
        
        delta_q3s = new_q3_count_affected - current_q3_count_affected
        
        # Delta edges
        delta_edges = 1 if (adj_bits[u] >> v) & 1 else -1
        
        # Delta cost
        delta_cost = -delta_edges + 10.0 * delta_q3s
        
        # Acceptance
        if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
            # Accept
            current_q3s += delta_q3s
            current_edges += delta_edges
            current_cost += delta_cost
            
            if current_q3s == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_q3s = 0
                best_adj_bits = adj_bits[:]
                
            # Even if q3s > 0, we might want to keep track of best "penalty" state?
            # No, the goal is Q3-free. 
            # However, SA might need to traverse states with Q3s to get to better ones.
            # We only update 'best' if it is Q3-free.
            
        else:
            # Reject: Flip back
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
        
        T *= cooling
    
    # Final cleanup: If best_adj_bits has Q3s (shouldn't if we only update on 0), 
    # but just in case, or if we didn't find any 0-Q3 state, we need to ensure validity.
    # The problem requires returning a Q3-free graph.
    # If best_adj_bits has Q3s, we must remove edges to make it free.
    
    final_adj_bits = best_adj_bits[:]
    
    # Verify and clean
    while time.time() - start_time < time_limit - 1:
        q3_count = 0
        # Find one Q3 to break
        # We can just scan
        found = None
        for subset in all_subsets:
            if is_q3_fast(subset, final_adj_bits):
                found = subset
                break
        
        if found is None:
            break
            
        # Remove a random edge from this Q3
        # The Q3 has 12 edges in the induced subgraph.
        # Pick a random pair from 'found' that is an edge in final_adj_bits
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = found[i], found[j]
                if (final_adj_bits[u] >> v) & 1:
                    edges_in_q3.append((u, v))
        
        if not edges_in_q3:
            break
            
        u, v = random.choice(edges_in_q3)
        final_adj_bits[u] &= ~(1 << v)
        final_adj_bits[v] &= ~(1 << u)
        
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (final_adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
