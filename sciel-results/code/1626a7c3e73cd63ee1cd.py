import random
import itertools
import time

# Precompute Q3 properties
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    subset is a tuple/list of 8 vertex ids.
    """
    # Map local indices 0-7 to global vertices
    # Build local adjacency bitmasks
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    # Check if local_adj contains Q3 as subgraph
    # Q3 vertices 0-7, edges where popcount(i^j)==1
    # Use backtracking with pruning
    mapping = [-1]*8
    
    # Degree check: all must have degree >= 3 in local graph
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:  # 12 edges * 2
        return False
    
    # Backtracking
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        # Try each unmapped local vertex
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            # Check compatibility with already mapped neighbors
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
    
    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
    
    return False

def find_q3_subset(adj_bits, n):
    """Find any 8-vertex subset containing Q3. Returns subset or None."""
    # Random sampling first
    for _ in range(100):
        subset = random.sample(range(n), 8)
        if contains_q3_fast(subset, adj_bits):
            return list(subset)
    
    # Full scan with early termination
    for subset in itertools.combinations(range(n), 8):
        if contains_q3_fast(subset, adj_bits):
            return list(subset)
    
    return None

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    best_edges = 0
    best_adj_bits = [0]*n
    
    # Multiple restarts to escape local optima
    for restart in range(20):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Initialize with random graph
        # Density 0.8 is a good starting point for n=16
        adj_bits = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.8:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Phase 1: Repair (Remove edges to destroy Q3s)
        while time.time() - start_time < time_limit - 5:
            q3_sub = find_q3_subset(adj_bits, n)
            if q3_sub is None:
                break
            
            # Find edges in this Q3
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                        edges_in_q3.append((q3_sub[i], q3_sub[j]))
            
            if not edges_in_q3:
                break
            
            # Remove a random edge from the Q3
            u, v = random.choice(edges_in_q3)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        
        # Phase 2: Greedy Add
        # Try to add all missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj_bits[i] >> j) & 1):
                    missing.append((i, j))
        
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Temporarily add
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            # Check if creates Q3
            creates_q3 = False
            other_vertices = [x for x in range(n) if x != u and x != v]
            
            # Optimization: Only check subsets containing u and v
            for rest in itertools.combinations(other_vertices, 6):
                subset = list(rest) + [u, v]
                if contains_q3_fast(subset, adj_bits):
                    creates_q3 = True
                    break
            
            if creates_q3:
                # Revert
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
        
        # Calculate current edge count
        current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
        
        # Phase 3: Local Search (Swaps)
        # Try to improve by swapping edges
        no_improve_count = 0
        max_no_improve = 500
        
        while no_improve_count < max_no_improve and time.time() - start_time < time_limit - 2:
            # Get current edge set and missing set
            edge_set = set()
            missing_set = set()
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_bits[i] >> j) & 1:
                        edge_set.add((i, j))
                    else:
                        missing_set.add((i, j))
            
            if not edge_set or not missing_set:
                break
            
            # Pick a random edge to remove
            u1, v1 = random.choice(list(edge_set))
            
            # Try a few random missing edges to add
            found_swap = False
            candidates = random.sample(list(missing_set), min(10, len(missing_set)))
            
            for u2, v2 in candidates:
                # Perform swap
                adj_bits[u1] &= ~(1 << v1)
                adj_bits[v1] &= ~(1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                # Check Q3
                has_q3 = find_q3_subset(adj_bits, n) is not None
                
                if not has_q3:
                    # Keep swap
                    found_swap = True
                    no_improve_count = 0
                    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj_bits = adj_bits[:]
                    break
                else:
                    # Revert
                    adj_bits[u1] |= (1 << v1)
                    adj_bits[v1] |= (1 << u1)
                    adj_bits[u2] &= ~(1 << v2)
                    adj_bits[v2] &= ~(1 << u2)
            
            if not found_swap:
                no_improve_count += 1
        
        # If we didn't improve in this restart, we can break early if we've used too much time
        if current_edges == best_edges and restart > 5:
             # Optional: break if stuck
             pass

    # Final Verification on best_adj_bits
    adj_bits = best_adj_bits
    if time.time() - start_time < time_limit - 0.5:
        while time.time() - start_time < time_limit - 0.1:
            q3_sub = find_q3_subset(adj_bits, n)
            if q3_sub is None:
                break
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                        edges_in_q3.append((q3_sub[i], q3_sub[j]))
            if not edges_in_q3:
                break
            u, v = random.choice(edges_in_q3)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
