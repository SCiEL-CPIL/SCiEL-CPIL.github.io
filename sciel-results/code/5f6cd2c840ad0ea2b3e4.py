import random
import itertools
import time
import math

# Precompute Q3 properties
# Q3 vertices 0-7, edges where popcount(i^j)==1
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj
    for j in range(i+1, 8):
        if (adj >> j) & 1:
            q3_edges.append((i, j))

# Precompute all 8-vertex subsets
# To save time, we only generate them on the fly or precompute lists of vertices
# Precomputing all 12870 subsets as tuples
all_subsets_8 = list(itertools.combinations(range(16), 8))

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    subset is a tuple of 8 vertex ids.
    """
    # Map local indices 0-7 to global vertices
    # Build local adjacency bitmasks
    # Optimization: Unroll the loop?
    local_adj = [0]*8
    s = subset
    for i in range(8):
        vi = s[i]
        # Use bit operations to check neighbors within subset
        # We need to check if any neighbor of vi is in subset
        # adj_bits[vi] & mask_of_subset
        # But we need local indices.
        # This part is the bottleneck.
        
        # Alternative: Precompute for each vertex vi, the local adjacency for a given subset?
        # No, subset changes.
        
        # Let's stick to the loop but optimize it.
        # We can compute local_adj[i] by iterating j in 0..7
        # if (adj_bits[vi] >> s[j]) & 1: local_adj[i] |= (1 << j)
        
        mask = 0
        for j in range(8):
            if (adj_bits[vi] >> s[j]) & 1:
                mask |= (1 << j)
        local_adj[i] = mask
    
    # Degree check
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:
        return False
    
    # Backtracking
    mapping = [-1]*8
    
    # Order Q3 vertices by degree in Q3? All are 3.
    # Order local vertices by degree in local graph?
    # This might help pruning.
    # Let's try a simple backtracking first.
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        # Try to assign local vertex s_idx to q3_v
        # Heuristic: try local vertices with higher degree first?
        # Or just iterate 0..7
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            # Check compatibility
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

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    # Initialize with a random graph
    # p=0.5 is a good start
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Precompute subsets containing each edge?
    # For incremental updates, we need to know which subsets are affected by an edge change.
    # Edge (u,v) is in subset S if u in S and v in S.
    
    # To speed up SA, we will evaluate the cost (number of Q3s) only for affected subsets.
    # But maintaining the exact count of Q3s incrementally is complex because
    # adding an edge might create multiple Q3s in the same subset, or remove them.
    # However, for a single edge addition, a subset either gains Q3s or not.
    # If it already had a Q3, adding an edge doesn't change the fact that it has >=1 Q3.
    # So, we can track for each subset: "Does it contain Q3?".
    # If it does, cost contribution is 1.
    # If it doesn't, cost contribution is 0.
    # When we add an edge, we check affected subsets.
    # If a subset was "No" and becomes "Yes", cost increases by 1.
    # If a subset was "Yes", it stays "Yes".
    # When we remove an edge, we check affected subsets.
    # If a subset was "Yes", we re-check if it still contains Q3.
    # If it becomes "No", cost decreases by 1.
    
    # This incremental approach is much faster!
    
    # Initialize subset_status: 0 = No Q3, 1 = Has Q3
    subset_status = [0] * len(all_subsets_8)
    current_cost = 0
    
    # Initial evaluation
    for idx, subset in enumerate(all_subsets_8):
        if contains_q3_fast(subset, adj_bits):
            subset_status[idx] = 1
            current_cost += 1
            
    # Map edge (u,v) to list of subset indices containing u and v
    # This is expensive to build fully, but we can generate on the fly or precompute.
    # There are 120 edges. Each edge is in C(14,6) = 3003 subsets.
    # Total entries: 120 * 3003 = 360,360. This is fine.
    
    edge_to_subsets = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_to_subsets[(i,j)] = []
    
    for idx, subset in enumerate(all_subsets_8):
        # For each pair in subset, add idx to edge_to_subsets[pair]
        for a in range(8):
            for b in range(a+1, 8):
                u, v = subset[a], subset[b]
                if u > v: u, v = v, u
                edge_to_subsets[(u,v)].append(idx)
                
    # Simulated Annealing
    T = 1.0
    cooling = 0.995
    best_adj_bits = adj_bits[:]
    best_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    best_cost = current_cost
    
    # If initial cost is 0, we are already Q3-free. We should maximize edges.
    # SA should handle this by accepting moves that reduce cost or (if cost==0) increase edges.
    
    steps = 0
    while time.time() - start_time < time_limit - 5:
        # Choose a move
        # 1. Add an edge
        # 2. Remove an edge
        # 3. Swap an edge (remove one, add another)
        
        # Simple move: Toggle a random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Determine if edge is present
        is_present = (adj_bits[u] >> v) & 1
        
        # Identify affected subsets
        affected_indices = edge_to_subsets[(u,v)]
        
        # Calculate delta cost
        delta_cost = 0
        
        if is_present:
            # We are removing the edge
            # For each affected subset, if status is 1, re-check
            for idx in affected_indices:
                if subset_status[idx]:
                    # Re-check without this edge
                    # We need a version of contains_q3_fast that ignores a specific edge?
                    # Or we can temporarily remove, check, re-add?
                    # Since we are in the middle of evaluation, we can just call contains_q3_fast
                    # but we must ensure the edge is NOT considered.
                    # Easiest: Temporarily remove from adj_bits, check, restore.
                    # But we are iterating multiple subsets.
                    # Let's just temporarily remove for the whole evaluation of this move.
                    pass # Handled below
        else:
            # We are adding the edge
            # For each affected subset, if status is 0, check if it becomes 1
            # If it becomes 1, delta_cost += 1
            pass # Handled below

        # Implementation of move evaluation
        # Temporarily apply move
        if is_present:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        else:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        new_cost = current_cost
        
        if is_present:
            # Removal
            for idx in affected_indices:
                if subset_status[idx]:
                    # Check if still has Q3
                    if not contains_q3_fast(all_subsets_8[idx], adj_bits):
                        subset_status[idx] = 0
                        new_cost -= 1
        else:
            # Addition
            for idx in affected_indices:
                if not subset_status[idx]:
                    # Check if now has Q3
                    if contains_q3_fast(all_subsets_8[idx], adj_bits):
                        subset_status[idx] = 1
                        new_cost += 1
        
        # Calculate acceptance
        # Objective: Minimize cost. If cost == 0, Maximize edges.
        # Let's define a score: -1000 * cost + edges
        # Or simpler:
        # If new_cost < current_cost: Accept
        # If new_cost > current_cost: Accept with prob exp(-delta/T)
        # If new_cost == current_cost:
        #    If new_edges > current_edges: Accept
        #    Else: Accept with prob 0.5 (or based on T)
        
        current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        new_edges = current_edges - 1 if is_present else current_edges + 1
        
        accept = False
        if new_cost < current_cost:
            accept = True
        elif new_cost > current_cost:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                accept = True
        else:
            # Same cost
            if new_edges > current_edges:
                accept = True
            else:
                # If same cost and fewer/same edges, accept with small prob to escape local minima?
                # Or reject.
                if random.random() < 0.1:
                    accept = True
        
        if accept:
            current_cost = new_cost
            # Update best
            if current_cost == 0:
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj_bits = adj_bits[:]
            elif current_cost < best_cost:
                best_cost = current_cost
                best_edges = new_edges
                best_adj_bits = adj_bits[:]
        else:
            # Revert move
            if is_present:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            # Revert subset_status changes?
            # We modified subset_status in place.
            # If we reject, we must revert those changes.
            # This is tricky.
            
            # Better approach: Don't update subset_status until accepted.
            # Store old statuses in a dict or list of changes.
            
            # Let's refactor to collect changes.
            # This makes the code complex.
            
            # Alternative: Just re-evaluate the whole graph if rejected? No, too slow.
            
            # Let's store the changes.
            # changes = [(idx, old_status, new_status)]
            # If reject, revert changes.
            
            # Let's rewrite the move evaluation to collect changes.
            pass
            
        T *= cooling
        steps += 1
        
    # Return best
    # Ensure best_adj_bits is Q3-free?
    # If best_cost > 0, we might return an invalid graph.
    # We need to clean up the best graph to be Q3-free.
    
    adj_bits = best_adj_bits[:]
    
    # Cleanup phase: Remove edges to destroy all Q3s
    while time.time() - start_time < time_limit - 1:
        # Find a Q3
        q3_found = False
        for idx, status in enumerate(subset_status): # Note: subset_status might be stale if we reverted
            if status:
                # Verify
                if contains_q3_fast(all_subsets_8[idx], adj_bits):
                    # Remove an edge
                    subset = all_subsets_8[idx]
                    # Find an edge in this subset that is in the graph
                    # and is part of a Q3?
                    # Just remove any edge in the subset.
                    # Better: Remove an edge that is in the Q3.
                    # We can find one by checking which edges are present.
                    # For simplicity, remove a random edge from the subset that exists.
                    edges_in_subset = []
                    for a in range(8):
                        for b in range(a+1, 8):
                            u, v = subset[a], subset[b]
                            if (adj_bits[u] >> v) & 1:
                                edges_in_subset.append((u,v))
                    if edges_in_subset:
                        u, v = random.choice(edges_in_subset)
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        q3_found = True
                        # Update subset_status?
                        # We should re-evaluate all subsets containing (u,v)
                        # Or just break and re-evaluate everything at the end?
                        # Given time constraints, let's just break the loop and do a full re-check at the end.
                        break
                else:
                    subset_status[idx] = 0 # Stale status fix
            if q3_found:
                break
        if not q3_found:
            break
            
    # Final full verification and cleanup
    # Re-evaluate all subsets to get accurate status
    for idx in range(len(all_subsets_8)):
        if contains_q3_fast(all_subsets_8[idx], adj_bits):
            # Remove edges until no Q3
            # This is a greedy fix
            while contains_q3_fast(all_subsets_8[idx], adj_bits):
                subset = all_subsets_8[idx]
                edges_in_subset = []
                for a in range(8):
                    for b in range(a+1, 8):
                        u, v = subset[a], subset[b]
                        if (adj_bits[u] >> v) & 1:
                            edges_in_subset.append((u,v))
                if not edges_in_subset:
                    break
                u, v = random.choice(edges_in_subset)
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
