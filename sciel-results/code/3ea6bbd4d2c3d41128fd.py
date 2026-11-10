import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    random.seed(42)

    # Precompute all 8-subsets of vertices
    all_subsets_8 = list(itertools.combinations(range(n), 8))
    
    # Precompute bitmasks for all 4-subsets of an 8-set? 
    # Actually, we just need to iterate 4-subsets of the 8-subset.
    # There are 35 such partitions (combinations of 4 from 8, divided by 2 for symmetry? 
    # No, A and B are distinct roles in the bipartite check, but the condition is symmetric.
    # If (A,B) works, (B,A) works. So we can just iterate 4-subsets A, and B is the rest.
    # We don't need to divide by 2 if we check all A.
    
    # Precompute list of 4-subsets of indices 0..7
    # But the vertices in the 8-subset are mapped to global indices.
    # It's easier to just iterate combinations of the 8 global vertices.
    
    # To speed up, we can precompute the list of 4-subsets of {0,1,2,3,4,5,6,7}
    # and map them to the global vertices of the 8-subset.
    local_4_subsets = list(itertools.combinations(range(8), 4))
    
    def is_q3_free(adj):
        """
        Check if the graph defined by adj (list of 16 bitmasks) is Q3-free.
        Returns True if Q3-free, False otherwise.
        """
        for S in all_subsets_8:
            # S is a tuple of 8 global vertex indices
            # Create a bitmask for the entire set S? Not directly needed for the check logic below.
            
            # For each local 4-subset A_local (indices 0..7), map to global A
            for A_local in local_4_subsets:
                A = [S[i] for i in A_local]
                B = [S[i] for i in range(8) if i not in A_local]
                
                # Calculate bitmask for A and B
                mask_A = 0
                for v in A:
                    mask_A |= (1 << v)
                mask_B = 0
                for v in B:
                    mask_B |= (1 << v)
                
                # Check if bipartite graph between A and B is 3-regular
                is_q3 = True
                for a in A:
                    if (adj[a] & mask_B).bit_count() != 3:
                        is_q3 = False
                        break
                if not is_q3:
                    continue
                
                for b in B:
                    if (adj[b] & mask_A).bit_count() != 3:
                        is_q3 = False
                        break
                
                if is_q3:
                    return False
        
        return True

    # Initialize graph
    # Start with a random graph and remove invalid edges
    adj = [0] * n
    edge_count = 0
    
    # Generate random edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # Add edges one by one, checking validity
    # This is slow if done naively. 
    # Better: Start with K16, remove edges that form Q3s.
    adj = [(1 << n) - 2] * n # Mask out self-loops? 
    # adj[i] should have bit i unset.
    for i in range(n):
        adj[i] &= ~(1 << i)
    
    # K16 is definitely not Q3-free.
    # We need to remove edges.
    # We can use a greedy removal:
    # While not is_q3_free(adj):
    #   Find a Q3, remove a random edge from it.
    
    # To find a Q3 quickly, we can reuse the logic from is_q3_free but stop at the first one found.
    def find_q3(adj):
        for S in all_subsets_8:
            for A_local in local_4_subsets:
                A = [S[i] for i in A_local]
                B = [S[i] for i in range(8) if i not in A_local]
                
                mask_A = 0
                for v in A:
                    mask_A |= (1 << v)
                mask_B = 0
                for v in B:
                    mask_B |= (1 << v)
                
                is_q3 = True
                for a in A:
                    if (adj[a] & mask_B).bit_count() != 3:
                        is_q3 = False
                        break
                if not is_q3:
                    continue
                
                for b in B:
                    if (adj[b] & mask_A).bit_count() != 3:
                        is_q3 = False
                        break
                
                if is_q3:
                    # Return one of the edges in this Q3
                    # Edges are between A and B
                    return (A[0], B[0])
        return None

    # Greedy removal to get a valid start
    while time.time() < deadline - 60.0:
        q3_edge = find_q3(adj)
        if q3_edge is None:
            break
        u, v = q3_edge
        # Remove edge (u, v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_count -= 1
    
    # Now we have a valid graph.
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_adj = adj[:]
    best_edge_count = edge_count
    
    current_adj = adj[:]
    current_edge_count = edge_count
    
    # Precompute list of all edges for random selection
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline - 5.0:
        # Move: Try to add an edge
        # Pick a random non-edge
        u, v = random.choice(all_edges)
        if (current_adj[u] >> v) & 1:
            continue # Already present
        
        # Tentatively add
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        new_edge_count = current_edge_count + 1
        
        if is_q3_free(current_adj):
            current_edge_count = new_edge_count
            if current_edge_count > best_edge_count:
                best_edge_count = current_edge_count
                best_adj = current_adj[:]
        else:
            # Revert
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            # Try a swap? Remove a random edge, then add this one?
            # For simplicity, just reject for now. 
            # Swaps are better for escaping local optima.
            # Let's implement swaps in the next loop iteration or as an alternative.
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check: try to add any remaining edges
    for u, v in all_edges:
        if time.time() > deadline - 2.0:
            break
        if (best_adj[u] >> v) & 1:
            continue
        best_adj[u] |= (1 << v)
        best_adj[v] |= (1 << u)
        if is_q3_free(best_adj):
            best_edge_count += 1
        else:
            best_adj[u] &= ~(1 << v)
            best_adj[v] &= ~(1 << u)

    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result

