import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Vertices 0-3 are S, 4-15 are L
    # neighbors[u] for u in 0..11 (L vertices) is a 4-bit mask for S vertices
    
    def has_q3(neighbors):
        # Check all 4-subsets of L
        # There are C(12, 4) = 495 subsets
        for combo in itertools.combinations(range(12), 4):
            # Build the 4x4 adjacency matrix
            # Rows are S vertices (0-3), Cols are L vertices (combo)
            # adj[i][j] is 1 if S_i is connected to L_j
            
            # Check if this 4x4 graph contains Q3
            # Q3 is 3-regular bipartite on 8 vertices.
            # So we need to check if there is a subgraph isomorphic to Q3.
            # Since Q3 has 12 edges, and the max edges here is 16.
            # We can check if the graph is "dense enough" and has the right structure.
            # A simpler check: 
            # If the graph contains K4,4, it contains Q3.
            # If the graph is K4,4 minus 1 edge, it contains Q3.
            # If the graph is K4,4 minus 2 edges, it might contain Q3.
            # Q3 has 12 edges.
            # So if the induced graph has >= 12 edges, it *might* contain Q3.
            # But not all graphs with 12 edges contain Q3.
            # However, if it has 16, 15, 14, 13, 12 edges, we need to be sure.
            
            # Let's just check if it contains Q3 by trying all isomorphisms?
            # Too slow.
            
            # Heuristic: If the number of edges is >= 12, assume it contains Q3?
            # No, C8 has 8 edges. Q3 has 12.
            # A 3-regular bipartite graph on 8 vertices has 12 edges.
            # If the induced graph is 3-regular, it is a candidate.
            # If it has more edges, it contains a 3-regular subgraph?
            # Not necessarily.
            
            # Let's use the property: Q3 is the only 3-regular bipartite graph on 8 vertices with girth 4?
            # No.
            
            # Safe check: 
            # Count edges. If < 12, no Q3.
            # If >= 12, we need to check.
            # For simplicity in this time-constrained context, I will assume that any induced subgraph with >= 12 edges contains Q3.
            # This is a sufficient condition for removal, but might be too aggressive.
            
            edge_count = 0
            for u in combo:
                edge_count += bin(neighbors[u]).count('1')
                
            if edge_count >= 12:
                return True
                
        return False

    def count_edges(neighbors):
        return sum(bin(n).count('1') for n in neighbors)

    best_edges = 0
    best_neighbors = None
    
    for _ in range(100): # Restart limit
        if time.time() > deadline:
            break
            
        neighbors = []
        for _ in range(12):
            # Random mask
            mask = random.randint(0, 15)
            neighbors.append(mask)
            
        # Local search
        # Try to add edges
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Pick a random vertex in L and a random bit to set
            u = random.randint(0, 11)
            bit = random.randint(0, 3)
            
            if (neighbors[u] >> bit) & 1:
                continue
                
            neighbors[u] |= (1 << bit)
            
            if has_q3(neighbors):
                # Revert
                neighbors[u] &= ~(1 << bit)
            else:
                curr_edges = count_edges(neighbors)
                if curr_edges > best_edges:
                    best_edges = curr_edges
                    best_neighbors = neighbors[:]
                    
    if best_neighbors is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for u in range(12):
        mask = best_neighbors[u]
        for v in range(4):
            if (mask >> v) & 1:
                edges.append([4 + u, v])
                
    return edges
