import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 250.0  # 300s limit, keep 50s margin
    
    # Target n
    target_n = 42
    
    # Initialize with Paley(37) + 5 random vertices
    # This provides a strong starting point
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Adjacency list for Paley
    adj = [set() for _ in range(target_n)]
    
    # Add Paley edges
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                
    # Add 5 random vertices (37-41)
    # Connect them randomly to existing and each other
    for i in range(q, target_n):
        for j in range(i):
            if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
        for j in range(i + 1, target_n):
             if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)

    # Convert to bitmask for fast operations
    # 42 bits fit in a 64-bit int
    masks = [0] * target_n
    for i in range(target_n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m

    def count_k4_in_mask(mask):
        """Count K4s in the subgraph induced by the vertices set in mask."""
        # Get list of vertices
        verts = [i for i in range(target_n) if (mask >> i) & 1]
        n_v = len(verts)
        if n_v < 4:
            return 0
        
        count = 0
        # Iterate over all 4-subsets
        for a in range(n_v):
            va = verts[a]
            # Neighbors of va within the subset
            # We need to check if the other 3 form a triangle in the induced subgraph?
            # No, K4 means all 4 are mutually connected.
            # So for a fixed vertex va, the other 3 must form a triangle in G[verts] and be connected to va.
            # Actually, simpler: just check all 4-tuples.
            pass
            
        # More efficient: 
        # For each vertex va, let N_va = neighbors of va within verts.
        # Count triangles in G[N_va].
        # This is complex to implement efficiently in pure python without precomputed triangle counts.
        
        # Given the small size (max 20 vertices in neighborhood), brute force 4-subsets is okay.
        # C(20,4) = 4845.
        # We do this for each vertex v (42 times) and for both N(v) and ~N(v).
        # Total ~400k checks per full energy eval.
        # In a local search, we update incrementally.
        
        # Let's just do brute force for now, it's simpler and less error prone.
        for a in range(n_v):
            for b in range(a + 1, n_v):
                for c in range(b + 1, n_v):
                    for d in range(c + 1, n_v):
                        va, vb, vc, vd = verts[a], verts[b], verts[c], verts[d]
                        # Check if all 6 edges exist
                        if ((masks[va] >> vb) & 1) and ((masks[va] >> vc) & 1) and ((masks[va] >> vd) & 1) and \
                           ((masks[vb] >> vc) & 1) and ((masks[vb] >> vd) & 1) and \
                           ((masks[vc] >> vd) & 1):
                            count += 1
        return count

    def get_energy(masks):
        total_k5 = 0
        total_ind5 = 0
        all_mask = (1 << target_n) - 1
        
        for v in range(target_n):
            n_mask = masks[v]
            # Count K4 in N(v) -> contributes to K5 count (each K5 counted 5 times)
            k4_count = count_k4_in_mask(n_mask)
            total_k5 += k4_count
            
            # Count K4 in ~N(v) (independent set in G)
            # ~N(v) is (all_mask ^ n_mask) & ~(1 << v)
            ind_mask = (all_mask ^ n_mask) & ~(1 << v)
            # To count independent sets of size 4, we count K4s in the complement graph induced by ind_mask.
            # Complement mask for a vertex u in ind_mask is: (ind_mask & ~(1<<u)) ^ masks[u]
            # This is getting complicated to do inside count_k4_in_mask which uses global masks.
            
            # Let's write a helper that takes a mask and a "complement" flag or just pass the relevant masks.
            # Actually, count_k4_in_mask uses `masks`. For independent sets, we need to check non-edges.
            
            # Let's modify count_k4_in_mask to accept a mode: 'clique' or 'independent'.
            # Or better, just count K4s in the complement graph.
            
            # For now, let's just count K4s in N(v) for cliques.
            # For independent sets, we can count K4s in the complement of G induced by ~N(v).
            
            # Let's create a function count_k4(mask, is_complement=False)
            pass
            
        # This brute force energy function is too slow for SA with many iterations.
        # We need incremental updates or a faster check.
        
        # Alternative: Just check if the graph is valid (E=0) occasionally.
        # Or use a simpler heuristic: Greedy construction.
        
        return total_k5, total_ind5 # Placeholder

    # Given the complexity and time, let's try a simpler approach:
    # 1. Try to find a known 42-vertex graph? No, we must construct it.
    # 2. Use a faster local search.
    
    # Let's implement a fast validity check using bit operations.
    # A graph is (5,5)-free if no K5 and no Ind5.
    
    def is_valid(masks):
        all_mask = (1 << target_n) - 1
        for v in range(target_n):
            # Check K5: Does N(v) contain a K4?
            n_mask = masks[v]
            if count_k4_in_mask(n_mask) > 0:
                return False
            
            # Check Ind5: Does ~N(v) contain an Ind4?
            # Ind4 in G is K4 in ~G.
            ind_mask = (all_mask ^ n_mask) & ~(1 << v)
            # We need to count K4s in the complement graph induced by ind_mask.
            # Let's write a specific counter for this.
            if count_k4_in_complement(ind_mask, masks) > 0:
                return False
        return True

    def count_k4_in_complement(mask, masks):
        """Count K4s in the complement graph induced by vertices in mask."""
        verts = [i for i in range(target_n) if (mask >> i) & 1]
        n_v = len(verts)
        if n_v < 4:
            return 0
        
        count = 0
        for a in range(n_v):
            for b in range(a + 1, n_v):
                for c in range(b + 1, n_v):
                    for d in range(c + 1, n_v):
                        va, vb, vc, vd = verts[a], verts[b], verts[c], verts[d]
                        # Check if all 6 non-edges exist in G
                        # i.e., none of the edges exist in G
                        if not ((masks[va] >> vb) & 1) and not ((masks[va] >> vc) & 1) and not ((masks[va] >> vd) & 1) and \
                           not ((masks[vb] >> vc) & 1) and not ((masks[vb] >> vd) & 1) and \
                           not ((masks[vc] >> vd) & 1):
                            count += 1
        return count

    # This is still slow. 400k checks per validity check.
    # If we do 1000 restarts, that's 400M checks. Too slow.
    
    # Let's reduce the search space or use a better initialization.
    # The Paley graph is (5,5)-free.
    # We can try to add vertices one by one using a greedy method with backtracking.
    
    def greedy_construct():
        # Start with Paley(37)
        n = 37
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
            
        # Try to add vertices up to 42
        for new_v in range(37, 42):
            added = False
            # Try all possible edge configurations? 2^37 is too many.
            # Greedy: Add edges one by one, or pick a random configuration and fix.
            
            # Heuristic: Connect new vertex to a random subset, then fix conflicts.
            # Let's try 100 random configurations for the new vertex.
            for attempt in range(100):
                # Randomly connect to existing vertices
                new_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        new_mask |= (1 << i)
                        
                # Check if adding this vertex with these edges creates a K5 or Ind5
                # We only need to check 5-sets involving the new vertex.
                # A K5 involving new_v means: new_v is connected to 4 vertices that form a K4.
                # An Ind5 involving new_v means: new_v is NOT connected to 4 vertices that form an Ind4.
                
                valid = True
                # Check K5
                # Find K4s in the neighborhood of new_v in the new graph.
                # The neighborhood of new_v is the set of vertices connected to it.
                # Let S be the set of neighbors.
                # We need to check if G[S] has a K4.
                # Note: The edges within S are unchanged from the previous graph.
                
                # Get neighbors
                neighbors = [i for i in range(n) if (new_mask >> i) & 1]
                # Check if G[neighbors] has a K4
                if count_k4_in_subset(neighbors, masks, n) > 0:
                    valid = False
                    
                if valid:
                    # Check Ind5
                    # Non-neighbors of new_v
                    non_neighbors = [i for i in range(n) if not ((new_mask >> i) & 1)]
                    # Check if G[non_neighbors] has an Ind4 (which is a K4 in ~G)
                    if count_ind4_in_subset(non_neighbors, masks, n) > 0:
                        valid = False
                        
                if valid:
                    # Add the vertex
                    adj.append(set())
                    for i in range(n):
                        if (new_mask >> i) & 1:
                            adj[new_v].add(i)
                            adj[i].add(new_v)
                            
                    # Update masks
                    masks.append(new_mask)
                    for i in range(n):
                        if (new_mask >> i) & 1:
                            masks[i] |= (1 << new_v)
                            
                    n += 1
                    added = True
                    break
                    
            if not added:
                break
                
        return n, adj

    def count_k4_in_subset(verts, masks, n):
        """Count K4s in the subgraph induced by verts, using existing masks."""
        n_v = len(verts)
        if n_v < 4:
            return 0
        count = 0
        for a in range(n_v):
            for b in range(a + 1, n_v):
                for c in range(b + 1, n_v):
                    for d in range(c + 1, n_v):
                        va, vb, vc, vd = verts[a], verts[b], verts[c], verts[d]
                        if ((masks[va] >> vb) & 1) and ((masks[va] >> vc) & 1) and ((masks[va] >> vd) & 1) and \
                           ((masks[vb] >> vc) & 1) and ((masks[vb] >> vd) & 1) and \
                           ((masks[vc] >> vd) & 1):
                            count += 1
        return count

    def count_ind4_in_subset(verts, masks, n):
        """Count Ind4s in the subgraph induced by verts."""
        n_v = len(verts)
        if n_v < 4:
            return 0
        count = 0
        for a in range(n_v):
            for b in range(a + 1, n_v):
                for c in range(b + 1, n_v):
                    for d in range(c + 1, n_v):
                        va, vb, vc, vd = verts[a], verts[b], verts[c], verts[d]
                        if not ((masks[va] >> vb) & 1) and not ((masks[va] >> vc) & 1) and not ((masks[va] >> vd) & 1) and \
                           not ((masks[vb] >> vc) & 1) and not ((masks[vb] >> vd) & 1) and \
                           not ((masks[vc] >> vd) & 1):
                            count += 1
        return count

    # Run greedy construction
    best_n = 37
    best_edges = []
    
    # Get seed edges
    for i in range(37):
        for j in range(i + 1, 37):
            if (j - i) % q in residues:
                best_edges.append((i, j))
                
    # Try greedy construction multiple times
    for _ in range(50):
        if time.time() - start_time > max_time:
            break
        n, adj = greedy_construct()
        if n > best_n:
            best_n = n
            best_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        best_edges.append((i, j))
                        
    return best_n, best_edges

# EVOLVE-BLOCK-END
