import itertools
import random
import time

def construct():
    vertices = list(range(16))
    all_triples = list(itertools.combinations(vertices, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n = len(all_triples)
    
    # Build conflict graph
    # t and t' conflict if they are in the same K(2,2,2)
    # A K(2,2,2) is determined by 3 disjoint pairs of vertices.
    # Let the pairs be (a,b), (c,d), (e,f).
    # The 8 transversal triples are all combinations picking one from each pair.
    # e.g., (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f).
    
    neighbors = [set() for _ in range(n)]
    
    # Generate all K(2,2,2) instances
    # Choose 6 distinct vertices
    for six_verts in itertools.combinations(vertices, 6):
        # Partition into 3 pairs
        # There are 15 ways to partition 6 items into 3 pairs
        # We can generate all pairings
        v0, v1, v2, v3, v4, v5 = six_verts
        # Pairings:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # (0,1)(2,5)(3,4)
        # (0,2)(1,3)(4,5)
        # (0,2)(1,4)(3,5)
        # (0,2)(1,5)(3,4)
        # (0,3)(1,2)(4,5)
        # (0,3)(1,4)(2,5)
        # (0,3)(1,5)(2,4)
        # (0,4)(1,2)(3,5)
        # (0,4)(1,3)(2,5)
        # (0,4)(1,5)(2,3)
        # (0,5)(1,2)(3,4)
        # (0,5)(1,3)(2,4)
        # (0,5)(1,4)(2,3)
        
        pairings = [
            [(0,1), (2,3), (4,5)],
            [(0,1), (2,4), (3,5)],
            [(0,1), (2,5), (3,4)],
            [(0,2), (1,3), (4,5)],
            [(0,2), (1,4), (3,5)],
            [(0,2), (1,5), (3,4)],
            [(0,3), (1,2), (4,5)],
            [(0,3), (1,4), (2,5)],
            [(0,3), (1,5), (2,4)],
            [(0,4), (1,2), (3,5)],
            [(0,4), (1,3), (2,5)],
            [(0,4), (1,5), (2,3)],
            [(0,5), (1,2), (3,4)],
            [(0,5), (1,3), (2,4)],
            [(0,5), (1,4), (2,3)]
        ]
        
        for pairs in pairings:
            p1, p2, p3 = pairs
            # Generate the 8 triples
            triples_in_k = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = tuple(sorted((six_verts[a], six_verts[b], six_verts[c])))
                        triples_in_k.append(t)
            
            # All pairs of these 8 triples are conflicts
            for i in range(8):
                for j in range(i+1, 8):
                    t_i = triples_in_k[i]
                    t_j = triples_in_k[j]
                    idx_i = triple_to_idx[t_i]
                    idx_j = triple_to_idx[t_j]
                    neighbors[idx_i].add(idx_j)
                    neighbors[idx_j].add(idx_i)
    
    degrees = [len(neigh) for neigh in neighbors]
    
    best_size = 0
    best_set = set()
    
    # Run multiple restarts
    num_restarts = 200
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > 110:
            break
            
        available = set(range(n))
        current_set = set()
        
        # Initialize weights
        # Lower degree vertices are "better" to pick first as they block fewer others
        # We want to maximize the set, so we prefer vertices that don't kill many options.
        # Weight = 1 / (degree + 1)
        weights = [1.0 / (deg + 1) for deg in degrees]
        
        # Add small random noise to break ties and explore different paths
        for i in range(n):
            weights[i] += random.uniform(0, 0.01)
            
        while available:
            # Find vertex with max weight
            max_w = -1
            best_v = -1
            # Optimization: if available is large, iterating all is O(N).
            # N=560, so 560 * 560/2 operations roughly. 200 restarts * 560 steps * 560 checks = ~63M ops.
            # Python might be slow. Let's try to optimize the selection.
            # We can just iterate.
            for v in available:
                if weights[v] > max_w:
                    max_w = weights[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            current_set.add(best_v)
            available.discard(best_v)
            
            # Remove neighbors of best_v from available
            # And we don't need to update weights of remaining vertices in a complex way for basic greedy,
            # but the directive mentions "dynamic vertex weights updated by the frequency of conflicts".
            # A simple dynamic update: if a neighbor is removed, it's gone.
            # We can also penalize vertices that are "close" to many removed vertices?
            # For now, simple removal is the standard greedy MIS.
            # To add "dynamic" aspect, we can re-normalize or adjust based on remaining degree?
            # Let's stick to the weight defined at start but allow noise.
            # Actually, a better heuristic is to update the weight of remaining vertices
            # based on how many of their neighbors were just removed?
            # If many neighbors are removed, the vertex is "freed up" or "isolated"?
            # If a vertex's neighbors are removed, it doesn't conflict with them anymore (since they are out).
            # So its effective degree drops. Its "cost" drops. So its weight should go UP.
            # Let's implement this dynamic update.
            
            for u in neighbors[best_v]:
                if u in available:
                    # u is a neighbor of best_v. best_v is removed.
                    # u's degree in the *remaining* graph decreases by 1.
                    # Original degree: degrees[u].
                    # We can track remaining degree.
                    pass # We don't track remaining degree explicitly in the simple loop above.
            
            # To do this efficiently, we should track remaining degrees.
            # Let's refine the inner loop.
            
        # The above loop didn't update weights dynamically based on remaining degree.
        # Let's rewrite the inner loop to be more efficient and dynamic.
        
        # Re-doing the restart logic with dynamic weights
        available = set(range(n))
        current_set = set()
        remaining_degrees = degrees[:]
        weights = [1.0 / (deg + 1) for deg in degrees]
        for i in range(n):
            weights[i] += random.uniform(0, 0.05) # Slightly larger noise for diversity
        
        while available:
            # Select max weight
            max_w = -1
            best_v = -1
            for v in available:
                if weights[v] > max_w:
                    max_w = weights[v]
                    best_v = v
            
            if best_v == -1:
                break
            
            current_set.add(best_v)
            available.discard(best_v)
            
            # Update neighbors
            for u in neighbors[best_v]:
                if u in available:
                    # u loses a neighbor (best_v)
                    remaining_degrees[u] -= 1
                    # Update weight: new weight based on new degree
                    # We want to prefer vertices with lower remaining degree.
                    # Weight = 1 / (remaining_degree + 1)
                    # But we also have the initial noise. Let's just recompute base weight.
                    # To keep the noise effect, we can store the noise separately or just re-randomize slightly?
                    # Better: weights[u] = 1.0 / (remaining_degrees[u] + 1)
                    # This makes the choice purely based on current degree.
                    # This is a known heuristic for greedy MIS.
                    weights[u] = 1.0 / (remaining_degrees[u] + 1)
            
            # We don't update weights of non-neighbors.
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    return [list(all_triples[i]) for i in best_set]
