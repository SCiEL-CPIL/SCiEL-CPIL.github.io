import itertools
import random

def construct():
    n = 16
    points = list(range(n))
    
    # Generate all triples
    triples = list(itertools.combinations(points, 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all octahedra (K_{2,2,2}^{(3)})
    # An octahedron is defined by 6 points partitioned into 3 pairs.
    # The 8 edges are the transversals.
    octahedra = []
    edge_to_octahedra = [[] for _ in range(num_triples)]
    
    # Precompute pairings for 6 points
    # A set of 6 points {a,b,c,d,e,f} can be partitioned into 3 pairs in 15 ways.
    # We need to identify the 8 transversals for each partition.
    
    # To avoid duplicates and ensure we cover all, we iterate over all 6-subsets.
    for six_set in itertools.combinations(points, 6):
        # There are 15 partitions of 6 items into 3 pairs.
        # We can generate them by fixing the first element's partner.
        # Or just brute force all permutations? No, 15 is small.
        # Standard way: 
        # p0 is paired with p1.
        # Remaining 4: p2,p3,p4,p5.
        # p2 paired with p3, p4 with p5.
        # p2 paired with p4, p3 with p5.
        # p2 paired with p5, p3 with p4.
        # This generates 3 partitions for a fixed p0-p1 pair.
        # But p0 can be paired with any of the other 5.
        # So 5 * 3 = 15 partitions.
        
        # Let's generate the 15 partitions explicitly.
        partitions = []
        # Fix the smallest element index in six_set to avoid duplicate partitions?
        # No, we just need the set of 8 edges.
        # Different partitions yield different sets of 8 edges.
        
        # Generate all 15 pairings
        # Pair 0 with 1, 2, 3, 4, 5
        for i in range(1, 6):
            p0 = six_set[0]
            p1 = six_set[i]
            remaining = [six_set[j] for j in range(6) if j != 0 and j != i]
            # Pair remaining 4
            # r0 with r1, r2 with r3
            # r0 with r2, r1 with r3
            # r0 with r3, r1 with r2
            pairs1 = [(p0, p1), (remaining[0], remaining[1]), (remaining[2], remaining[3])]
            pairs2 = [(p0, p1), (remaining[0], remaining[2]), (remaining[1], remaining[3])]
            pairs3 = [(p0, p1), (remaining[0], remaining[3]), (remaining[1], remaining[2])]
            
            for pairs in [pairs1, pairs2, pairs3]:
                # Generate 8 transversals
                # pairs = [(a1,a2), (b1,b2), (c1,c2)]
                a1, a2 = pairs[0]
                b1, b2 = pairs[1]
                c1, c2 = pairs[2]
                
                edges = [
                    tuple(sorted((a1, b1, c1))),
                    tuple(sorted((a1, b1, c2))),
                    tuple(sorted((a1, b2, c1))),
                    tuple(sorted((a1, b2, c2))),
                    tuple(sorted((a2, b1, c1))),
                    tuple(sorted((a2, b1, c2))),
                    tuple(sorted((a2, b2, c1))),
                    tuple(sorted((a2, b2, c2)))
                ]
                
                # Convert to indices
                edge_indices = [triple_index[e] for e in edges]
                octahedra.append(edge_indices)
                
                # Update edge_to_octahedra
                for idx in edge_indices:
                    edge_to_octahedra[idx].append(len(octahedra) - 1)
                    
    num_octahedra = len(octahedra)
    
    # Initialize state
    # active_edges: set of indices of triples currently in the solution
    active_edges = set(range(num_triples))
    
    # oct_count: number of active edges in each octahedron
    oct_count = [8] * num_octahedra
    
    # full_octahedra: set of indices of octahedra that are currently full (count 8)
    full_octahedra = set(range(num_octahedra))
    
    # We will remove edges until no full octahedra remains.
    # To speed up, we maintain a list of full octahedra.
    # Since we modify it, we use a list and pop.
    # But we need to handle the case where an octahedron becomes full again?
    # No, we only remove edges, so counts only decrease.
    # So an octahedron is full initially, and may become non-full later.
    # It never becomes full again.
    
    full_list = list(range(num_octahedra))
    
    while full_list:
        # Pick a full octahedron
        # To be efficient, pick the first one
        o_idx = full_list.pop()
        
        # Check if it's still full (it might have been updated by previous removals 
        # if we didn't remove it from full_list when it became non-full? 
        # Wait, we only pop from full_list when we process it.
        # But other removals might have made it non-full before we popped it?
        # Yes. So we must check oct_count[o_idx] == 8.
        while full_list and oct_count[full_list[-1]] < 8:
            full_list.pop()
            
        if not full_list:
            break
            
        o_idx = full_list[-1] # Peek first to check, or just pop and check
        
        # Let's refine the pop logic:
        # We want to pick ANY full octahedron.
        # We can just iterate through full_list and find the first one with count 8.
        # But full_list might contain stale entries.
        # Better: Use a set for full_octahedra, and iterate?
        # Given the small number of iterations (~100), we can just scan the full_octahedra set.
        
        # Let's restart the loop logic with a set for full_octahedra
        # This is simpler to reason about.
        
        if not full_octahedra:
            break
            
        # Pick an octahedron from the set
        o_idx = next(iter(full_octahedra))
        
        # Get the 8 edges of this octahedron
        edges_in_o = octahedra[o_idx]
        
        # Choose which edge to remove
        # Heuristic: Remove the edge that is in the most OTHER full octahedra
        best_edge = -1
        max_hits = -1
        
        for e in edges_in_o:
            hits = 0
            # Check other octahedra containing e
            for other_o in edge_to_octahedra[e]:
                if other_o == o_idx:
                    continue
                if other_o in full_octahedra:
                    hits += 1
            if hits > max_hits:
                max_hits = hits
                best_edge = e
                
        # Remove best_edge
        active_edges.remove(best_edge)
        
        # Update counts for all octahedra containing best_edge
        for o in edge_to_octahedra[best_edge]:
            oct_count[o] -= 1
            if oct_count[o] == 7: # No longer full
                if o in full_octahedra:
                    full_octahedra.remove(o)
                    
    # Convert indices back to triples
    result = [list(triples[i]) for i in active_edges]
    return result
