"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    
    Approach: 
    1. Generate all forbidden triples {a, b, c} with a, b, c distinct.
    2. Formulate as Minimum Hitting Set problem: Find smallest H such that 
       every forbidden triple intersects H.
    3. The solution is [1..180] \ H.
    4. Use Branch and Bound with greedy matching lower bound.
    """
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Generate Forbidden Triples
    # 1/a = 1/b + 1/c => a = bc / (b+c)
    # a, b, c distinct, 1 <= a,b,c <= N
    triples = set()
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted((a, b, c)))
                    triples.add(triple)
    
    # Convert to list of masks for fast intersection checks
    # Vertex i (1-indexed) corresponds to bit (i-1)
    edge_masks = []
    for t in triples:
        mask = 0
        for x in t:
            mask |= (1 << (x - 1))
        edge_masks.append(mask)
    
    # If no triples, return all
    if not edge_masks:
        return numbers

    # 2. Branch and Bound for Minimum Hitting Set
    best_hit_size = N # Upper bound
    best_hit_set = set()
    
    # Precompute vertex frequencies to order edges? 
    # Or just use the list.
    
    # To speed up, we can sort edges by some heuristic.
    # Let's count degree of each vertex.
    deg = [0] * (N + 1)
    for mask in edge_masks:
        for x in range(1, N + 1):
            if mask & (1 << (x - 1)):
                deg[x] += 1

    # Sort edges: prefer edges with high-degree vertices?
    # Actually, for hitting set, we want to pick vertices that hit many edges.
    # In B&B, we branch on an edge. The choice of edge affects branching.
    # Let's stick to a simple order, but maybe shuffle if random tie-breaks help?
    # Deterministic is better for reproducibility, but randomization can escape local minima in B&B?
    # B&B is exact (if completes), so order only affects speed.
    
    # Heuristic: Sort edges such that those with "rarest" vertices come first?
    # Or "most connected"?
    # Let's try sorting by the sum of degrees of its vertices.
    def edge_weight(mask):
        w = 0
        for x in range(1, N + 1):
            if mask & (1 << (x - 1)):
                w += deg[x]
        return w
        
    edge_masks.sort(key=edge_weight, reverse=True)

    # Greedy Matching Lower Bound
    # Returns a lower bound on the number of additional vertices needed to hit 'remaining_edges'
    def greedy_matching_lb(remaining_edges):
        # Find a maximal set of disjoint edges
        # This is an approximation of the size of a minimum hitting set for the subgraph
        # Actually, if we have k disjoint edges, we need at least k vertices.
        # Greedy: pick an edge, remove all intersecting edges, count 1.
        # This gives a valid lower bound? 
        # Yes, because any hitting set must pick at least one vertex from each disjoint edge.
        # So the size of a maximal matching is a lower bound on the size of the minimum hitting set.
        
        # To make it stronger, we can use a greedy algorithm that picks the vertex
        # that hits the most remaining edges, adds it to H, and removes all edges it hits.
        # The number of vertices picked is an UPPER bound on the minimum hitting set?
        # No, Greedy Hitting Set is an approximation algorithm, so it gives an Upper Bound on the Min Hitting Set.
        # We need a Lower Bound for pruning.
        # Lower Bound: Maximal Matching Size.
        
        # Let's compute Maximal Matching Size.
        # Since N is small and edges are few, we can do this quickly.
        # However, doing this at every node is expensive.
        # We can do a simpler LB: count of edges / max_degree_in_subgraph? No.
        # Let's use the disjoint edge count.
        
        # Optimization: Cache? Hard with dynamic lists.
        # Let's just count disjoint edges greedily.
        # To make it fast, we can just pick the first available edge, mark its vertices, 
        # and skip any edge that intersects them.
        
        used_vertices = 0
        count = 0
        # Iterate through remaining edges
        # This is O(E * V) worst case, but E is small (~500).
        # At deep nodes, E is small.
        for mask in remaining_edges:
            if mask & used_vertices == 0:
                count += 1
                used_vertices |= mask
        return count

    def search(current_mask, current_size, remaining_edges):
        nonlocal best_hit_size, best_hit_set
        
        # Pruning 1: If current size + LB >= best, prune
        # We need a fast LB. 
        # If remaining_edges is empty, LB = 0.
        # Else, LB = greedy_matching_lb(remaining_edges)
        
        if not remaining_edges:
            if current_size < best_hit_size:
                best_hit_size = current_size
                # Reconstruct set? 
                # We can just track the mask.
                best_hit_set_mask = current_mask
            return

        lb = greedy_matching_lb(remaining_edges)
        if current_size + lb >= best_hit_size:
            return

        # Pick an edge to branch on
        # Heuristic: Pick the edge that appears in the most remaining edges?
        # Or just the first one?
        # Let's pick the one with the highest "impact".
        # For simplicity, pick the first one in the sorted list.
        # But we can also pick the one with the most frequent vertex in remaining_edges.
        
        # Let's just pick the first one.
        edge = remaining_edges[0]
        
        # Branch on each vertex in the edge
        # To improve, try vertices in order of degree (descending)
        # Vertices in edge:
        verts = []
        for x in range(1, N + 1):
            if edge & (1 << (x - 1)):
                verts.append(x)
        
        # Sort by degree descending
        verts.sort(key=lambda x: deg[x], reverse=True)
        
        for v in verts:
            v_mask = 1 << (v - 1)
            new_mask = current_mask | v_mask
            new_size = current_size + 1
            
            # Filter remaining edges: remove those hit by v
            new_remaining = []
            for em in remaining_edges:
                if not (em & v_mask):
                    new_remaining.append(em)
            
            search(new_mask, new_size, new_remaining)
            
            # Early termination? If we found a solution of size 1, can't do better.
            # But we want min size.
            
    # Initial call
    # We can use a greedy initialization to get a good initial upper bound (best_hit_size)
    # Greedy Hitting Set:
    # While edges remain:
    #   Pick vertex v that hits most remaining edges
    #   Add v to H
    #   Remove all edges hit by v
    
    greedy_H_mask = 0
    greedy_H_size = 0
    rem_edges = edge_masks[:]
    
    # Precompute for each vertex, which edges it hits?
    # Or just iterate.
    
    while rem_edges:
        # Find vertex with max coverage
        best_v = -1
        best_count = -1
        # Count coverage for all vertices? O(N * E)
        # N=180, E~500. 90k ops. Fine.
        counts = [0] * (N + 1)
        for em in rem_edges:
            for x in range(1, N + 1):
                if em & (1 << (x - 1)):
                    counts[x] += 1
        
        for x in range(1, N + 1):
            if counts[x] > best_count:
                best_count = counts[x]
                best_v = x
                
        if best_v == -1:
            break
            
        greedy_H_mask |= (1 << (best_v - 1))
        greedy_H_size += 1
        
        # Remove hit edges
        v_mask = 1 << (best_v - 1)
        rem_edges = [em for em in rem_edges if not (em & v_mask)]
        
    best_hit_size = greedy_H_size
    best_hit_set_mask = greedy_H_mask

    # Run B&B
    start_time = time.time()
    # Use a time limit inside search? 
    # Python recursion depth? 
    # Depth is at most N. 180 is safe.
    
    # To handle timeout, we can't easily interrupt recursion in Python without signals.
    # We can check time in the search function.
    
    original_search = search
    def search_with_timeout(current_mask, current_size, remaining_edges):
        if time.time() - start_time > 55: # Leave 5s for final processing
            return
        original_search(current_mask, current_size, remaining_edges)
        
    # Replace search
    # Actually, nonlocal variable modification in nested function with timeout check is messy.
    # Let's just put the time check in the main loop or accept the risk.
    # Given N=180 and pruning, it should be fast.
    
    search(0, 0, edge_masks)
    
    # Construct result
    # H is the hitting set. S is the complement.
    # We want to return S.
    result_mask = ((1 << N) - 1) & ~best_hit_set_mask
    result = []
    for i in range(1, N + 1):
        if result_mask & (1 << (i - 1)):
            result.append(i)
            
    # Ensure nonempty
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
